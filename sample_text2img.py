# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from transformers.utils import ContextManagers
from download import find_model
from models import DiT_models
import argparse
import pandas as pd
import torch.distributed as dist
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):

    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")



    if args.ckpt is None:
        raise ValueError(
            f"--ckpt' checkpoint value '{args.ckpt}' is missing"
        )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=args.tokenizer_path
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=args.text_encoder_path
    ).to(device)

    text_encoder_dim = text_encoder.config.hidden_size

    text_encoder.requires_grad_(False)
    
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_encoder_dim = text_encoder_dim
    ).to(device)


    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(text_prompts, is_train=True):
        captions = []
        for caption in text_prompts:
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise ValueError(
                    f"Caption column `{text_prompts}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding=True, truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids


    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt #or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    sample_csv_folder_dir = f"{args.sample_dir}/csv/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(sample_csv_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    # iterations = int(samples_needed_this_gpu // n)


    def preprocess_text(examples):
        examples['text'] = examples['text']
        examples["input_ids"] = tokenize_captions(examples['text'])
        if examples['target']:
            examples['target'] = examples['target']
        examples['skin_tone'] = examples['skin_tone']
        return examples

    dataset = load_dataset("results/204-DiT-L-4", data_files="isic_llava_med_subgroups.csv")
    # dataset = load_dataset("data/images/isic_all/all", data_files="metadata.csv")
    dataset_len = len(dataset['train'])
    print("total samples: ", dataset_len)

    text_prompts = dataset['train'].with_transform(preprocess_text)


    def collate_fn(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        text_prompt = [example['text'] for example in examples]
        target = [example['target'] for example in examples]
        skin_tone = [example['skin_tone'] for example in examples]
        return {"input_ids": input_ids, "text": text_prompt, "target": target, "skin_tone":skin_tone}
    
    sampler = DistributedSampler(
        text_prompts,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )

    loader = DataLoader(
        text_prompts,
        batch_size=int(n),
        shuffle=False,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=4
    )

    pbar = loader
    pbar = tqdm(loader) if rank == 0 else pbar
    total = 70000

    with open(f"{sample_csv_folder_dir}/new2_sample_{rank}.csv", 'w') as csv_file:
                csv_file.write(f"text;file_name;target;image_name;skin_tone\n")

    # inception_dim = 2048
    # inception_model = fid_score.get_inception_model(n, device, dims=inception_dim)

    # pred_arr = np.empty((dataset_len, inception_dim))

    # start_idx = 0 + rank

    for batch in pbar:
        # Sample inputs:
        k = len(batch["input_ids"])
        z = torch.randn(k, model.in_channels, latent_size, latent_size, device=device)
        # y = torch.randint(0, args.num_classes, (n,), device=device)
        y = batch["input_ids"].to(device)
        y = text_encoder(y)[0]  # encoder_hidden_states


        model_kwargs = dict(y=y)
        sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1)

        


        # Save samples to disk as individual .png files

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            with open(f"{sample_csv_folder_dir}/new2_sample_{rank}.csv", 'a') as csv_file:
                # csv_file.write(f"{batch['text'][i]};{index:06d}.png;{batch['malignant'][i]}\n")
                csv_file.write(f"{batch['text'][i]};{index:06d}.png;{batch['target'][i]};{index:06d}.png;{batch['skin_tone'][i]}\n")
        total += global_batch_size
        # if total>100:
            # break

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
    #     create_npz_from_sample_folder(sample_folder_dir, len(dataset))
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

# 2500000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--tokenizer_path", type=str, default="results/tokenizer")
    parser.add_argument("--text_encoder_path", type=str, default="results/text_encoder")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--sample-dir", type=str, default="results/204-DiT-L-4/0460000/new2")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # torchrun --nnodes=1 --nproc_per_node=2 sample_text2img_fid.py --model DiT-L/4 --ckpt results/202-DiT-L-4/checkpoints/0420000.pt
    # torchrun --nnodes=1 --nproc_per_node=2 sample_text2img_fid.py --model DiT-L/4 --ckpt results/204-DiT-L-4/checkpoints/0460000.pt

    args = parser.parse_args()
    main(args)
