#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import json

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)

from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.dataset.latent_datasets import LatentDataset_LR, latent_collate_function_LR
from fastvideo.models.hunyuan.inference import HunyuanVideoSampler, ShardedHunyuanVideoSampler
from fastvideo.utils.load import load_transformer
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)


def initialize_distributed():
    """Initialize distributed environment for inference."""
    # If not using torchrun, manually set environment variables
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_NODELIST", "localhost").split()[0]
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Initializing distributed: rank={local_rank}, world_size={world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    initialize_sequence_parallel_state(world_size)

    return rank, world_size


def load_hunyuan_sharded_fsdp(model_root_path, rank, checkpoint_dir, args):
    """
    Load a HunyuanVideoSampler with FSDP support, where each shard is loaded to
    its corresponding rank. This function handles models that are wrapped with
    Fully Sharded Data Parallel (FSDP).

    Args:
        model_root_path (Path or str): Root path to the model files (VAE, text encoder, etc.)
        checkpoint_dir (Path or str): Directory containing the sharded model checkpoints
                                     (model_shard_0.pth, model_shard_1.pth, etc.)
        args (argparse.Namespace): Configuration arguments

    Returns:
        HunyuanVideoSampler: Loaded sampler with the shard for the current rank
    """
    import os
    import torch
    import torch.distributed as dist
    from pathlib import Path
    import copy
    from fastvideo.models.hunyuan.inference import HunyuanVideoSampler

    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))

    # Set the device for this rank
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(local_rank)

    # Ensure distributed is initialized
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl")
        print(f"Distributed initialized: rank={rank}/{world_size}, rank={rank}")

    # Initialize FSDP
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload
    )
    from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs

    # Convert paths to Path objects
    model_root_path = Path(model_root_path)
    checkpoint_dir = Path(checkpoint_dir)

    # Path to the specific shard for this rank
    shard_path = os.path.join(checkpoint_dir, f"model_shard_{rank}.pth")

    # Check if the shard exists
    # if not shard_path.exists():
    #     raise FileNotFoundError(f"Shard not found at {shard_path} for rank {local_rank}")

    print(f"Rank {rank}/{world_size} loading shard: {shard_path}")

    # Create a copy of args to avoid modifying the original
    shard_args = copy.deepcopy(args)

    # Store the original dit_weight for reference
    original_dit_weight = shard_args.dit_weight

    # Set the shard path in the args
    shard_args.dit_weight = str(shard_path)

    # Disable gradient computation for inference
    torch.set_grad_enabled(False)

    # Load VAE and related components
    from fastvideo.models.hunyuan.vae import load_vae
    vae, _, s_ratio, t_ratio = load_vae(
        shard_args.vae,
        shard_args.vae_precision,
        device=device if not shard_args.use_cpu_offload else "cpu",
    )
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # Load text encoder
    from fastvideo.models.hunyuan.constants import PROMPT_TEMPLATE
    from fastvideo.models.hunyuan.text_encoder import TextEncoder

    # Get prompt templates
    if shard_args.prompt_template_video is not None:
        crop_start = PROMPT_TEMPLATE[shard_args.prompt_template_video].get("crop_start", 0)
    elif shard_args.prompt_template is not None:
        crop_start = PROMPT_TEMPLATE[shard_args.prompt_template].get("crop_start", 0)
    else:
        crop_start = 0

    max_length = shard_args.text_len + crop_start

    prompt_template = (PROMPT_TEMPLATE[shard_args.prompt_template]
                       if shard_args.prompt_template is not None else None)

    prompt_template_video = (PROMPT_TEMPLATE[shard_args.prompt_template_video]
                             if shard_args.prompt_template_video is not None else None)

    text_encoder = TextEncoder(
        text_encoder_type=shard_args.text_encoder,
        max_length=max_length,
        text_encoder_precision=shard_args.text_encoder_precision,
        tokenizer_type=shard_args.tokenizer,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=shard_args.hidden_state_skip_layer,
        apply_final_norm=shard_args.apply_final_norm,
        reproduce=shard_args.reproduce,
        device=device if not shard_args.use_cpu_offload else "cpu",
    )

    text_encoder_2 = None
    if shard_args.text_encoder_2 is not None:
        text_encoder_2 = TextEncoder(
            text_encoder_type=shard_args.text_encoder_2,
            max_length=shard_args.text_len_2,
            text_encoder_precision=shard_args.text_encoder_precision_2,
            tokenizer_type=shard_args.tokenizer_2,
            reproduce=shard_args.reproduce,
            device=device if not shard_args.use_cpu_offload else "cpu",
        )

    # Load the model
    from fastvideo.models.hunyuan.constants import PRECISION_TO_TYPE
    from fastvideo.models.hunyuan.modules import load_model

    factor_kwargs = {
        "device": device,
        "dtype": PRECISION_TO_TYPE[shard_args.precision]
    }
    in_channels = shard_args.latent_channels
    out_channels = shard_args.latent_channels

    # Create base model (unwrapped)
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32
    )

    # Move model to device before FSDP wrapping
    # transformer = transformer.to(device)

    # Need to handle FSDP kwargs properly based on the actual function signature
    from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs

    # Set sharding strategy
    sharding_strategy = "full"  # or use args.sharding_strategy if available

    # Call get_dit_fsdp_kwargs with required parameters
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        args=shard_args,
        transformer=transformer,
        sharding_strategy=sharding_strategy,
        use_lora=False,  # Set based on your needs
        cpu_offload=shard_args.use_cpu_offload,
        master_weight_type="fp32"  # Or adjust based on precision
    )
    if args.model_type == "hunyuan_controlnet":
        transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)
    # print(transformer)
    print(f"Using auto_wrap_policy: {fsdp_kwargs['auto_wrap_policy']}")

    # Wrap model with FSDP
    print(f"Rank {local_rank}: Wrapping model with FSDP")
    model = FSDP(
        transformer,
        **fsdp_kwargs
    )

    # Load the state dict for this shard
    print(f"Rank {local_rank}: Loading state dict from {shard_path}")
    checkpoint = torch.load(shard_path, map_location=device)

    if rank == 0:
        # Print checkpoint structure
        print(f"Rank 0: Checkpoint contains {len(checkpoint.keys())} keys")
        for i, key in enumerate(list(checkpoint.keys())):
            print(f"  - {key}")

        print(f"Rank 0: Model contains {sum(1 for _ in model.named_parameters())} parameters")
        for i, (name, _) in enumerate(model.named_parameters()):
            print(f"  - {name}")

    # Direct loading approach for sharded checkpoints
    try:
        print(f"Rank {local_rank}: Attempting to load checkpoint directly")
        # FSDP models may require unwrapping before loading
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            local_state_dict = {k: v for k, v in checkpoint.items() if not k.endswith('.local_shards')}
            model.load_state_dict(local_state_dict, strict=False)
    except Exception as e:
        print(f"Rank {local_rank}: Direct loading failed: {str(e)}")

        # Fallback to flat parameter loading approach
        print(f"Rank {local_rank}: Falling back to flat parameter loading")
        flat_params = {}

        # First try to load directly to the wrapped model parameters
        for name, param in model.named_parameters():
            param_name = name
            # Remove '_fsdp_wrapped_module.' prefix if present
            if '_fsdp_wrapped_module.' in param_name:
                param_name = param_name.replace('_fsdp_wrapped_module.', '')

            if param_name in checkpoint:
                if param.shape == checkpoint[param_name].shape:
                    flat_params[name] = checkpoint[param_name]
                else:
                    print(
                        f"Rank {local_rank}: Shape mismatch for {param_name}: {param.shape} vs {checkpoint[param_name].shape}")

        if flat_params:
            # Try loading the collected flat parameters
            try:
                model.load_state_dict(flat_params, strict=False)
                print(f"Rank {local_rank}: Successfully loaded {len(flat_params)} parameters")
            except Exception as e:
                print(f"Rank {local_rank}: Flat parameter loading failed: {str(e)}")

                # Last resort: load directly to the model's parameter data
                for name, param in model.named_parameters():
                    param_name = name.replace('_fsdp_wrapped_module.', '')
                    if param_name in checkpoint:
                        if param.shape == checkpoint[param_name].shape:
                            param.data.copy_(checkpoint[param_name])
                        else:
                            print(f"Rank {local_rank}: Shape mismatch, skipping {param_name}")

    model.eval()

    # Synchronize before creating the sampler
    if world_size > 1:
        dist.barrier()

    # Create and return the HunyuanVideoSampler
    sampler = HunyuanVideoSampler(
        args=shard_args,
        vae=vae,
        vae_kwargs=vae_kwargs,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        model=model,
        use_cpu_offload=shard_args.use_cpu_offload,
        device=device,
    )

    # Restore original dit_weight path in args
    shard_args.dit_weight = original_dit_weight

    # Synchronize all processes again
    if world_size > 1:
        dist.barrier()

    print(f"Rank {local_rank}/{world_size} successfully loaded its model shard")
    return sampler


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment
    rank, world_size = initialize_distributed()
    print(f"Process {rank}/{world_size} initialized")

    # Setup model path and check if exists
    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create output directory
    save_path = args.output_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Modify the dit_weight path to directly point to the shard for this rank
    # checkpoint_dir = args.dit_weight
    # shard_path = os.path.join(checkpoint_dir, f"model_shard_{local_rank}.pth")
    #
    # # Check if shard exists
    # if not os.path.exists(shard_path):
    #     raise FileNotFoundError(f"Checkpoint shard not found at {shard_path}")
    # print(f"Rank {local_rank} using shard: {shard_path}")
    #
    # # Override the dit_weight with the specific shard path
    # args.dit_weight = shard_path

    # Load the model directly with the HunyuanVideoSampler
    # Since we're passing the specific shard path for this rank,
    # each sampler will load its own part of the model
    print(f"Rank {rank} loading model...")
    hunyuan_video_sampler = load_hunyuan_sharded_fsdp(models_root_path, rank, args.dit_weight, args)

    # Synchronize after loading
    dist.barrier()

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Only rank 0 handles the dataset and inference
    if rank == 0:
        with open(args.prompt) as f:
            prompts = f.readlines()

        # add lr_latents:
        train_dataset = LatentDataset_LR("/scracth/10320/lanqing001/xinrui/FastVideo/data/Inte4K/videos2caption.json",
                                         16, 0.0)

        sampler = (LengthGroupedSampler(
            1,  # batch size
            rank=0,
            world_size=1,
            lengths=train_dataset.lengths,
            group_frame=args.group_frame,
            group_resolution=args.group_resolution,
        ) if (args.group_frame or args.group_resolution) else DistributedSampler(
            train_dataset, rank=0, num_replicas=1, shuffle=False))

        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=latent_collate_function_LR,
            pin_memory=True,
            batch_size=1,
            num_workers=8,
            drop_last=True,
            shuffle=False,
        )

        for i, batch in enumerate(train_dataloader):
            if i > 2:  # Process only the first 3
                break

            latent, lr_latents, prompt_embed, prompt_attention_mask, _ = batch
            lr_latents = lr_latents.cuda()
            prompt = prompts[i].strip()

            print(f"Processing prompt: {prompt}")

            # Run inference
            outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                lr_latents=lr_latents,
                height=args.height,
                width=args.width,
                video_length=args.num_frames,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
            )

            # Process high-resolution samples
            high_res_video = outputs["samples"]
            high_res_frames = []
            frames_high = rearrange(high_res_video, "b c t h w -> t b c h w")
            for frame in frames_high:
                frame_grid = torchvision.utils.make_grid(frame, nrow=6)
                frame_grid = frame_grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                high_res_frames.append((frame_grid * 255).numpy().astype(np.uint8))

            # Save the high-resolution video
            output_filename = os.path.join(os.path.dirname(args.output_path),
                                           f"{i}_{prompt[:20].replace(' ', '_')}.mp4")
            imageio.mimsave(output_filename, high_res_frames, fps=args.fps)
            print(f"Saved high-res video to {output_filename}")

            if outputs.get("samples_lr", 0) != 0:
                # Process low-resolution samples
                low_res_video = outputs["samples_lr"]
                low_res_frames = []
                frames_low = rearrange(low_res_video, "b c t h w -> t b c h w")
                for frame in frames_low:
                    frame_grid = torchvision.utils.make_grid(frame, nrow=6)
                    frame_grid = frame_grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    low_res_frames.append((frame_grid * 255).numpy().astype(np.uint8))

                # Save the low-resolution video
                low_res_filename = os.path.join(os.path.dirname(args.output_path),
                                                f"{i}_{prompt[:20].replace(' ', '_')}_low_res.mp4")
                imageio.mimsave(low_res_filename, low_res_frames, fps=args.fps)
                print(f"Saved low-res video to {low_res_filename}")

    # Wait for all processes to complete
    dist.barrier()
    print(f"Rank {local_rank} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)

    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed for evaluation.")
    parser.add_argument("--neg_prompt",
                        type=str,
                        default=None,
                        help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift",
                        type=int,
                        default=7,
                        help="Flow shift parameter.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help=
        "Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hunyuan_controlnet",
        help=
        "model type",
    )

    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        required=True,
        help="Path to checkpoint directory containing model_shard_X.pth files",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help=
        "Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help=
        "Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver",
                        type=str,
                        default="euler",
                        help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision",
                        type=str,
                        default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--rope-theta",
                        type=int,
                        default=256,
                        help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision",
                        type=str,
                        default="fp16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--vae-sp", action="store_true", default=False)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template",
                        type=str,
                        default="dit-llm-encode")
    parser.add_argument("--prompt-template-video",
                        type=str,
                        default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")

    args = parser.parse_args()
    # process for vae sequence parallel
    if args.vae_sp and not args.vae_tiling:
        raise ValueError(
            "Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True."
        )
    main(args)