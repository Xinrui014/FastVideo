import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.models.hunyuan.constants import (NEGATIVE_PROMPT,
                                                PRECISION_TO_TYPE,
                                                PROMPT_TEMPLATE)
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline, HunyuanVideoPipeline_LR
from fastvideo.models.hunyuan.diffusion.schedulers import \
    FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.utils.parallel_states import nccl_info


class Inference(object):

    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.device = (device if device is not None else
                       "cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.parallel_args = parallel_args

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        args,
                        device=None,
                        **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================
        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        # ==================== Initialize Distributed Environment ================
        if nccl_info.sp_size > 1:
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        parallel_args = None  # {"ulysses_degree": args.ulysses_degree, "ring_degree": args.ring_degree}

        # ======================== Get the args path =============================

        # Disable gradient
        torch.set_grad_enabled(False)

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {
            "device": device,
            "dtype": PRECISION_TO_TYPE[args.precision]
        }
        in_channels = args.latent_channels
        out_channels = args.latent_channels

        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )
        model = model.to(device)
        model = Inference.load_state_dict(args, model, pretrained_model_path)
        model.eval()

        # ============================= Build extra models ========================
        # VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            args.vae_precision,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
                "crop_start", 0)
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get(
                "crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template
        prompt_template = (PROMPT_TEMPLATE[args.prompt_template]
                           if args.prompt_template is not None else None)

        # prompt_template_video
        prompt_template_video = (PROMPT_TEMPLATE[args.prompt_template_video]
                                 if args.prompt_template_video is not None else
                                 None)

        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args,
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path):
        load_key = args.load_key
        dit_weight = Path(args.dit_weight)

        if dit_weight is None:
            model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
            files = list(model_dir.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {model_dir}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [
                    f for f in files if str(f).endswith("_model_states.pt")
                ]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {dit_weight}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        else:
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith("pytorch_model_"):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith("_model_states.pt") for f in files):
                    files = [
                        f for f in files if str(f).endswith("_model_states.pt")
                    ]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(
                            f"Multiple model weights found in {dit_weight}, using {model_path}"
                        )
                    bare_model = False
                else:
                    raise ValueError(
                        f"Invalid model path: {dit_weight} with unrecognized weight format: "
                        f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                        f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                        f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                        f"specific weight file, please provide the full path to the file."
                    )
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = "unknown"
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")


        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        if model_path.suffix == ".safetensors":
            # Use safetensors library for .safetensors files
            state_dict = safetensors_load_file(model_path)
        elif model_path.suffix == ".pt":
            # Use torch for .pt files
            state_dict = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
        elif model_path.suffix == ".pth":
            # Use torch for .pt files
            state_dict = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
        else:
            raise ValueError(f"Unsupported file format: {model_path}")

        if bare_model == "unknown" and ("ema" in state_dict
                                        or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}.")
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):

    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None,
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args,
        )

        self.pipeline= self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline_LR(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline

    @torch.no_grad()
    def predict(
        self,
        prompt,
        lr_latents=None,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
        """

        out_dict = dict()

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
        ]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver,
        )
        self.pipeline.scheduler = scheduler

        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8,
                            width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8,
                            width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.debug(debug_str)

        print("work at HunyuanVideoSampler")

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        samples_lr = [0]
        start_time = time.time()
        if lr_latents is None:
            samples_lr, lr_latents = self.pipeline(
                prompt=prompt,
                height=target_height,
                width=target_width,
                video_length=target_video_length,
                num_inference_steps=infer_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                output_type="pil",
                n_tokens=n_tokens,
                embedded_guidance_scale=embedded_guidance_scale,
                data_type="video" if target_video_length > 1 else "image",
                is_progress_bar=True,
                vae_ver=self.args.vae,
                enable_tiling=self.args.vae_tiling,
                enable_vae_sp=self.args.vae_sp,
            )
            b, s, c, h, w = lr_latents.shape
            lr_latents_reshaped = lr_latents.view(b * s, c, h, w)
            lr_latents_resized = F.interpolate(lr_latents_reshaped, size=(136, 240), mode='bilinear', align_corners=False)
            lr_latents = lr_latents_resized.view(b, s, c, 136, 240)

        samples, latents = self.pipeline(
            prompt=prompt,
            lr_fea=lr_latents,
            strength=0.6,
            height=1088,
            width=1920,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            enable_vae_sp=self.args.vae_sp,
        )
        out_dict["samples_lr"] = samples_lr[0]
        out_dict["samples"] = samples[0]
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict


import os
import torch
from pathlib import Path


class ShardedHunyuanVideoSampler(HunyuanVideoSampler):
    """
    Extension of HunyuanVideoSampler to support loading from sharded FSDP checkpoints.
    """

    @classmethod
    def from_pretrained_sharded(cls,
                                pretrained_model_path,
                                args,
                                device=None,
                                **kwargs):
        """
        Initialize the Inference pipeline with sharded FSDP checkpoints.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path for base model
            args (argparse.Namespace): The arguments for the pipeline
            device (int): The device for inference. Default is current CUDA device.
        """
        # Get local rank and world size for distributed loading
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Ensure we're using the correct device
        if device is None:
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(local_rank)

        # Disable gradient
        torch.set_grad_enabled(False)

        # # Create a modified args object with the original checkpoint path
        # checkpoint_dir = args.dit_weight
        # original_dit_weight = args.dit_weight
        #
        # # Verify that we have a directory with sharded checkpoints
        # if not os.path.isdir(checkpoint_dir):
        #     raise ValueError(f"Expected checkpoint directory, got: {checkpoint_dir}")
        #
        # # Check if shard exists for this rank
        # shard_path = os.path.join(checkpoint_dir, f"model_shard_{local_rank}.pth")
        # if not os.path.exists(shard_path):
        #     raise FileNotFoundError(f"Checkpoint shard not found at {shard_path}")
        #
        # # Modify args to use the specific shard path for this rank
        # args.dit_weight = shard_path

        # Load the model using the standard from_pretrained method
        # This will load only the shard for this rank
        instance = super().from_pretrained(
            pretrained_model_path,
            args=args,
            device=device,
            **kwargs
        )

        # Restore the original dit_weight value for reference
        # args.dit_weight = original_dit_weight

        # Return the instance with the sharded model loaded
        return instance

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path):
        """
        Override to load a specific shard based on local rank.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Check if we're loading a shard
        if "model_shard_" in str(args.dit_weight):
            print(f"Rank {local_rank} loading shard: {args.dit_weight}")

            # Load the shard
            state_dict = torch.load(
                args.dit_weight,
                map_location=lambda storage, loc: storage
            )

            # When loading shards, we need different handling
            print(f"Shard keys: {list(state_dict.keys())[:5]}...")

            # Load the state dict into model
            model.load_state_dict(state_dict, strict=False)

            return model
        else:
            # If not a shard, use the original loading method
            return super().load_state_dict(args, model, pretrained_model_path)
