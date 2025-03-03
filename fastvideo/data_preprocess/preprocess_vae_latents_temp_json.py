import argparse
import json
import os

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.dataset import getdataset
from fastvideo.utils.load import load_vae

logger = get_logger(__name__)


def main(args):
    # 如果没有使用 torchrun，而是用 srun，则手动设置分布式环境变量
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
    if "MASTER_ADDR" not in os.environ:
        # 从 SLURM_NODELIST 中取第一个节点作为 master
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_NODELIST", "127.0.0.1").split()[0]
    if "MASTER_PORT" not in os.environ:
        # 指定一个端口，确保该端口在所有节点上都可以使用
        os.environ["MASTER_PORT"] = "29500"

    # 读取环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    print("world_size", world_size, "local rank", local_rank)
    train_dataset = getdataset(args)
    sampler = DistributedSampler(train_dataset,
                                 rank=global_rank,
                                 num_replicas=world_size,
                                 shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    encoder_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=global_rank)
    # vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    # vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

    video_base_path = "/scratch/10320/lanqing001/xinrui/dataset/raw_mp4"
    json_data = []
    for _, data in tqdm(enumerate(train_dataloader),
                        disable=local_rank != 0,
                        total=len(train_dataloader),
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        ):
        with torch.inference_mode():
            # with torch.autocast("cuda", dtype=autocast_type):
            #     latents = vae.encode(data["pixel_values"].to(
            #         encoder_device))["latent_dist"].sample()
            for idx, video_path in enumerate(data["path"]):
                # video_name = os.path.basename(video_path).split(".")[0]
                # latent_path = os.path.join(args.output_dir, "latent",
                #                            video_name + ".pt")
                # torch.save(latents[idx].to(torch.bfloat16), latent_path)
                # item = {}
                # item["length"] = latents[idx].shape[1]
                # item["latent_path"] = video_name + ".pt"
                # item["caption"] = data["text"][idx]
                # json_data.append(item)
                # print(f"{video_name} processed")
                relative_path = os.path.relpath(video_path, video_base_path)
                # 去掉扩展名得到相对名称，如 "9/xxxx"
                relative_name = os.path.splitext(relative_path)[0]
                latent_path = os.path.join(args.output_dir, "latent", relative_name + ".pt")
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)

                # torch.save(latents[idx].to(torch.bfloat16), latent_path)
                item = {}
                # item["length"] = latents[idx].shape[1]
                item["length"] = 12
                # 这里将 latent_path 存为相对路径，例如 "9/xxxx.pt"
                item["latent_path"] = relative_name + ".pt"
                item["caption"] = data["text"][idx]
                json_data.append(item)
                # print(f"{relative_name} processed")



    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption_temp.json"),
                  "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=64,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range",
                        type=int,
                        default=10.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--dataset", default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name",
                        type=str,
                        default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    args = parser.parse_args()
    main(args)
