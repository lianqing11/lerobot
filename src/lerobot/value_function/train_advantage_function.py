#!/usr/bin/env python
"""
Train a KAI-0-style advantage function for offline RL.

The advantage function predicts A(o_ref, o_cur, ℓ) — the progress difference
between a reference observation and a current observation, conditioned on
instruction ℓ. Output is a scalar in [-1, 1].

Supports multi-GPU via `accelerate`:
    # Single GPU
    python -m lerobot.value_function.train_advantage_function --dataset_list_file ... --output_dir ...

    # Multi-GPU
    accelerate launch --num_processes 4 -m lerobot.value_function.train_advantage_function \
        --dataset_list_file ... --output_dir ...

All episodes in the dataset_list_file are assumed to be successful demonstrations.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset, Subset

# Force unbuffered output so logs appear in real time on cloud platforms
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_log_handler.flush = lambda: sys.stderr.flush()  # type: ignore[assignment]
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)

_file_handler = None


def _setup_file_logging(output_dir: str):
    """Add a file handler so logs are written to output_dir/train.log."""
    global _file_handler  # noqa: PLW0603
    if _file_handler is not None:
        return
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(Path(output_dir) / "train.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    _file_handler = fh


def format_advantage_prompts(tasks: list[str]) -> list[str]:
    """Format advantage-function prompts for two-timestep comparison."""
    prompts = []
    for task in tasks:
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        prompts.append(f"Compare reference and current observation. Task: {cleaned};")
    return prompts


# ── Dataset list file parsing (reuse from value function) ─────────────


def parse_dataset_list_file(filepath: str) -> list[tuple[str, str]]:
    """Parse dataset list file, return list of (repo_id, root_path)."""
    entries = []
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                path = parts[0].rstrip("/")
                repo_id = Path(path).name
                entries.append((repo_id, path))
            elif len(parts) == 2:
                entries.append((parts[0], parts[1].rstrip("/")))
            else:
                raise ValueError(f"Invalid line: '{raw_line}'")
    return entries


@dataclass
class EpisodeInfo:
    ds_idx: int
    repo_id: str
    episode_index: int
    task_name: str
    length: int
    dataset_from_index: int
    dataset_to_index: int
    frame_start: int = 0
    frame_end: int = 0


@dataclass
class AdvantageFrameInfo:
    ds_idx: int
    repo_id: str
    episode_index: int
    task_name: str
    frame_in_episode: int
    global_frame_idx: int
    episode_from_index: int
    episode_to_index: int
    episode_length: int


def _extract_episode_task_name(ds, episode_row) -> str:
    """Best-effort task lookup for one episode."""
    episode_tasks = episode_row.get("tasks")
    if episode_tasks:
        if isinstance(episode_tasks, (list, tuple)):
            return str(episode_tasks[0])
        return str(episode_tasks)

    if ds.meta.tasks is None or len(ds.meta.tasks) == 0:
        return "unknown"

    task_idx = episode_row.get("task_index")
    if task_idx is not None:
        return str(ds.meta.tasks.iloc[int(task_idx)].name)

    return str(ds.meta.tasks.index[0])


def frame_indices_from_episode_infos(episode_infos: list[EpisodeInfo]) -> list[int]:
    return [idx for info in episode_infos for idx in range(info.frame_start, info.frame_end)]


def split_episode_infos_for_eval(
    episode_infos: list[EpisodeInfo], eval_episodes: int
) -> tuple[list[EpisodeInfo], list[EpisodeInfo]]:
    """Deterministically reserve whole episodes for evaluation."""
    total_episodes = len(episode_infos)
    max_eval_episodes = max(total_episodes - 1, 0)
    eval_episodes = max(0, min(eval_episodes, max_eval_episodes))
    if eval_episodes == 0:
        return episode_infos, []

    task_to_indices = {}
    for idx, info in enumerate(episode_infos):
        task_to_indices.setdefault(info.task_name, []).append(idx)

    eval_episode_indices = set()
    while len(eval_episode_indices) < eval_episodes:
        added_any = False
        for task_name in sorted(task_to_indices):
            candidates = task_to_indices[task_name]
            if not candidates:
                continue
            eval_episode_indices.add(candidates.pop())
            added_any = True
            if len(eval_episode_indices) >= eval_episodes:
                break
        if not added_any:
            break

    train_infos = [info for idx, info in enumerate(episode_infos) if idx not in eval_episode_indices]
    eval_infos = [info for idx, info in enumerate(episode_infos) if idx in eval_episode_indices]
    return train_infos, eval_infos


# ── Advantage function dataset ───────────────────────────────────────


class AdvantageDataset(Dataset):
    """Wraps multiple LeRobotDatasets and returns two-timestep pairs with progress targets.

    Each sample contains:
        - All fields from the underlying LeRobotDataset for the current frame
        - ref_* keys for the reference frame's image features
        - progress_target: float in [-1, 1], progress(current) - progress(ref)
        - current_progress: float in [0, 1], frame_idx / (ep_len - 1)
        - ref_progress: float in [0, 1], ref_frame_idx / (ep_len - 1)
    """

    def __init__(
        self,
        dataset_entries: list[tuple[str, str]],
        video_backend: str = "pyav",
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.sub_datasets = []
        self.episode_infos: list[EpisodeInfo] = []
        self.frame_map: list[AdvantageFrameInfo] = []

        logger.info(f"Loading {len(dataset_entries)} datasets...")

        for repo_id, root in dataset_entries:
            ds = LeRobotDataset(repo_id, root=root, video_backend=video_backend)
            ds_idx = len(self.sub_datasets)
            self.sub_datasets.append(ds)

            for ep in ds.meta.episodes:
                ep_idx = int(ep["episode_index"])
                length = int(ep["length"])
                task_name = _extract_episode_task_name(ds, ep)
                from_idx = int(ep["dataset_from_index"])
                to_idx = int(ep["dataset_to_index"])
                self.episode_infos.append(
                    EpisodeInfo(
                        ds_idx=ds_idx,
                        repo_id=repo_id,
                        episode_index=ep_idx,
                        task_name=task_name,
                        length=length,
                        dataset_from_index=from_idx,
                        dataset_to_index=to_idx,
                    )
                )

        # Build frame map
        for episode_info in self.episode_infos:
            episode_info.frame_start = len(self.frame_map)
            for frame_in_ep in range(episode_info.length):
                global_idx = episode_info.dataset_from_index + frame_in_ep
                self.frame_map.append(
                    AdvantageFrameInfo(
                        ds_idx=episode_info.ds_idx,
                        repo_id=episode_info.repo_id,
                        episode_index=episode_info.episode_index,
                        task_name=episode_info.task_name,
                        frame_in_episode=frame_in_ep,
                        global_frame_idx=global_idx,
                        episode_from_index=episode_info.dataset_from_index,
                        episode_to_index=episode_info.dataset_to_index,
                        episode_length=episode_info.length,
                    )
                )
            episode_info.frame_end = len(self.frame_map)

        logger.info(
            f"Total frames: {len(self.frame_map)}, "
            f"Datasets: {len(self.sub_datasets)}, "
            f"Episodes: {len(self.episode_infos)}"
        )

    def __len__(self):
        return len(self.frame_map)

    def __getitem__(self, idx):
        frame_info = self.frame_map[idx]
        ds = self.sub_datasets[frame_info.ds_idx]

        # Get current frame
        item = ds[frame_info.global_frame_idx]

        # Pick a random different frame from the same episode
        ep_from = frame_info.episode_from_index
        ep_to = frame_info.episode_to_index
        ep_length = frame_info.episode_length

        if ep_length <= 1:
            # Single-frame episode: reference is the same frame, progress_target = 0
            ref_frame_in_ep = frame_info.frame_in_episode
            ref_global_idx = frame_info.global_frame_idx
        else:
            # Pick random frame from same episode, different from current
            while True:
                ref_global_idx = random.randint(ep_from, ep_to - 1)
                if ref_global_idx != frame_info.global_frame_idx:
                    break
            ref_frame_in_ep = ref_global_idx - ep_from

        # Get reference frame
        ref_item = ds[ref_global_idx]

        # Add reference images with ref_ prefix
        for key in list(ref_item.keys()):
            if "image" in key and isinstance(ref_item[key], torch.Tensor):
                item[f"ref_{key}"] = ref_item[key]

        # Compute progress values (single stage: progress = frame_idx / (ep_len - 1))
        denominator = max(ep_length - 1, 1)
        current_progress = frame_info.frame_in_episode / denominator
        ref_progress = ref_frame_in_ep / denominator
        progress_target = current_progress - ref_progress  # in [-1, 1]

        item["progress_target"] = progress_target
        item["current_progress"] = current_progress
        item["ref_progress"] = ref_progress

        return item


# ── Collate / processor helpers ──────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals)
        elif isinstance(vals[0], str):
            collated[k] = vals
        else:
            collated[k] = vals
    return collated


def prepare_advantage_inputs(batch: dict, processor, model, image_keys, device="cuda"):
    tasks = batch.get("task", [""] * len(batch["progress_target"]))
    prompts = format_advantage_prompts(tasks)
    ref_image_keys = [f"ref_{k}" for k in image_keys]
    return model.prepare_inputs(batch, image_keys, ref_image_keys, processor, prompts, device=device)


def infer_image_keys_from_adv_dataset(dataset: AdvantageDataset) -> list[str]:
    """Infer image keys from dataset metadata without decoding a sample in the main process."""
    image_keys = sorted(
        {
            key
            for sub_dataset in dataset.sub_datasets
            for key in getattr(sub_dataset.meta, "video_keys", [])
            if "image" in key
        }
    )
    if not image_keys:
        raise ValueError("No image keys found in dataset metadata")
    return image_keys


def load_advantage_processor(backbone_family: str, pretrained_model_name: str):
    if backbone_family == "paligemma":
        from transformers import AutoTokenizer

        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        cache_root = Path(hf_home) / "hub" / "models--google--paligemma-3b-pt-224"
        ref_main = cache_root / "refs" / "main"
        if not ref_main.exists():
            raise FileNotFoundError(f"Missing cached tokenizer ref: {ref_main}")

        snapshot_dir = cache_root / "snapshots" / ref_main.read_text().strip()
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Missing cached tokenizer snapshot: {snapshot_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Loaded PaliGemma tokenizer from local Hugging Face cache: %s", snapshot_dir)
        return tokenizer

    from transformers import AutoProcessor
    from lerobot.value_function.advantage_function_pi05 import _hf_cache_snapshot_dir, _without_socks_proxy

    logger.info("Loading Qwen-VL processor from %s", pretrained_model_name)
    cached_dir = _hf_cache_snapshot_dir(pretrained_model_name)
    processor_source = str(cached_dir) if cached_dir is not None else pretrained_model_name
    processor_kwargs = {"trust_remote_code": True, "use_fast": True}
    if cached_dir is not None:
        processor_kwargs["local_files_only"] = True
    with _without_socks_proxy():
        return AutoProcessor.from_pretrained(processor_source, **processor_kwargs)


# ── Evaluation ───────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, image_keys, processor, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    raw_model = model.module if hasattr(model, "module") else model

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        prepared_inputs = prepare_advantage_inputs(
            batch, processor, raw_model, image_keys, device=str(device)
        )
        progress_target = batch["progress_target"].float().to(device)

        loss, info = raw_model.compute_loss(prepared_inputs, progress_target)

        bs = progress_target.shape[0]
        total_loss += info["loss"] * bs
        total_mae += info["mae"] * bs
        total_samples += bs

    model.train()
    if total_samples == 0:
        return {"eval_loss": 0.0, "eval_mae": 0.0, "eval_samples": 0}

    return {
        "eval_loss": total_loss / total_samples,
        "eval_mae": total_mae / total_samples,
        "eval_samples": total_samples,
    }


# ── Training ─────────────────────────────────────────────────────────


def train(args):
    # ── Accelerator ──────────────────────────────────────────────────
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16" if args.precision == "bfloat16" else "no",
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        _setup_file_logging(args.output_dir)
        with open(Path(args.output_dir) / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Dataset ──────────────────────────────────────────────────────
    if is_main:
        logger.info("Loading datasets...")
    dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    adv_dataset = AdvantageDataset(
        dataset_entries, video_backend=args.video_backend,
    )

    train_episode_infos, eval_episode_infos = split_episode_infos_for_eval(
        adv_dataset.episode_infos, args.eval_episodes
    )
    train_frame_indices = frame_indices_from_episode_infos(train_episode_infos)
    eval_frame_indices = frame_indices_from_episode_infos(eval_episode_infos)
    train_subset = Subset(adv_dataset, train_frame_indices)
    eval_subset = Subset(adv_dataset, eval_frame_indices)
    if is_main:
        logger.info(
            f"Train: {len(train_frame_indices)} frames / {len(train_episode_infos)} episodes, "
            f"Eval: {len(eval_frame_indices)} frames / {len(eval_episode_infos)} episodes"
        )
        if eval_episode_infos:
            eval_tasks = sorted({info.task_name for info in eval_episode_infos})
            logger.info(f"Eval tasks: {eval_tasks}")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    image_keys = infer_image_keys_from_adv_dataset(adv_dataset)
    if is_main:
        logger.info(f"Image features: {image_keys}")

    # ── Model ────────────────────────────────────────────────────────
    from lerobot.value_function.advantage_function_pi05 import DEFAULT_QWEN_VL_MODEL, PI05AdvantageFunction

    if not getattr(args, "pretrained_model_name", None):
        args.pretrained_model_name = DEFAULT_QWEN_VL_MODEL

    processor = load_advantage_processor(args.backbone_family, args.pretrained_model_name)
    load_pretrained_backbone = (
        args.backbone_family == "qwen_vl" and not args.resume and not args.allow_random_init
    )

    model = PI05AdvantageFunction(
        backbone_family=args.backbone_family,
        pretrained_model_name=args.pretrained_model_name,
        vlm_variant=args.vlm_variant,
        image_resolution=(224, 224),
        precision="float32",  # accelerator handles mixed precision
        gradient_checkpointing=args.gradient_checkpointing,
        load_pretrained_backbone=load_pretrained_backbone,
    )

    if args.backbone_family == "paligemma":
        if args.init_from_policy:
            if is_main:
                logger.info(f"Initializing VLM from policy: {args.init_from_policy}")
            _load_vlm_weights_from_policy(model, args.init_from_policy)
        elif not args.resume:
            if not args.allow_random_init:
                raise ValueError(
                    "--init_from_policy is required for fresh PaliGemma advantage training. "
                    "Pass --allow_random_init to override."
                )
            logger.warning("PaliGemma VLM backbone starts from RANDOM weights.")
    else:
        if args.init_from_policy and is_main:
            logger.warning("--init_from_policy is ignored for qwen_vl backbones")
        if not args.resume:
            if args.allow_random_init:
                logger.warning("Qwen-VL backbone starts from RANDOM config weights.")
            elif is_main:
                logger.info("Initializing Qwen-VL backbone from pretrained model: %s", args.pretrained_model_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1,
    )

    # ── Resume ───────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            if (resume_path / "training_state.pt").exists():
                resume_path = resume_path / "training_state.pt"
            else:
                resume_path = resume_path / "advantage_function.pt"
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            logger.warning("No optimizer state in checkpoint, starting optimizer from scratch")
        start_step = ckpt["step"]
        for _ in range(start_step):
            scheduler.step()
        if is_main:
            logger.info(f"Resumed from step {start_step}")

    # ── Wrap with accelerator ────────────────────────────────────────
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )

    # ── Wandb (main process only) ────────────────────────────────────
    wandb_run = None
    if args.wandb_project and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=vars(args), resume="allow",
        )

    # ── Training loop ────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    running_loss = 0.0
    running_mae = 0.0
    log_count = 0
    train_start = time.time()

    raw_model = accelerator.unwrap_model(model)

    if is_main:
        logger.info(
            f"Training for {args.steps} steps (from {start_step}), "
            f"batch_size={args.batch_size} x {accelerator.num_processes} GPUs"
        )
        sys.stderr.flush()

    DETAIL_LOG_STEPS = 10  # log every phase for the first N steps

    for global_step in range(start_step + 1, args.steps + 1):
        step_t0 = time.time()
        verbose = is_main and (global_step - start_step) <= DETAIL_LOG_STEPS

        if verbose:
            logger.info("[step %d] loading batch ...", global_step)
            sys.stderr.flush()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t_data = time.time()
        if verbose:
            logger.info("[step %d] batch loaded (%.2fs), preparing inputs ...", global_step, t_data - step_t0)
            sys.stderr.flush()

        prepared_inputs = prepare_advantage_inputs(
            batch, processor, raw_model, image_keys, device=str(device)
        )
        progress_target = batch["progress_target"].float().to(device)

        t_prep = time.time()
        if verbose:
            logger.info("[step %d] inputs prepared (%.2fs), forward ...", global_step, t_prep - t_data)
            sys.stderr.flush()

        with accelerator.autocast():
            pred = model(prepared_inputs=prepared_inputs).squeeze(-1)  # [B]
            loss = F.mse_loss(pred, progress_target)

        t_fwd = time.time()
        if verbose:
            logger.info("[step %d] forward done (%.2fs), backward ...", global_step, t_fwd - t_prep)
            sys.stderr.flush()

        with torch.no_grad():
            mae = (pred.detach() - progress_target).abs().mean()

        info = {
            "loss": loss.item(),
            "mae": mae.item(),
            "pred_mean": pred.detach().mean().item(),
            "target_mean": progress_target.mean().item(),
        }

        optimizer.zero_grad()
        accelerator.backward(loss)
        if args.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        t_bwd = time.time()
        if verbose:
            logger.info(
                "[step %d] done (%.2fs total: data=%.2f prep=%.2f fwd=%.2f bwd=%.2f) loss=%.4f",
                global_step, t_bwd - step_t0,
                t_data - step_t0, t_prep - t_data, t_fwd - t_prep, t_bwd - t_fwd,
                info["loss"],
            )
            sys.stderr.flush()

        running_loss += info["loss"]
        running_mae += info["mae"]
        log_count += 1

        # ── Log (main only) ──────────────────────────────────────
        if global_step % args.log_freq == 0 and is_main:
            avg_loss = running_loss / log_count
            avg_mae = running_mae / log_count
            elapsed = time.time() - train_start
            steps_done = global_step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

            logger.info(
                f"step {global_step}/{args.steps} | "
                f"loss={avg_loss:.4f} | mae={avg_mae:.4f} | "
                f"pred={info['pred_mean']:.4f} | tgt={info['target_mean']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {steps_per_sec:.2f} it/s"
            )
            sys.stderr.flush()
            if wandb_run:
                wandb_run.log({
                    "train/loss": avg_loss, "train/mae": avg_mae,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/steps_per_sec": steps_per_sec,
                }, step=global_step)

            running_loss = 0.0
            running_mae = 0.0
            log_count = 0

        # ── Eval (main only) ─────────────────────────────────────
        if args.eval_freq > 0 and global_step % args.eval_freq == 0 and is_main:
            eval_metrics = evaluate(
                model, eval_loader, image_keys, processor, device, max_batches=50,
            )
            logger.info(
                f"[Eval] step {global_step} | "
                f"loss={eval_metrics['eval_loss']:.4f} | mae={eval_metrics['eval_mae']:.4f}"
            )
            if wandb_run:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=global_step)

        # ── Save (main only) ─────────────────────────────────────
        if (global_step % args.save_freq == 0 or global_step == args.steps) and is_main:
            ckpt_path = Path(args.output_dir) / f"checkpoint_{global_step:06d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model_config = {
                "model_type": "advantage",
                "backbone_family": args.backbone_family,
                "pretrained_model_name": args.pretrained_model_name,
                "vlm_variant": args.vlm_variant,
                "precision": args.precision,
                "image_keys": image_keys,
            }
            state_dict = raw_model.state_dict()
            # Lightweight inference checkpoint
            torch.save(
                {"step": global_step, "model_state_dict": state_dict, "config": model_config},
                ckpt_path / "advantage_function.pt",
            )
            # Full training checkpoint (+ optimizer state for resuming)
            torch.save(
                {"step": global_step, "model_state_dict": state_dict,
                 "optimizer_state_dict": optimizer.state_dict(), "config": model_config},
                ckpt_path / "training_state.pt",
            )
            logger.info(f"Saved checkpoint -> {ckpt_path}")

        # Sync all processes at save boundaries
        if global_step % args.save_freq == 0:
            accelerator.wait_for_everyone()

    total_time = time.time() - train_start
    if is_main:
        logger.info(f"Done. {args.steps - start_step} steps in {total_time/60:.1f} min")
        if wandb_run:
            wandb_run.finish()


def _load_vlm_weights_from_policy(model, policy_path: str):
    """Load VLM (PaliGemma) weights from a pretrained PI05 policy checkpoint."""
    from safetensors.torch import load_file

    safetensors_path = Path(policy_path) / "model.safetensors"
    if not safetensors_path.exists():
        logger.warning(f"No model.safetensors at {policy_path}, skipping VLM init")
        return

    state_dict = load_file(str(safetensors_path))
    vlm_state = {}
    for key, val in state_dict.items():
        prefix = "paligemma_with_expert.paligemma."
        model_prefix = "model.paligemma_with_expert.paligemma."
        if key.startswith(model_prefix):
            new_key = "paligemma." + key[len(model_prefix):]
            vlm_state[new_key] = val
        elif key.startswith(prefix):
            new_key = "paligemma." + key[len(prefix):]
            vlm_state[new_key] = val

    if not vlm_state:
        logger.warning("No VLM weights found in policy checkpoint")
        return

    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    loaded = len(vlm_state) - len(unexpected)
    logger.info(f"Loaded {loaded} VLM weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Train KAI-0-style advantage function")
    # Data
    p.add_argument("--dataset_list_file", type=str, required=True)
    p.add_argument("--video_backend", type=str, default="pyav")
    # Model
    p.add_argument("--backbone_family", type=str, default="qwen_vl", choices=["qwen_vl", "paligemma"])
    p.add_argument("--pretrained_model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--vlm_variant", type=str, default="gemma_300m", choices=["gemma_300m", "gemma_2b"])
    p.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--init_from_policy", type=str, default=None)
    p.add_argument("--allow_random_init", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Training
    p.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Logging
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=2000)
    p.add_argument("--eval_freq", type=int, default=100000)
    p.add_argument("--eval_episodes", type=int, default=5, help="Approx episodes held out for eval")
    p.add_argument("--output_dir", type=str, default="checkpoints/advantage_function")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
