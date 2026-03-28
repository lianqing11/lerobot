#!/usr/bin/env python3
"""
Clean DAgger dataset: remove stationary frames at policy→human transition
and optionally smooth the action trajectory at the boundary.

The stationary period happens when the human presses SPACE to take over
but hasn't started moving the Leader arm yet — the action/state values
freeze for ~1-2 seconds.

This script:
  1. Detects still frames per episode (action + state delta < threshold)
  2. Removes them from parquet data
  3. Re-encodes videos without the removed frames
  4. Optionally applies linear interpolation to smooth the transition boundary
  5. Rebuilds all metadata (info.json, episodes.parquet, stats.json)

Usage:
  # Dry-run: only print detection results
  python clean_dagger.py /path/to/dataset --dry-run

  # Clean and write to a new directory
  python clean_dagger.py /path/to/dataset -o /path/to/cleaned

  # Clean in-place (overwrites original!)
  python clean_dagger.py /path/to/dataset --inplace

  # With smoothing at transition boundaries
  python clean_dagger.py /path/to/dataset -o /path/to/cleaned --smooth 5
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def detect_still_region(
    actions: np.ndarray,
    states: np.ndarray,
    interventions: np.ndarray,
    *,
    combined_threshold: float = 0.5,
    min_consecutive: int = 3,
    sustained_motion_frames: int = 10,
) -> tuple[int, int] | None:
    """Find the still zone right after policy→human transition.

    Strategy: starting from the first human-control frame, scan forward to
    find the first point where *sustained_motion_frames* consecutive frames
    all have combined delta > threshold.  Everything between the transition
    and that sustained-motion start is marked as "still" and will be removed.

    This handles non-contiguous still zones (e.g. still→brief spike→still)
    which a simple contiguous-run detector would miss.

    Returns (start, end) frame indices (inclusive) of the still region within
    the episode, or None if no transition / no still region found.
    """
    changes = np.where(np.diff(interventions) != 0)[0]
    if len(changes) == 0:
        return None

    trans = changes[0]  # first policy→human transition

    action_deltas = np.abs(np.diff(actions, axis=0)).sum(axis=1)
    state_deltas = np.abs(np.diff(states[:, :7], axis=0)).sum(axis=1)
    combined = action_deltas + state_deltas

    # The first human frame is trans + 1
    search_start = trans + 1
    search_end = min(len(combined), trans + 300)

    # Find the first frame where sustained motion begins: N consecutive
    # frames all above threshold
    motion_start = None
    for i in range(search_start, search_end - sustained_motion_frames + 1):
        if all(combined[j] >= combined_threshold for j in range(i, i + sustained_motion_frames)):
            motion_start = i + 1  # delta[i] is between frame i and i+1
            break

    if motion_start is None:
        return None

    still_start = search_start
    still_end = motion_start - 1

    if still_end < still_start or (still_end - still_start + 1) < min_consecutive:
        return None

    return still_start, still_end


def smooth_transition(
    actions: np.ndarray,
    keep_mask: np.ndarray,
    still_start: int,
    still_end: int,
    smooth_window: int = 5,
) -> np.ndarray:
    """Apply linear interpolation at transition boundary.

    Blends the last `smooth_window` frames before the still region with
    the first frame after it, so the trajectory ramps smoothly instead
    of having a discontinuity.
    """
    actions = actions.copy()
    first_motion = still_end + 1
    if first_motion >= len(actions):
        return actions

    blend_start = max(0, still_start - smooth_window)
    src = actions[blend_start]
    dst = actions[first_motion]
    n = first_motion - blend_start
    for i in range(n):
        alpha = i / n
        idx = blend_start + i
        if keep_mask[idx]:
            actions[idx] = src * (1 - alpha) + dst * alpha

    return actions


def reencode_video(
    src_path: Path,
    dst_path: Path,
    keep_indices: set[int],
    fps: int,
    codec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Re-encode a video keeping only frames whose index is in keep_indices."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    in_container = av.open(str(src_path))
    in_stream = in_container.streams.video[0]

    out_container = av.open(str(dst_path), mode="w")
    out_stream = out_container.add_stream(codec, rate=fps)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = pix_fmt
    out_stream.time_base = in_stream.time_base

    out_pts = 0
    pts_step = int(1 / (fps * in_stream.time_base))

    frame_idx = 0
    for frame in in_container.decode(video=0):
        if frame_idx in keep_indices:
            new_frame = frame.reformat(width=in_stream.width, height=in_stream.height, format=pix_fmt)
            new_frame.pts = out_pts
            new_frame.time_base = in_stream.time_base
            out_pts += pts_step
            for packet in out_stream.encode(new_frame):
                out_container.mux(packet)
        frame_idx += 1

    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()


def compute_stats(values: np.ndarray) -> dict:
    """Compute statistics matching LeRobot's format."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean DAgger dataset: remove still frames at transitions")
    parser.add_argument("dataset", type=Path, help="Path to the LeRobot dataset directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: <dataset>-cleaned)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the original dataset")
    parser.add_argument("--dry-run", action="store_true", help="Only detect and print, don't modify")
    parser.add_argument("--threshold", type=float, default=0.5, help="Combined action+state delta threshold for 'still' (default: 0.5)")
    parser.add_argument("--min-consecutive", type=int, default=3, help="Minimum consecutive still frames to trigger removal (default: 3)")
    parser.add_argument("--smooth", type=int, default=0, help="Smoothing window size at transition boundary (0=disabled, default: 0)")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec for re-encoding (default: libsvtav1)")
    args = parser.parse_args()

    src = args.dataset.resolve()
    if not src.exists():
        parser.error(f"Dataset not found: {src}")

    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    fps = info["fps"]
    features = info["features"]
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    df = pd.read_parquet(src / "data" / "chunk-000" / "file-000.parquet")
    episodes = sorted(df["episode_index"].unique())

    logger.info(f"Dataset: {src}")
    logger.info(f"Episodes: {len(episodes)}, Total frames: {len(df)}, FPS: {fps}")
    logger.info(f"Video keys: {video_keys}")
    logger.info(f"Threshold: {args.threshold}, Min consecutive: {args.min_consecutive}")

    # --- Phase 1: Detect still regions ---
    episode_removals: dict[int, tuple[int, int]] = {}
    total_removed = 0

    for ep in episodes:
        ep_mask = df["episode_index"] == ep
        ep_df = df[ep_mask].reset_index(drop=True)
        actions = np.stack(ep_df["action"].values)
        states = np.stack(ep_df["observation.state"].values)
        interventions = ep_df["intervention"].values

        result = detect_still_region(
            actions, states, interventions,
            combined_threshold=args.threshold,
            min_consecutive=args.min_consecutive,
        )

        if result is None:
            logger.info(f"  Episode {ep}: no still region detected")
            continue

        still_start, still_end = result
        n_still = still_end - still_start + 1
        t_still = n_still / fps

        changes = np.where(np.diff(interventions) != 0)[0]
        trans_frame = changes[0] + 1 if len(changes) > 0 else -1

        logger.info(
            f"  Episode {ep}: transition at frame {trans_frame}, "
            f"still region frames {still_start}-{still_end} "
            f"({YELLOW}{n_still} frames, {t_still:.1f}s{RESET})"
        )

        episode_removals[ep] = (still_start, still_end)
        total_removed += n_still

    logger.info(f"\nTotal frames to remove: {RED}{total_removed}{RESET} / {len(df)}")
    remaining = len(df) - total_removed
    logger.info(f"Frames after cleaning: {GREEN}{remaining}{RESET}")

    if args.dry_run:
        logger.info("Dry-run mode — no files modified.")
        return

    # --- Phase 2: Build cleaned data ---
    dst = args.output or Path(str(src) + "-cleaned")
    if args.inplace:
        dst = src
    else:
        if dst.exists():
            logger.warning(f"Output directory exists, will overwrite: {dst}")
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    logger.info(f"\nWriting cleaned dataset to: {dst}")

    keep_rows: list[int] = []
    ep_keep_frames: dict[int, list[int]] = {}

    global_idx = 0
    for ep in episodes:
        ep_mask = df["episode_index"] == ep
        ep_df = df[ep_mask].reset_index(drop=True)
        n_ep = len(ep_df)
        ep_global_start = ep_df.index[0]

        if ep in episode_removals:
            still_start, still_end = episode_removals[ep]
            local_keep = [i for i in range(n_ep) if not (still_start <= i <= still_end)]
        else:
            local_keep = list(range(n_ep))

        ep_keep_frames[ep] = local_keep

        base_global = df[ep_mask].index[0]
        for local_i in local_keep:
            keep_rows.append(base_global + local_i)

    cleaned_df = df.iloc[keep_rows].copy().reset_index(drop=True)

    # Apply smoothing if requested
    if args.smooth > 0:
        for ep in episodes:
            if ep not in episode_removals:
                continue
            still_start, still_end = episode_removals[ep]
            ep_mask_orig = df["episode_index"] == ep
            ep_actions_orig = np.stack(df[ep_mask_orig].reset_index(drop=True)["action"].values)

            keep_mask = np.ones(len(ep_actions_orig), dtype=bool)
            keep_mask[still_start : still_end + 1] = False

            smoothed = smooth_transition(
                ep_actions_orig, keep_mask, still_start, still_end, args.smooth
            )

            ep_mask_clean = cleaned_df["episode_index"] == ep
            kept_original_indices = ep_keep_frames[ep]
            smoothed_kept = smoothed[kept_original_indices]
            rows = cleaned_df[ep_mask_clean].index
            for i, row_idx in enumerate(rows):
                cleaned_df.at[row_idx, "action"] = smoothed_kept[i].astype(np.float32)
        logger.info(f"Applied smoothing with window={args.smooth}")

    # Reindex: frame_index, index, timestamp
    new_frame_idx = []
    new_index = []
    new_timestamp = []
    global_counter = 0
    for ep in episodes:
        ep_mask = cleaned_df["episode_index"] == ep
        ep_len = ep_mask.sum()
        for local_i in range(ep_len):
            new_frame_idx.append(local_i)
            new_index.append(global_counter)
            new_timestamp.append(local_i / fps)
            global_counter += 1

    cleaned_df["frame_index"] = new_frame_idx
    cleaned_df["index"] = new_index
    cleaned_df["timestamp"] = np.array(new_timestamp, dtype=np.float32)

    # Write parquet
    data_dir = dst / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(data_dir / "file-000.parquet", index=False)
    logger.info(f"Wrote {len(cleaned_df)} frames to parquet")

    # --- Phase 3: Re-encode videos ---
    for vkey in video_keys:
        src_video = src / "videos" / vkey / "chunk-000" / "file-000.mp4"
        if not src_video.exists():
            logger.warning(f"Video not found: {src_video}")
            continue

        # Build per-episode global frame offset in original video
        # Video file-000.mp4 contains ALL episodes sequentially
        dst_video = dst / "videos" / vkey / "chunk-000" / "file-000.mp4"

        # Compute which global video frames to keep
        keep_video_frames: set[int] = set()
        offset = 0
        for ep in episodes:
            ep_mask_orig = df["episode_index"] == ep
            ep_len_orig = ep_mask_orig.sum()
            for local_i in ep_keep_frames[ep]:
                keep_video_frames.add(offset + local_i)
            offset += ep_len_orig

        logger.info(f"Re-encoding {vkey}: {offset} -> {len(keep_video_frames)} frames ...")
        reencode_video(
            src_video,
            dst_video,
            keep_video_frames,
            fps=fps,
            codec=args.vcodec,
            pix_fmt=features[vkey]["info"].get("video.pix_fmt", "yuv420p"),
        )
        logger.info(f"  Done: {dst_video}")

    # --- Phase 4: Rebuild metadata ---
    meta_dir = dst / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Copy tasks.parquet
    shutil.copy2(src / "meta" / "tasks.parquet", meta_dir / "tasks.parquet")

    # Rebuild episodes metadata
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    src_ep_df = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep_records = []

    dataset_offset = 0
    for ep in episodes:
        ep_mask = cleaned_df["episode_index"] == ep
        ep_data = cleaned_df[ep_mask]
        ep_len = len(ep_data)

        src_row = src_ep_df[src_ep_df["episode_index"] == ep].iloc[0].to_dict()
        src_row["length"] = ep_len
        src_row["dataset_from_index"] = dataset_offset
        src_row["dataset_to_index"] = dataset_offset + ep_len

        for vkey in video_keys:
            ts_key_from = f"videos/{vkey}/from_timestamp"
            ts_key_to = f"videos/{vkey}/to_timestamp"
            src_row[ts_key_from] = 0.0
            src_row[ts_key_to] = (ep_len - 1) / fps

        # Recompute per-episode stats for numeric features
        for feat_name in ["action", "observation.state", "intervention"]:
            if feat_name in ep_data.columns:
                vals = np.stack(ep_data[feat_name].values)
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)
                st = compute_stats(vals.astype(float))
                for stat_key, stat_val in st.items():
                    col_name = f"stats/{feat_name}/{stat_key}"
                    src_row[col_name] = stat_val

        for feat_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if feat_name in ep_data.columns:
                vals = ep_data[feat_name].values.reshape(-1, 1).astype(float)
                st = compute_stats(vals)
                for stat_key, stat_val in st.items():
                    col_name = f"stats/{feat_name}/{stat_key}"
                    src_row[col_name] = stat_val

        ep_records.append(src_row)
        dataset_offset += ep_len

    new_ep_df = pd.DataFrame(ep_records)
    new_ep_df.to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    # Rebuild global stats
    global_stats = {}
    for feat_name in ["action", "observation.state", "intervention"]:
        if feat_name in cleaned_df.columns:
            vals = np.stack(cleaned_df[feat_name].values)
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            global_stats[feat_name] = compute_stats(vals.astype(float))

    for feat_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if feat_name in cleaned_df.columns:
            vals = cleaned_df[feat_name].values.reshape(-1, 1).astype(float)
            global_stats[feat_name] = compute_stats(vals)

    # Keep image stats from original (pixel stats don't change much)
    with open(src / "meta" / "stats.json") as f:
        orig_stats = json.load(f)
    for k, v in orig_stats.items():
        if k.startswith("observation.images."):
            global_stats[k] = v

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    # Rebuild info.json
    info["total_frames"] = len(cleaned_df)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"\n{GREEN}Done!{RESET}")
    logger.info(f"  Original: {len(df)} frames")
    logger.info(f"  Cleaned:  {len(cleaned_df)} frames (removed {total_removed})")
    logger.info(f"  Output:   {dst}")


if __name__ == "__main__":
    main()
