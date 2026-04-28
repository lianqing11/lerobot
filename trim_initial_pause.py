#!/usr/bin/env python3
"""
Trim initial pause/stillness frames from each episode in a LeRobot v3.0 dataset.

Detects the initial "stuck" period at the start of each episode (where the robot
is not moving) and removes those frames from both parquet data and videos.

Usage:
    # Dry run - show what would be trimmed
    python trim_initial_pause.py --dataset_path /path/to/dataset --dry_run

    # Actually trim (creates backup at /path/to/dataset_backup_before_trim)
    python trim_initial_pause.py --dataset_path /path/to/dataset

    # Batch mode - process multiple datasets from a file (one path per line)
    python trim_initial_pause.py --dataset_list datasets.txt --dry_run
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def detect_pause_end(actions: np.ndarray, threshold: float = 0.15, window: int = 3) -> int:
    """Detect the first frame where sustained motion begins.

    Looks at consecutive action differences and finds where the robot starts
    moving consistently (action change > threshold for `window` consecutive frames).

    Returns:
        Index of the first "moving" frame (0 means no pause detected).
    """
    if len(actions) < window + 1:
        return 0

    action_diff = np.abs(np.diff(actions, axis=0)).sum(axis=1)

    for i in range(len(action_diff)):
        if action_diff[i] > threshold:
            end = min(i + window, len(action_diff))
            if np.mean(action_diff[i:end]) > threshold:
                return i

    return 0


def get_video_keys(info: dict) -> list[str]:
    """Get video feature keys from dataset info."""
    return [
        key for key, feat in info["features"].items()
        if feat.get("dtype") == "video"
    ]


def trim_video_with_ffmpeg(
    input_paths_with_segments: list[tuple[Path, list[tuple[float, float]]]],
    output_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
):
    """Re-encode video keeping only the specified timestamp segments.

    Supports multiple input files — segments from each file are trimmed and
    concatenated into a single output. This handles the LeRobot case where
    episodes may be split across file-000.mp4, file-001.mp4, etc.

    Args:
        input_paths_with_segments: List of (video_path, segments) tuples.
            Each segments list contains (start_sec, end_sec) pairs to keep
            from that specific video file. Entries with empty segments are skipped.
        output_path: Path for the trimmed video.
        fps: Frame rate.
        vcodec: Video codec.
        pix_fmt: Pixel format.
    """
    filter_parts = []
    concat_inputs = []
    input_args = []
    input_idx = 0
    seg_idx = 0

    for input_path, segments in input_paths_with_segments:
        if not segments:
            continue
        input_args.extend(["-i", str(input_path)])
        for start, end in segments:
            filter_parts.append(
                f"[{input_idx}:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{seg_idx}]"
            )
            concat_inputs.append(f"[v{seg_idx}]")
            seg_idx += 1
        input_idx += 1

    if seg_idx == 0:
        raise ValueError("No segments to keep")

    concat_str = "".join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={seg_idx}:v=1:a=0[out]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")


def compute_stats_for_column(values: np.ndarray) -> dict:
    """Compute statistics for a single column (matching lerobot format)."""
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


def trim_dataset(
    dataset_path: str | Path,
    dry_run: bool = False,
    threshold: float = 0.15,
    window: int = 3,
    min_trim: int = 5,
    backup: bool = True,
):
    """Trim initial pause frames from a LeRobot v3.0 dataset.

    Args:
        dataset_path: Path to the dataset root directory.
        dry_run: If True, only print what would be trimmed without modifying.
        threshold: Action diff threshold to detect motion.
        window: Number of consecutive frames that must exceed threshold.
        min_trim: Minimum number of pause frames to trigger trimming.
        backup: If True, create a backup before modifying.

    Returns:
        dict mapping episode_index -> number of frames trimmed.
    """
    dataset_path = Path(dataset_path).resolve()
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_path.name}")
    print(f"{'='*60}")

    # Load info
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"  SKIP: {info_path} not found")
        return {}

    with open(info_path) as f:
        info = json.load(f)

    fps = info["fps"]
    total_episodes = info["total_episodes"]

    # Load parquet data
    data_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(data_path)

    # Load episode metadata
    ep_meta_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_meta = pd.read_parquet(ep_meta_path)

    # ---- Step 1: Detect pause frames per episode ----
    episode_trims = {}
    for ep_idx in range(total_episodes):
        ep_data = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        if len(ep_data) == 0:
            episode_trims[ep_idx] = 0
            continue

        actions = np.array(ep_data["action"].tolist())
        trim_count = detect_pause_end(actions, threshold=threshold, window=window)

        if trim_count < min_trim:
            trim_count = 0

        episode_trims[ep_idx] = trim_count

    total_trimmed = sum(episode_trims.values())
    trimmed_episodes = sum(1 for v in episode_trims.values() if v > 0)

    # Print summary
    for ep_idx, trim_count in episode_trims.items():
        if trim_count > 0:
            ep_len = len(df[df["episode_index"] == ep_idx])
            print(
                f"  Episode {ep_idx:3d}: trim {trim_count:3d} frames "
                f"({trim_count/fps:.2f}s), {ep_len} -> {ep_len - trim_count}"
            )

    print(
        f"  Summary: trim {total_trimmed} frames from {trimmed_episodes}/{total_episodes} episodes "
        f"({total_trimmed/fps:.2f}s total)"
    )

    if total_trimmed == 0:
        print("  Nothing to trim.")
        return episode_trims

    if dry_run:
        print("  [DRY RUN] No changes made.")
        return episode_trims

    # ---- Step 2: Create backup ----
    if backup:
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup_before_trim"
        if not backup_path.exists():
            print(f"  Creating backup at {backup_path.name} ...")
            shutil.copytree(dataset_path, backup_path)
        else:
            print(f"  Backup already exists at {backup_path.name}, skipping backup.")

    # ---- Step 3: Trim parquet data ----
    print("  Trimming parquet data...")
    rows_to_keep = []
    for ep_idx in range(total_episodes):
        ep_mask = df["episode_index"] == ep_idx
        ep_indices = df.index[ep_mask].tolist()
        trim_count = episode_trims[ep_idx]
        rows_to_keep.extend(ep_indices[trim_count:])

    df_trimmed = df.loc[rows_to_keep].copy()

    # Re-index: reset frame_index per episode, reset global index, reset timestamp
    new_frame_indices = []
    new_timestamps = []
    new_global_index = []
    global_idx = 0
    for ep_idx in range(total_episodes):
        ep_mask = df_trimmed["episode_index"] == ep_idx
        ep_len = ep_mask.sum()
        new_frame_indices.extend(range(ep_len))
        new_timestamps.extend([i / fps for i in range(ep_len)])
        new_global_index.extend(range(global_idx, global_idx + ep_len))
        global_idx += ep_len

    df_trimmed["frame_index"] = new_frame_indices
    df_trimmed["timestamp"] = new_timestamps
    df_trimmed["index"] = new_global_index
    df_trimmed = df_trimmed.reset_index(drop=True)

    # Save trimmed parquet
    df_trimmed.to_parquet(data_path)
    print(f"  Saved trimmed parquet: {len(df)} -> {len(df_trimmed)} rows")

    # ---- Step 4: Trim videos ----
    video_keys = get_video_keys(info)
    for video_key in video_keys:
        print(f"  Trimming video: {video_key} ...")

        vcodec = info["features"][video_key]["info"].get("video.codec", "av1")
        pix_fmt = info["features"][video_key]["info"].get("video.pix_fmt", "yuv420p")
        codec_map = {"av1": "libsvtav1", "h264": "libx264", "h265": "libx265"}
        ffmpeg_codec = codec_map.get(vcodec, "libsvtav1")

        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"

        # Group episodes by their source video file, sorted by (chunk, file)
        file_groups = ep_meta.groupby([chunk_col, file_col], sort=True)

        input_paths_with_segments = []
        for (chunk_idx, file_idx), group in file_groups:
            video_path = (
                dataset_path / "videos" / video_key
                / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.mp4"
            )
            if not video_path.exists():
                print(f"    WARN: Video not found: {video_path}")
                continue

            segments = []
            for _, row in group.sort_values("episode_index").iterrows():
                ep_idx = int(row["episode_index"])
                from_ts = row[f"videos/{video_key}/from_timestamp"]
                to_ts = row[f"videos/{video_key}/to_timestamp"]
                trim_count = episode_trims[ep_idx]
                new_from_ts = from_ts + trim_count / fps
                if new_from_ts < to_ts:
                    segments.append((new_from_ts, to_ts))

            input_paths_with_segments.append((video_path, segments))

        if not any(segs for _, segs in input_paths_with_segments):
            print(f"    WARN: No video segments to keep for {video_key}")
            continue

        # All trimmed segments are consolidated into a single file-000.mp4
        output_path = dataset_path / "videos" / video_key / "chunk-000" / "file-000.mp4"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            trim_video_with_ffmpeg(
                input_paths_with_segments, tmp_path, fps, ffmpeg_codec, pix_fmt
            )
            shutil.move(str(tmp_path), str(output_path))
            print(f"    Done: {video_key}")
        except Exception as e:
            print(f"    ERROR trimming {video_key}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Remove leftover video files (file-001.mp4, etc.) after consolidation
        video_dir = dataset_path / "videos" / video_key
        for chunk_dir in sorted(video_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for f in sorted(chunk_dir.iterdir()):
                if f.suffix == ".mp4" and f != output_path:
                    f.unlink()
                    print(f"    Removed leftover: {f.relative_to(dataset_path)}")

    # ---- Step 5: Update episode metadata ----
    print("  Updating episode metadata...")

    # Recompute episode lengths and timestamps
    new_ep_meta = ep_meta.copy()
    cumulative_frames = 0
    cumulative_ts = 0.0

    for ep_idx in range(total_episodes):
        trim_count = episode_trims[ep_idx]
        old_length = int(ep_meta.iloc[ep_idx]["length"])
        new_length = old_length - trim_count

        new_ep_meta.at[ep_idx, "length"] = new_length
        new_ep_meta.at[ep_idx, "dataset_from_index"] = cumulative_frames
        new_ep_meta.at[ep_idx, "dataset_to_index"] = cumulative_frames + new_length

        new_from_ts = cumulative_ts
        new_to_ts = cumulative_ts + new_length / fps

        for video_key in video_keys:
            new_ep_meta.at[ep_idx, f"videos/{video_key}/chunk_index"] = 0
            new_ep_meta.at[ep_idx, f"videos/{video_key}/file_index"] = 0
            new_ep_meta.at[ep_idx, f"videos/{video_key}/from_timestamp"] = new_from_ts
            new_ep_meta.at[ep_idx, f"videos/{video_key}/to_timestamp"] = new_to_ts

        # Recompute per-episode stats for numeric columns
        ep_data = df_trimmed[df_trimmed["episode_index"] == ep_idx]
        for col in ["action", "observation.state"]:
            if col in df_trimmed.columns:
                values = np.array(ep_data[col].tolist())
                if len(values) > 0:
                    stats = compute_stats_for_column(values)
                    for stat_name, stat_val in stats.items():
                        col_name = f"stats/{col}/{stat_name}"
                        if col_name in new_ep_meta.columns:
                            new_ep_meta.at[ep_idx, col_name] = stat_val

        # Update scalar column stats
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df_trimmed.columns:
                values = ep_data[col].values.astype(float)
                if len(values) > 0:
                    stats = compute_stats_for_column(values.reshape(-1, 1))
                    for stat_name, stat_val in stats.items():
                        col_name = f"stats/{col}/{stat_name}"
                        if col_name in new_ep_meta.columns:
                            new_ep_meta.at[ep_idx, col_name] = stat_val

        cumulative_frames += new_length
        cumulative_ts = new_to_ts

    new_ep_meta.to_parquet(ep_meta_path)

    # ---- Step 6: Update info.json ----
    info["total_frames"] = len(df_trimmed)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # ---- Step 7: Update global stats.json ----
    print("  Updating global stats...")
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        global_stats = {}
        for col in ["action", "observation.state"]:
            if col in df_trimmed.columns:
                values = np.array(df_trimmed[col].tolist())
                global_stats[col] = compute_stats_for_column(values)

        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df_trimmed.columns:
                values = df_trimmed[col].values.astype(float).reshape(-1, 1)
                global_stats[col] = compute_stats_for_column(values)

        # Keep image stats from original (pixel stats don't change much)
        with open(stats_path) as f:
            old_stats = json.load(f)
        for key in old_stats:
            if key not in global_stats:
                global_stats[key] = old_stats[key]

        with open(stats_path, "w") as f:
            json.dump(global_stats, f, indent=2)

    print(f"  Done! Trimmed {total_trimmed} frames ({total_trimmed/fps:.2f}s)")
    return episode_trims


def main():
    parser = argparse.ArgumentParser(description="Trim initial pause from LeRobot datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_path", type=str, help="Path to a single dataset")
    group.add_argument("--dataset_list", type=str, help="File with one dataset path per line")

    parser.add_argument("--dry_run", action="store_true", help="Only show what would be trimmed")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Action diff threshold for motion detection (default: 0.15)")
    parser.add_argument("--window", type=int, default=3,
                        help="Consecutive frames needed to confirm motion (default: 3)")
    parser.add_argument("--min_trim", type=int, default=5,
                        help="Minimum pause frames to trigger trimming (default: 5)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Skip creating backup before trimming")

    args = parser.parse_args()

    if args.dataset_path:
        dataset_paths = [args.dataset_path]
    else:
        with open(args.dataset_list) as f:
            dataset_paths = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

    for path in dataset_paths:
        path = Path(path)
        if not path.exists():
            print(f"\nSKIP: {path} does not exist")
            continue
        try:
            trim_dataset(
                path,
                dry_run=args.dry_run,
                threshold=args.threshold,
                window=args.window,
                min_trim=args.min_trim,
                backup=not args.no_backup,
            )
        except Exception as e:
            print(f"\nERROR processing {path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
