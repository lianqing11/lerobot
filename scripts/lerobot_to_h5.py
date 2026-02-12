"""
Convert LeRobot dataset to H5 format (one hdf5 file per episode).

Output H5 structure per episode:
    - observation/image_0: [T, H, W, 3] uint8 (first camera)
    - observation/image_1: [T, H, W, 3] uint8 (second camera, if exists)
    - ...
    - proprio: [T, D] float32 (observation.state)
    - action: [T, D] float32
    - instruction: str (attribute)
    - fps: int (attribute)
    - repo_id: str (attribute)

Filename convention: {instruction}_{collect_date}_{episode_id}.h5
    - instruction: sanitized task string (spaces -> underscores, max 50 chars)
    - collect_date: from --collect_date arg or defaults to today (YYYYMMDD)
    - episode_id: zero-padded 6 digits

Usage:
    # From HuggingFace Hub (downloads to ~/.cache/huggingface/lerobot/)
    python scripts/lerobot_to_h5.py \
        --repo_id lerobot/aloha_sim_insertion_human_image \
        --output_dir /path/to/output \
        --collect_date 20250101 \
        --nproc 8

    # From local dataset directory
    python scripts/lerobot_to_h5.py \
        --repo_id my_dataset \
        --root /path/to/local/dataset \
        --output_dir /path/to/output \
        --nproc 8

    # Only convert specific episodes
    python scripts/lerobot_to_h5.py \
        --repo_id lerobot/aloha_sim_insertion_human_image \
        --output_dir /path/to/output \
        --episodes 0 1 2 3 \
        --nproc 4
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def load_tasks(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "meta" / "tasks.parquet")


def load_episodes_metadata(root: Path):
    """Load episode metadata from meta/episodes/ parquet files."""
    episodes_dir = root / "meta" / "episodes"
    paths = sorted(episodes_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found in {episodes_dir}")
    tables = [pq.read_table(p) for p in paths]
    import pyarrow as pa
    combined = pa.concat_tables(tables).to_pandas()
    return combined


def load_parquet_data(root: Path):
    """Load all parquet data files into a single pandas DataFrame."""
    data_dir = root / "data"
    paths = sorted(data_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found in {data_dir}")
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def decode_all_video_frames(video_path: str, from_ts: float, to_ts: float, fps: int) -> np.ndarray:
    """Decode all frames for an episode from an MP4 file.

    Returns [T, H, W, 3] uint8 numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(from_ts * video_fps))
    end_frame = int(round(to_ts * video_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path} [{from_ts:.3f}s - {to_ts:.3f}s]")

    return np.stack(frames, axis=0)


def sanitize_instruction(instruction: str, max_len: int = 50) -> str:
    """Sanitize instruction string for use as filename component."""
    s = instruction.strip().lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len] if s else "no_task"


def save_episode_h5(output_path: str, images_dict: dict, action: np.ndarray,
                    proprio: np.ndarray, instruction: str, fps: int, repo_id: str):
    """Save one episode to an H5 file."""
    with h5py.File(output_path, 'w') as f:
        obs_grp = f.create_group('observation')
        for cam_name, imgs in images_dict.items():
            obs_grp.create_dataset(cam_name, data=imgs, compression='gzip', compression_opts=4)

        f.create_dataset('action', data=action, dtype=np.float32)

        if proprio is not None and len(proprio) > 0:
            f.create_dataset('proprio', data=proprio, dtype=np.float32)

        f.attrs['instruction'] = instruction
        f.attrs['fps'] = fps
        f.attrs['repo_id'] = repo_id


def process_episode(args):
    """Worker function for multiprocessing. Processes and saves one episode."""
    (ep_idx, ep_meta, ep_frames, info, tasks_df, video_keys, image_keys,
     camera_keys, root, output_dir, collect_date, repo_id) = args

    fps = info["fps"]
    features = info["features"]

    # --- Get instruction ---
    task_indices = ep_frames["task_index"].unique()
    task_idx = int(task_indices[0])
    # tasks_df is indexed by task name, with a "task_index" column
    matching = tasks_df[tasks_df["task_index"] == task_idx]
    instruction = str(matching.index[0]) if len(matching) > 0 else ""

    # --- Build filename ---
    sanitized = sanitize_instruction(instruction)
    filename = f"{sanitized}_{collect_date}_{ep_idx:06d}.h5"
    output_path = os.path.join(output_dir, filename)

    # --- Extract action ---
    action_col = "action" if "action" in ep_frames.columns else None
    action = np.stack(ep_frames[action_col].values) if action_col else np.array([])

    # --- Extract proprio (observation.state) ---
    state_col = None
    for col in ep_frames.columns:
        if col == "observation.state":
            state_col = col
            break
    proprio = np.stack(ep_frames[state_col].values) if state_col else np.array([])

    # --- Extract images ---
    images_dict = {}
    cam_idx = 0

    # Handle video keys: decode from MP4
    for vid_key in video_keys:
        chunk_key = f"videos/{vid_key}/chunk_index"
        file_key = f"videos/{vid_key}/file_index"
        from_ts_key = f"videos/{vid_key}/from_timestamp"
        to_ts_key = f"videos/{vid_key}/to_timestamp"

        # Get video location from episode metadata
        chunk_idx = int(ep_meta.get(chunk_key, ep_meta.get("data/chunk_index", 0)))
        file_idx = int(ep_meta.get(file_key, ep_meta.get("data/file_index", 0)))

        video_path_template = info.get("video_path", "")
        video_path = os.path.join(
            str(root),
            video_path_template.format(
                video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx
            )
        )

        from_ts = float(ep_meta.get(from_ts_key, 0.0))
        to_ts = float(ep_meta.get(to_ts_key, 0.0))

        if os.path.exists(video_path) and to_ts > from_ts:
            try:
                frames = decode_all_video_frames(video_path, from_ts, to_ts, fps)
                images_dict[f"image_{cam_idx}"] = frames
                cam_idx += 1
            except Exception as e:
                print(f"Warning: Failed to decode video {vid_key} for episode {ep_idx}: {e}")
        else:
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")

    # Handle image keys: read from parquet (stored as PIL or dicts with path/bytes)
    for img_key in image_keys:
        if img_key in ep_frames.columns:
            img_data = ep_frames[img_key].values
            frames = []
            for item in img_data:
                if isinstance(item, dict):
                    # HF datasets Image format: {"bytes": ..., "path": ...}
                    if "bytes" in item and item["bytes"] is not None:
                        import io
                        from PIL import Image
                        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                        frames.append(np.array(img, dtype=np.uint8))
                    elif "path" in item and item["path"] is not None:
                        img_path = item["path"]
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(str(root), img_path)
                        img = cv2.imread(img_path)
                        if img is not None:
                            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    # Might be raw numpy or PIL
                    from PIL import Image as PILImage
                    if isinstance(item, PILImage.Image):
                        frames.append(np.array(item.convert("RGB"), dtype=np.uint8))
                    elif isinstance(item, np.ndarray):
                        frames.append(item if item.dtype == np.uint8 else (item * 255).astype(np.uint8))

            if frames:
                images_dict[f"image_{cam_idx}"] = np.stack(frames, axis=0)
                cam_idx += 1

    save_episode_h5(output_path, images_dict, action, proprio, instruction, fps, repo_id)
    return ep_idx, filename


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to H5 format")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo id or local dataset name")
    parser.add_argument("--root", type=str, default=None,
                        help="Local dataset root dir. If not set, uses ~/.cache/huggingface/lerobot/<repo_id>")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save H5 files")
    parser.add_argument("--collect_date", type=str, default=None,
                        help="Collection date string (YYYYMMDD). Defaults to today.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices to convert. Defaults to all.")
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of worker processes (default: 8)")
    args = parser.parse_args()

    # Resolve root
    if args.root:
        root = Path(args.root)
    else:
        root = Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
    if not root.exists():
        print(f"Error: Dataset root not found: {root}")
        print("Either download the dataset first or specify --root for a local dataset.")
        sys.exit(1)

    collect_date = args.collect_date or datetime.now().strftime("%Y%m%d")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    print(f"Loading dataset from {root} ...")
    info = load_info(root)
    tasks_df = load_tasks(root)
    episodes_meta = load_episodes_metadata(root)
    features = info["features"]

    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    image_keys = [k for k, v in features.items() if v["dtype"] == "image"]
    camera_keys = video_keys + image_keys

    print(f"  repo_id:    {args.repo_id}")
    print(f"  episodes:   {info['total_episodes']}")
    print(f"  frames:     {info['total_frames']}")
    print(f"  fps:        {info['fps']}")
    print(f"  video_keys: {video_keys}")
    print(f"  image_keys: {image_keys}")
    print(f"  features:   {list(features.keys())}")

    # Load parquet data
    print("Loading parquet data ...")
    df = load_parquet_data(root)

    # Determine which episodes to convert
    all_ep_indices = sorted(df["episode_index"].unique())
    if args.episodes is not None:
        ep_indices = [e for e in args.episodes if e in all_ep_indices]
        if len(ep_indices) != len(args.episodes):
            missing = set(args.episodes) - set(ep_indices)
            print(f"Warning: Episodes not found in dataset: {missing}")
    else:
        ep_indices = all_ep_indices

    print(f"Converting {len(ep_indices)} episodes with {args.nproc} workers ...")

    # Pre-group dataframe by episode for fast access
    grouped = {ep_idx: group for ep_idx, group in df.groupby("episode_index") if ep_idx in ep_indices}

    # Build episode metadata lookup (episodes_meta is a DataFrame, indexed by row order = episode_index)
    ep_meta_list = []
    for ep_idx in ep_indices:
        if ep_idx < len(episodes_meta):
            ep_meta_list.append(episodes_meta.iloc[ep_idx].to_dict())
        else:
            ep_meta_list.append({})

    # Prepare args for each worker
    worker_args = []
    for i, ep_idx in enumerate(ep_indices):
        worker_args.append((
            ep_idx, ep_meta_list[i], grouped[ep_idx], info, tasks_df,
            video_keys, image_keys, camera_keys,
            root, args.output_dir, collect_date, args.repo_id
        ))

    # Run with multiprocessing
    if args.nproc <= 1:
        results = []
        for wa in tqdm(worker_args, desc="Converting"):
            results.append(process_episode(wa))
    else:
        results = []
        with Pool(processes=args.nproc) as pool:
            for result in tqdm(
                pool.imap_unordered(process_episode, worker_args),
                total=len(worker_args),
                desc="Converting"
            ):
                results.append(result)

    print(f"\nDone! Saved {len(results)} episodes to {args.output_dir}")

    # Verify first file
    if results:
        first_ep_idx, first_filename = sorted(results, key=lambda x: x[0])[0]
        first_path = os.path.join(args.output_dir, first_filename)
        print(f"\nVerifying {first_path} ...")
        with h5py.File(first_path, 'r') as f:
            print(f"  Keys: {list(f.keys())}")
            print(f"  Instruction: '{f.attrs.get('instruction', '')}'")
            print(f"  FPS: {f.attrs.get('fps', 'N/A')}")
            if 'observation' in f:
                for cam in f['observation']:
                    print(f"  observation/{cam}: {f['observation'][cam].shape} {f['observation'][cam].dtype}")
            if 'action' in f:
                print(f"  action: {f['action'].shape} {f['action'].dtype}")
            if 'proprio' in f:
                print(f"  proprio: {f['proprio'].shape} {f['proprio'].dtype}")


if __name__ == "__main__":
    main()
