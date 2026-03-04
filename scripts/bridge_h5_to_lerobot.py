#!/usr/bin/env python
"""
Convert Bridge h5 dataset to LeRobot v3.0 format with absolute EEF action space.

Optimizations vs naive approach:
  1. h264 ultrafast encoding (3x faster than SVT-AV1)
  2. No per-pixel image statistics (Pi0.5 uses IDENTITY for VISUAL, stats not needed)
  3. Multi-process: each chunk (1000 episodes) processed by an independent worker
  4. Direct numpy->video encoding (no PNG intermediate)

Usage:
    python scripts/bridge_h5_to_lerobot.py \
        --h5_dir /VLA-Data/scripts/lianqing/data/openX/x-vla/bridge \
        --output_dir /VLA-Data/scripts/lianqing/data/lerobot/bridge_abs_eef \
        --nproc 8

    Use --max_episodes=N to limit conversion for testing (0 = all).
    Use --codec=libsvtav1 if you want AV1 (slower but smaller files).
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BRIDGE_FPS = 5
CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
VIDEO_KEY = "observation.images.top"
IMAGE_KEY_H5 = "observation/image_0"
DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
Q_NAMES = ["q01", "q10", "q50", "q90", "q99"]

# --------------------------------------------------------------------------- #
#  Per-chunk worker (runs in a subprocess)
# --------------------------------------------------------------------------- #

def _process_chunk(
    chunk_idx: int,
    h5_paths: list[str],
    ep_idx_start: int,
    global_frame_start: int,
    output_dir: str,
    codec: str,
) -> dict:
    """Process one chunk of episodes independently. Returns metadata dict."""
    output_dir = Path(output_dir)
    video_base = output_dir / "videos" / VIDEO_KEY
    vpath = video_base / f"chunk-{chunk_idx:03d}" / "file-000.mp4"
    vpath.parent.mkdir(parents=True, exist_ok=True)

    encoder = None
    ep_idx = ep_idx_start
    gf = global_frame_start
    rows = []
    ep_metas = []
    tasks = set()
    skipped = 0

    for h5_str in h5_paths:
        h5_path = Path(h5_str)
        try:
            with h5py.File(h5_path, "r") as f:
                if IMAGE_KEY_H5 not in f or "proprio" not in f or "action" not in f:
                    skipped += 1
                    continue
                images = f[IMAGE_KEY_H5][:]
                proprio = f["proprio"][:]
                action_raw = f["action"][:]
                T = proprio.shape[0]
                if T < 2:
                    skipped += 1
                    continue
                instruction = str(f.attrs.get("instruction", ""))
                if not instruction.strip():
                    instruction = "robot manipulation task"
        except Exception as e:
            logger.error(f"[chunk {chunk_idx}] read error {h5_path.name}: {e}")
            skipped += 1
            continue

        # Lazy-init encoder on first valid episode (need H, W)
        if encoder is None:
            _, H, W, _ = images.shape
            container = av.open(str(vpath), mode="w")
            stream = container.add_stream(codec, rate=BRIDGE_FPS)
            stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
            if codec == "libx264":
                stream.options = {"crf": "23", "preset": "ultrafast", "g": "2"}
            elif codec == "libsvtav1":
                stream.options = {"crf": "30", "preset": "10", "g": "2"}
            elif codec == "h264_nvenc":
                stream.options = {"preset": "p1", "rc": "constqp", "qp": "23", "g": "2"}
            encoder = (container, stream, 0)  # (container, stream, total_frames)

        container, stream, total_frames = encoder
        from_ts = total_frames / BRIDGE_FPS

        for t in range(T):
            frame = av.VideoFrame.from_ndarray(images[t], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        total_frames += T
        encoder = (container, stream, total_frames)
        to_ts = total_frames / BRIDGE_FPS

        tasks.add(instruction)
        # Position/rotation from proprio (absolute), gripper from action (commanded)
        gripper_cmd = np.clip(action_raw[:, -1:], 0, 1).astype(np.float32)
        state = np.concatenate([proprio[:, :6].astype(np.float32), gripper_cmd], axis=-1)

        for t in range(T):
            rows.append({
                "observation.state": state[t],
                "action": state[t],
                "timestamp": np.float32(t / BRIDGE_FPS),
                "frame_index": t,
                "episode_index": ep_idx,
                "index": gf + t,
                "task_index": -1,
                "_task": instruction,
            })

        # Per-episode stats (state/action only — images use IDENTITY normalization)
        ep_stats = _episode_stats(state, T, ep_idx, gf)

        ep_metas.append({
            "episode_index": ep_idx,
            "tasks": [instruction],
            "length": T,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,
            "dataset_from_index": gf,
            "dataset_to_index": gf + T,
            f"videos/{VIDEO_KEY}/chunk_index": chunk_idx,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": from_ts,
            f"videos/{VIDEO_KEY}/to_timestamp": to_ts,
            **ep_stats,
        })

        gf += T
        ep_idx += 1

    # Flush encoder
    if encoder is not None:
        container, stream, _ = encoder
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

    # Write data parquet
    if rows:
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "observation.state": pa.array([r["observation.state"] for r in rows], type=pa.list_(pa.float32())),
            "action": pa.array([r["action"] for r in rows], type=pa.list_(pa.float32())),
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
            "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "index": pa.array([r["index"] for r in rows], type=pa.int64()),
            "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
            "_task": pa.array([r["_task"] for r in rows], type=pa.string()),
        })
        pq.write_table(table, chunk_dir / "file-000.parquet")

    return {
        "chunk_idx": chunk_idx,
        "num_converted": ep_idx - ep_idx_start,
        "num_skipped": skipped,
        "total_frames": gf - global_frame_start,
        "tasks": list(tasks),
        "ep_metas": ep_metas,
    }


def _episode_stats(state: np.ndarray, T: int, ep_idx: int, gf: int) -> dict:
    """Compute per-episode stats for state/action and scalar fields only."""
    s = {}
    qs = np.array(DEFAULT_QUANTILES)

    for key in ("observation.state", "action"):
        s[f"stats/{key}/min"] = state.min(0).tolist()
        s[f"stats/{key}/max"] = state.max(0).tolist()
        s[f"stats/{key}/mean"] = state.mean(0).tolist()
        s[f"stats/{key}/std"] = state.std(0).tolist()
        s[f"stats/{key}/count"] = [T]
        qvals = np.quantile(state, qs, axis=0)
        for i, qn in enumerate(Q_NAMES):
            s[f"stats/{key}/{qn}"] = qvals[i].tolist()

    ts = np.arange(T, dtype=np.float64) / BRIDGE_FPS
    for sn, sa in [("timestamp", ts),
                   ("frame_index", np.arange(T, dtype=np.float64)),
                   ("index", np.arange(gf, gf + T, dtype=np.float64)),
                   ("episode_index", np.full(T, ep_idx, dtype=np.float64)),
                   ("task_index", np.zeros(T, dtype=np.float64))]:
        s[f"stats/{sn}/min"] = [float(sa.min())]
        s[f"stats/{sn}/max"] = [float(sa.max())]
        s[f"stats/{sn}/mean"] = [float(sa.mean())]
        s[f"stats/{sn}/std"] = [float(sa.std())]
        s[f"stats/{sn}/count"] = [T]
        sv = np.quantile(sa, qs)
        for i, qn in enumerate(Q_NAMES):
            s[f"stats/{sn}/{qn}"] = [float(sv[i])]

    # Placeholder image stats (IDENTITY normalization — values don't matter)
    for stat in ["min", "max", "mean", "std"] + Q_NAMES:
        v = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.25,
             "q01": 0.0, "q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 1.0}[stat]
        s[f"stats/{VIDEO_KEY}/{stat}"] = [[v]] * 3
    s[f"stats/{VIDEO_KEY}/count"] = [T]

    return s


# --------------------------------------------------------------------------- #
#  Global aggregation
# --------------------------------------------------------------------------- #

def _aggregate_global_stats(all_ep_metas: list[dict]) -> dict:
    base_keys = set()
    for ep in all_ep_metas:
        for k in ep:
            if k.startswith("stats/") and k.endswith("/count"):
                base_keys.add(k.rsplit("/", 1)[0])

    gstats = {}
    for base in sorted(base_keys):
        feat = base.replace("stats/", "")
        counts, mins_l, maxs_l, means_l, stds_l = [], [], [], [], []
        qd = {qn: [] for qn in Q_NAMES}

        for ep in all_ep_metas:
            c = ep.get(f"{base}/count")
            if c is None:
                continue
            n = c[0] if isinstance(c, list) else c
            counts.append(n)
            mins_l.append(np.array(ep[f"{base}/min"]))
            maxs_l.append(np.array(ep[f"{base}/max"]))
            means_l.append(np.array(ep[f"{base}/mean"]))
            stds_l.append(np.array(ep[f"{base}/std"]))
            for qn in Q_NAMES:
                qd[qn].append(np.array(ep[f"{base}/{qn}"]))

        if not counts:
            continue
        c = np.array(counts, dtype=np.float64)
        tot = c.sum()
        w = c / tot

        fs = {
            "min": np.min(mins_l, axis=0).tolist(),
            "max": np.max(maxs_l, axis=0).tolist(),
            "mean": np.average(means_l, weights=w, axis=0).tolist(),
            "count": [int(tot)],
        }
        var_w = np.average(np.array(stds_l) ** 2, weights=w, axis=0)
        var_b = np.average((np.array(means_l) - np.array(fs["mean"])) ** 2, weights=w, axis=0)
        fs["std"] = np.sqrt(var_w + var_b).tolist()
        for qn in Q_NAMES:
            fs[qn] = np.median(qd[qn], axis=0).tolist()

        gstats[feat] = fs
    return gstats


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def list_h5_files(h5_dir: Path) -> list[Path]:
    files = sorted(h5_dir.glob("episode_*.hdf5"))
    if not files:
        files = sorted(h5_dir.glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(f"No episode h5 files found in {h5_dir}")
    return files


def convert(args):
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    codec = args.codec

    h5_files = list_h5_files(h5_dir)
    if args.max_episodes > 0:
        h5_files = h5_files[: args.max_episodes]

    total_eps = len(h5_files)
    logger.info(f"Episodes: {total_eps} | Workers: {args.nproc} | Codec: {codec}")
    logger.info(f"Output: {output_dir}")

    if output_dir.exists():
        logger.error(f"Output dir exists: {output_dir}  — remove it first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)

    # Split h5 files into chunks of CHUNKS_SIZE
    chunks = []
    for i in range(0, total_eps, CHUNKS_SIZE):
        chunks.append(h5_files[i : i + CHUNKS_SIZE])

    logger.info(f"Split into {len(chunks)} chunks of up to {CHUNKS_SIZE} episodes")

    # Pre-compute global_frame_start and ep_idx_start per chunk.
    # We need a first pass to know T for each episode — too expensive.
    # Instead, we'll assign ep_idx sequentially per chunk and fix global offsets after.
    # Strategy: each worker returns its results; we do a global fixup pass.

    # Launch workers
    all_results = [None] * len(chunks)

    if args.nproc <= 1:
        for ci, chunk_files in enumerate(tqdm(chunks, desc="Chunks")):
            all_results[ci] = _process_chunk(
                ci, [str(p) for p in chunk_files], 0, 0, str(output_dir), codec
            )
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.nproc) as pool:
            for ci, chunk_files in enumerate(chunks):
                fut = pool.submit(
                    _process_chunk, ci, [str(p) for p in chunk_files], 0, 0,
                    str(output_dir), codec,
                )
                futures[fut] = ci

            with tqdm(total=len(chunks), desc="Chunks") as pbar:
                for fut in as_completed(futures):
                    ci = futures[fut]
                    all_results[ci] = fut.result()
                    pbar.update(1)

    # Global fixup: re-number episode_index and index across chunks
    logger.info("Re-numbering episode/frame indices globally...")
    ep_offset = 0
    frame_offset = 0
    all_tasks = set()
    all_ep_metas = []

    for ci, res in enumerate(all_results):
        if res is None or res["num_converted"] == 0:
            continue

        all_tasks.update(res["tasks"])

        # Fix episode metadata
        for em in res["ep_metas"]:
            em["episode_index"] += ep_offset
            em["dataset_from_index"] += frame_offset
            em["dataset_to_index"] += frame_offset
            # Fix stats that reference ep_idx or global index
            if f"stats/episode_index/min" in em:
                em[f"stats/episode_index/min"] = [float(em["episode_index"])]
                em[f"stats/episode_index/max"] = [float(em["episode_index"])]
                em[f"stats/episode_index/mean"] = [float(em["episode_index"])]
                em[f"stats/episode_index/std"] = [0.0]
                for qn in Q_NAMES:
                    em[f"stats/episode_index/{qn}"] = [float(em["episode_index"])]
            if f"stats/index/min" in em:
                T = em["length"]
                em[f"stats/index/min"] = [float(em["dataset_from_index"])]
                em[f"stats/index/max"] = [float(em["dataset_to_index"] - 1)]
                em[f"stats/index/mean"] = [float(em["dataset_from_index"] + (T - 1) / 2)]
            all_ep_metas.append(em)

        # Fix parquet file
        pq_path = output_dir / "data" / f"chunk-{ci:03d}" / "file-000.parquet"
        if pq_path.exists():
            tbl = pq.read_table(pq_path)
            ep_col = tbl.column("episode_index").to_pylist()
            idx_col = tbl.column("index").to_pylist()
            new_ep = [e + ep_offset for e in ep_col]
            new_idx = [i + frame_offset for i in idx_col]
            tbl = tbl.set_column(tbl.schema.get_field_index("episode_index"),
                                 "episode_index", pa.array(new_idx, type=pa.int64()))
            tbl = tbl.drop("episode_index")
            tbl = tbl.append_column("episode_index", pa.array(new_ep, type=pa.int64()))
            tbl = tbl.drop("index")
            tbl = tbl.append_column("index", pa.array(new_idx, type=pa.int64()))
            pq.write_table(tbl, pq_path)

        ep_offset += res["num_converted"]
        frame_offset += res["total_frames"]

    total_converted = ep_offset
    total_frames = frame_offset

    if total_converted == 0:
        logger.error("No episodes converted!")
        sys.exit(1)

    logger.info(f"Converted {total_converted} episodes, {total_frames} frames")

    # Build task mapping and fix task_index in all parquets
    task_list = sorted(all_tasks)
    task_to_idx = {t: i for i, t in enumerate(task_list)}

    logger.info(f"Fixing task indices across {len(chunks)} chunk parquets...")
    for ci in range(len(chunks)):
        pq_path = output_dir / "data" / f"chunk-{ci:03d}" / "file-000.parquet"
        if not pq_path.exists():
            continue
        tbl = pq.read_table(pq_path)
        if "_task" in tbl.schema.names:
            task_col = tbl.column("_task").to_pylist()
            tidx = [task_to_idx[t] for t in task_col]
            tbl = tbl.drop("_task").drop("task_index")
            tbl = tbl.append_column("task_index", pa.array(tidx, type=pa.int64()))
            pq.write_table(tbl, pq_path)

    # Write tasks parquet
    tasks_df = pd.DataFrame({"task_index": list(range(len(task_list)))}, index=task_list)
    tasks_df.to_parquet(output_dir / "meta" / "tasks.parquet")
    logger.info(f"Tasks: {len(task_list)}")

    # Write episode metadata parquets
    logger.info("Writing episode metadata...")
    ep_by_chunk: dict[int, list] = {}
    for em in all_ep_metas:
        c = em["data/chunk_index"]
        ep_by_chunk.setdefault(c, []).append(em)

    for ci, eps in ep_by_chunk.items():
        d = output_dir / "meta" / "episodes" / f"chunk-{ci:03d}"
        d.mkdir(parents=True, exist_ok=True)
        records = []
        for em in eps:
            rec = dict(em)
            rec["meta/episodes/chunk_index"] = ci
            rec["meta/episodes/file_index"] = 0
            records.append(rec)
        pd.DataFrame(records).to_parquet(d / "file-000.parquet")

    # Global stats
    logger.info("Aggregating global stats...")
    gstats = _aggregate_global_stats(all_ep_metas)
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(gstats, f, indent=2)

    # info.json
    video_codec_name = "h264" if codec == "libx264" else "av1"
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "widow_x",
        "total_episodes": total_converted,
        "total_frames": total_frames,
        "total_tasks": len(task_list),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": 500,
        "video_files_size_in_mb": 2000,
        "fps": BRIDGE_FPS,
        "splits": {"train": f"0:{total_converted}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 256, "video.width": 256,
                    "video.codec": video_codec_name,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": BRIDGE_FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32", "shape": [7],
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Done! {output_dir}")
    logger.info(f"  Episodes: {total_converted} | Frames: {total_frames} | Tasks: {len(task_list)}")


def main():
    parser = argparse.ArgumentParser(description="Convert Bridge h5 to LeRobot v3.0 (abs EEF)")
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="bridge_abs_eef")
    parser.add_argument("--max_episodes", type=int, default=0, help="0=all")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel workers (one per chunk)")
    parser.add_argument("--codec", type=str, default="libx264",
                        choices=["libx264", "libsvtav1", "h264_nvenc"], help="Video codec")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
