#!/usr/bin/env python3
"""
Web-based LeRobot dataset visualizer and instruction editor.

Usage:
    python visualize_lerobot_web.py /path/to/dataset [--port 8765]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uvicorn import run as uvicorn_run

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="LeRobot Dataset Web Visualizer")
parser.add_argument("dataset_path", type=str, help="Path to LeRobot dataset root")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--host", type=str, default="0.0.0.0")
args = parser.parse_args()

DATASET_ROOT = Path(args.dataset_path).expanduser().resolve()
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

# ── Data loading helpers ──────────────────────────────────────────────────────


def load_info() -> dict:
    with open(DATASET_ROOT / "meta" / "info.json") as f:
        return json.load(f)


def load_tasks() -> pd.DataFrame:
    return pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")


def load_episodes() -> pd.DataFrame:
    ep_dir = DATASET_ROOT / "meta" / "episodes"
    parts = sorted(ep_dir.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError("No episode parquet files found")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def save_tasks(tasks_df: pd.DataFrame):
    out = DATASET_ROOT / "meta" / "tasks.parquet"
    tasks_df.to_parquet(out, index=True)


def save_episodes(episodes_df: pd.DataFrame):
    """Save episodes back, preserving the original chunk/file split layout."""
    ep_dir = DATASET_ROOT / "meta" / "episodes"
    parts = sorted(ep_dir.rglob("*.parquet"))
    if len(parts) == 1:
        episodes_df.to_parquet(parts[0], index=False)
    else:
        for part_path in parts:
            chunk_str = part_path.parent.name
            chunk_idx = int(chunk_str.split("-")[-1])
            file_str = part_path.stem
            file_idx = int(file_str.split("-")[-1])
            mask = (episodes_df["meta/episodes/chunk_index"] == chunk_idx) & (
                episodes_df["meta/episodes/file_index"] == file_idx
            )
            subset = episodes_df[mask]
            if len(subset) > 0:
                subset.to_parquet(part_path, index=False)


def update_data_task_index(episode_indices: list[int], new_task_index: int):
    """Update task_index in data parquet files for given episodes."""
    data_dir = DATASET_ROOT / "data"
    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        mask = df["episode_index"].isin(episode_indices)
        if mask.any():
            df.loc[mask, "task_index"] = new_task_index
            df.to_parquet(parquet_file, index=False)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="LeRobot Dataset Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(Path(__file__).parent / "visualize_lerobot_web.html")


@app.get("/api/info")
def api_info():
    info = load_info()
    return info


@app.get("/api/tasks")
def api_tasks():
    tasks_df = load_tasks()
    result = []
    for task_text, row in tasks_df.iterrows():
        result.append({"task": str(task_text), "task_index": int(row["task_index"])})
    return result


@app.get("/api/episodes")
def api_episodes():
    episodes_df = load_episodes()
    tasks_df = load_tasks()
    task_map = {int(row["task_index"]): str(task_text) for task_text, row in tasks_df.iterrows()}
    info = load_info()

    result = []
    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        tasks_list = row["tasks"]
        length = int(row["length"])

        main_chunk = int(row.get("videos/observation.images.main/chunk_index", 0))
        main_file = int(row.get("videos/observation.images.main/file_index", 0))
        main_from = float(row.get("videos/observation.images.main/from_timestamp", 0))
        main_to = float(row.get("videos/observation.images.main/to_timestamp", length / info.get("fps", 30)))

        wrist_chunk = int(row.get("videos/observation.images.wrist/chunk_index", 0))
        wrist_file = int(row.get("videos/observation.images.wrist/file_index", 0))
        wrist_from = float(row.get("videos/observation.images.wrist/from_timestamp", 0))
        wrist_to = float(row.get("videos/observation.images.wrist/to_timestamp", length / info.get("fps", 30)))

        data_chunk = int(row.get("data/chunk_index", 0))
        data_file = int(row.get("data/file_index", 0))

        if isinstance(tasks_list, np.ndarray):
            tasks_list = tasks_list.tolist()
        elif not isinstance(tasks_list, list):
            tasks_list = [str(tasks_list)]

        result.append({
            "episode_index": ep_idx,
            "tasks": tasks_list,
            "length": length,
            "main_video": {
                "chunk": main_chunk,
                "file": main_file,
                "from_ts": main_from,
                "to_ts": main_to,
            },
            "wrist_video": {
                "chunk": wrist_chunk,
                "file": wrist_file,
                "from_ts": wrist_from,
                "to_ts": wrist_to,
            },
            "data_chunk": data_chunk,
            "data_file": data_file,
        })

    return result


@app.get("/api/video/{camera}/{chunk_idx}/{file_idx}")
def api_video(camera: str, chunk_idx: int, file_idx: int):
    video_key = f"observation.images.{camera}"
    video_path = DATASET_ROOT / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    return FileResponse(video_path, media_type="video/mp4")


class TaskUpdateRequest(BaseModel):
    task_index: int
    new_text: str


@app.post("/api/tasks/update")
def api_update_task(req: TaskUpdateRequest):
    """Update the text of an existing task."""
    backup_metadata()
    tasks_df = load_tasks()
    found = False
    new_index = tasks_df.index.tolist()
    new_task_indices = tasks_df["task_index"].tolist()

    for i, (task_text, row) in enumerate(tasks_df.iterrows()):
        if int(row["task_index"]) == req.task_index:
            new_index[i] = req.new_text
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Task index {req.task_index} not found")

    tasks_df.index = pd.Index(new_index, name="task")
    save_tasks(tasks_df)

    episodes_df = load_episodes()
    for idx, row in episodes_df.iterrows():
        old_tasks = row["tasks"]
        if isinstance(old_tasks, list):
            old_task_text = None
            for t_text, t_row in load_tasks().iterrows():
                if int(t_row["task_index"]) == req.task_index:
                    old_task_text = str(t_text)
                    break
        episodes_df.at[idx, "tasks"] = [req.new_text] if isinstance(old_tasks, list) and any(
            True for t in old_tasks if t != req.new_text
        ) else old_tasks

    episodes_df_fresh = load_episodes()
    data_dir = DATASET_ROOT / "data"

    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        ep_indices = df["episode_index"].unique()
        for ep_idx in ep_indices:
            ep_data = df[df["episode_index"] == ep_idx]
            ti = int(ep_data["task_index"].iloc[0])
            if ti == req.task_index:
                mask = episodes_df_fresh["episode_index"] == int(ep_idx)
                if mask.any():
                    episodes_df_fresh.at[mask.idxmax(), "tasks"] = np.array([req.new_text], dtype=object)

    save_episodes(episodes_df_fresh)
    return {"status": "ok", "message": f"Task {req.task_index} updated to: {req.new_text}"}


class EpisodeTaskUpdateRequest(BaseModel):
    episode_index: int
    new_task_text: str


@app.post("/api/episodes/update_task")
def api_update_episode_task(req: EpisodeTaskUpdateRequest):
    """Change the task/instruction for a specific episode."""
    backup_metadata()
    tasks_df = load_tasks()
    task_map = {str(t): int(row["task_index"]) for t, row in tasks_df.iterrows()}

    if req.new_task_text in task_map:
        new_task_idx = task_map[req.new_task_text]
    else:
        new_task_idx = max(int(row["task_index"]) for _, row in tasks_df.iterrows()) + 1
        new_row = pd.DataFrame({"task_index": [new_task_idx]}, index=pd.Index([req.new_task_text], name="task"))
        tasks_df = pd.concat([tasks_df, new_row])
        save_tasks(tasks_df)

        info = load_info()
        info["total_tasks"] = len(tasks_df)
        with open(DATASET_ROOT / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    episodes_df = load_episodes()
    mask = episodes_df["episode_index"] == req.episode_index
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Episode {req.episode_index} not found")
    for idx in episodes_df[mask].index:
        episodes_df.at[idx, "tasks"] = np.array([req.new_task_text], dtype=object)
    save_episodes(episodes_df)

    update_data_task_index([req.episode_index], new_task_idx)
    return {"status": "ok", "message": f"Episode {req.episode_index} → '{req.new_task_text}' (task_index={new_task_idx})"}


class BatchTaskUpdateRequest(BaseModel):
    episode_indices: list[int]
    new_task_text: str


@app.post("/api/episodes/batch_update_task")
def api_batch_update_task(req: BatchTaskUpdateRequest):
    """Batch update task for multiple episodes."""
    backup_metadata()
    tasks_df = load_tasks()
    task_map = {str(t): int(row["task_index"]) for t, row in tasks_df.iterrows()}

    if req.new_task_text in task_map:
        new_task_idx = task_map[req.new_task_text]
    else:
        new_task_idx = max(int(row["task_index"]) for _, row in tasks_df.iterrows()) + 1
        new_row = pd.DataFrame({"task_index": [new_task_idx]}, index=pd.Index([req.new_task_text], name="task"))
        tasks_df = pd.concat([tasks_df, new_row])
        save_tasks(tasks_df)

        info = load_info()
        info["total_tasks"] = len(tasks_df)
        with open(DATASET_ROOT / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    episodes_df = load_episodes()
    mask = episodes_df["episode_index"].isin(req.episode_indices)
    for idx in episodes_df[mask].index:
        episodes_df.at[idx, "tasks"] = np.array([req.new_task_text], dtype=object)
    save_episodes(episodes_df)

    update_data_task_index(req.episode_indices, new_task_idx)
    return {
        "status": "ok",
        "message": f"{len(req.episode_indices)} episodes → '{req.new_task_text}'",
    }


def backup_metadata():
    """Create a timestamped backup of metadata before modifications."""
    backup_dir = DATASET_ROOT / "meta" / "_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        src = DATASET_ROOT / "meta"
        for f in ["info.json", "tasks.parquet", "stats.json"]:
            if (src / f).exists():
                shutil.copy2(src / f, backup_dir / f)
        ep_src = src / "episodes"
        if ep_src.exists():
            ep_dst = backup_dir / "episodes"
            shutil.copytree(ep_src, ep_dst, dirs_exist_ok=True)


if __name__ == "__main__":
    print(f"🚀 LeRobot Dataset Visualizer")
    print(f"   Dataset: {DATASET_ROOT}")
    print(f"   URL: http://{args.host}:{args.port}")
    uvicorn_run(app, host=args.host, port=args.port)
