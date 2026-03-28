#!/usr/bin/env python3
"""
LeRobot Dataset Web Visualizer
- Episode navigation with task descriptions
- Dual camera video playback (main + wrist)
- Action & observation state trajectory charts
- Per-episode statistics overview
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_file


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


app = Flask(__name__)
app.json.encoder = NumpyEncoder  # type: ignore[attr-defined]

DATASET_ROOT = None
INFO = None
EPISODES_DF = None
DATA_DF = None
TASKS_DF = None
VIDEO_CLIP_CACHE = {}


def load_dataset(root: str):
    global DATASET_ROOT, INFO, EPISODES_DF, DATA_DF, TASKS_DF
    DATASET_ROOT = Path(root)

    with open(DATASET_ROOT / "meta" / "info.json") as f:
        INFO = json.load(f)

    TASKS_DF = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")

    ep_dir = DATASET_ROOT / "meta" / "episodes"
    ep_frames = []
    for chunk_dir in sorted(ep_dir.iterdir()):
        if chunk_dir.is_dir():
            for fp in sorted(chunk_dir.iterdir()):
                if fp.suffix == ".parquet":
                    ep_frames.append(pd.read_parquet(fp))
    EPISODES_DF = pd.concat(ep_frames, ignore_index=True)

    data_dir = DATASET_ROOT / "data"
    data_frames = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir():
            for fp in sorted(chunk_dir.iterdir()):
                if fp.suffix == ".parquet":
                    data_frames.append(pd.read_parquet(fp))
    DATA_DF = pd.concat(data_frames, ignore_index=True)


def get_video_path(video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        DATASET_ROOT
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def extract_episode_clip(video_key: str, episode_index: int) -> str:
    """Extract a video clip for a specific episode using ffmpeg, with caching."""
    cache_key = (video_key, episode_index)
    if cache_key in VIDEO_CLIP_CACHE and os.path.exists(VIDEO_CLIP_CACHE[cache_key]):
        return VIDEO_CLIP_CACHE[cache_key]

    row = EPISODES_DF[EPISODES_DF["episode_index"] == episode_index].iloc[0]

    vk_col = video_key.replace(".", "_").replace("/", "_")
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"

    chunk_idx = int(row[chunk_col])
    file_idx = int(row[file_col])
    from_ts = float(row[from_col])
    to_ts = float(row[to_col])

    src = get_video_path(video_key, chunk_idx, file_idx)
    if not src.exists():
        return None

    cache_dir = Path(tempfile.gettempdir()) / "lerobot_viz_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(cache_dir / f"{video_key.replace('.', '_').replace('/', '_')}_ep{episode_index:06d}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{from_ts:.6f}",
        "-to", f"{to_ts:.6f}",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)

    VIDEO_CLIP_CACHE[cache_key] = out_path
    return out_path


# ── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/info")
def api_info():
    tasks_list = []
    for _, row in TASKS_DF.iterrows():
        tasks_list.append({"task": row.name if isinstance(row.name, str) else str(row.name),
                           "task_index": int(row["task_index"])})
    return jsonify({
        "robot_type": INFO.get("robot_type", "unknown"),
        "total_episodes": INFO["total_episodes"],
        "total_frames": INFO["total_frames"],
        "fps": INFO["fps"],
        "tasks": tasks_list,
        "features": {
            "action": INFO["features"]["action"],
            "observation.state": INFO["features"]["observation.state"],
        },
        "video_keys": [k for k, v in INFO["features"].items() if v.get("dtype") == "video"],
    })


@app.route("/api/episodes")
def api_episodes():
    records = []
    for _, row in EPISODES_DF.iterrows():
        tasks = row["tasks"]
        if hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        elif not isinstance(tasks, list):
            tasks = list(tasks)
        records.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "tasks": tasks,
        })
    return jsonify(records)


@app.route("/api/episode/<int:ep_idx>/data")
def api_episode_data(ep_idx):
    ep_data = DATA_DF[DATA_DF["episode_index"] == ep_idx]
    if ep_data.empty:
        return jsonify({"error": "Episode not found"}), 404

    actions = [a.tolist() if hasattr(a, "tolist") else list(a) for a in ep_data["action"]]
    states = [s.tolist() if hasattr(s, "tolist") else list(s) for s in ep_data["observation.state"]]
    timestamps = [float(t) for t in ep_data["timestamp"]]
    frame_indices = [int(f) for f in ep_data["frame_index"]]

    action_names = INFO["features"]["action"]["names"]
    state_names = INFO["features"]["observation.state"]["names"]

    return jsonify({
        "episode_index": ep_idx,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "actions": actions,
        "states": states,
        "action_names": action_names,
        "state_names": state_names,
    })


@app.route("/api/episode/<int:ep_idx>/video/<path:video_key>")
def api_episode_video(ep_idx, video_key):
    clip_path = extract_episode_clip(video_key, ep_idx)
    if clip_path is None or not os.path.exists(clip_path):
        return Response("Video not found", status=404)

    file_size = os.path.getsize(clip_path)
    range_header = request.headers.get("Range")

    if range_header:
        byte_start = 0
        byte_end = file_size - 1
        match = __import__("re").search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            byte_start = int(match.group(1))
            if match.group(2):
                byte_end = int(match.group(2))

        length = byte_end - byte_start + 1
        with open(clip_path, "rb") as f:
            f.seek(byte_start)
            data = f.read(length)

        resp = Response(data, status=206, mimetype="video/mp4")
        resp.headers["Content-Range"] = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    return send_file(clip_path, mimetype="video/mp4")


# ── Frontend ─────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeRobot Dataset Visualizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #252830;
    --border: #2e3140;
    --text: #e4e6eb;
    --text2: #9ca0ab;
    --accent: #4f8ff7;
    --accent2: #7c5bf5;
    --green: #34d399;
    --orange: #fb923c;
    --red: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header h1 { font-size: 18px; font-weight: 600; white-space: nowrap; }
  .header .badge {
    background: var(--accent);
    color: #fff;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }
  .header .stats {
    margin-left: auto;
    display: flex;
    gap: 20px;
    font-size: 13px;
    color: var(--text2);
  }
  .header .stats span { color: var(--text); font-weight: 600; }
  .help-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text2);
    width: 28px; height: 28px;
    border-radius: 50%;
    font-size: 14px; font-weight: 700;
    cursor: pointer;
    transition: all .15s;
    display: flex; align-items: center; justify-content: center;
  }
  .help-btn:hover { color: var(--accent); border-color: var(--accent); }
  .layout {
    display: flex;
    height: calc(100vh - 53px);
  }
  .sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sidebar-header h2 { font-size: 14px; font-weight: 600; }
  .sidebar-header .ep-count {
    margin-left: auto;
    font-size: 11px;
    color: var(--text2);
    background: var(--surface2);
    padding: 1px 8px;
    border-radius: 10px;
  }
  .filter-bar {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .filter-bar select, .filter-bar input {
    width: 100%;
    padding: 6px 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
    outline: none;
    margin-bottom: 6px;
  }
  .filter-bar select:focus, .filter-bar input:focus { border-color: var(--accent); }
  .ep-list {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }
  .ep-list::-webkit-scrollbar { width: 6px; }
  .ep-list::-webkit-scrollbar-track { background: transparent; }
  .ep-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .ep-item {
    padding: 8px 16px;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: all .15s;
  }
  .ep-item:hover { background: var(--surface2); }
  .ep-item.active {
    background: rgba(79, 143, 247, 0.1);
    border-left-color: var(--accent);
  }
  .ep-item .ep-id { font-size: 13px; font-weight: 600; }
  .ep-item .ep-meta {
    font-size: 11px;
    color: var(--text2);
    margin-top: 2px;
    display: flex;
    gap: 8px;
  }
  .main-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .main-content::-webkit-scrollbar { width: 8px; }
  .main-content::-webkit-scrollbar-track { background: transparent; }
  .main-content::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-header .nav-btns { margin-left: auto; display: flex; gap: 6px; }
  .card-header .nav-btns button {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 12px;
    cursor: pointer;
    transition: all .15s;
  }
  .card-header .nav-btns button:hover { border-color: var(--accent); color: var(--accent); }
  .card-body { padding: 16px; }
  .videos-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .video-cell { position: relative; }
  .video-cell .video-label {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,0,0,0.75);
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    z-index: 10;
  }
  .video-cell .frame-overlay {
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(0,0,0,0.75);
    color: var(--green);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    z-index: 10;
    pointer-events: none;
  }
  .video-cell video {
    width: 100%;
    border-radius: 8px;
    background: #000;
    display: block;
  }

  .transport-bar {
    margin-top: 12px;
    background: var(--surface2);
    border-radius: 10px;
    padding: 12px 16px;
  }
  .timeline-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .time-display {
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: var(--text);
    white-space: nowrap;
    min-width: 110px;
  }
  .timeline-slider {
    flex: 1;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    border-radius: 3px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }
  .timeline-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid #fff;
    box-shadow: 0 0 4px rgba(0,0,0,0.4);
  }
  .controls-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .btn {
    padding: 5px 12px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    font-weight: 500;
    transition: all .15s;
    white-space: nowrap;
  }
  .btn:hover { opacity: 0.85; }
  .btn.secondary {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn.secondary:hover { border-color: var(--accent); color: var(--accent); }
  .btn.icon-btn { padding: 5px 8px; font-size: 14px; line-height: 1; }

  .speed-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: 8px;
  }
  .speed-bar .speed-label {
    font-size: 12px;
    color: var(--text2);
    margin-right: 2px;
  }
  .speed-chip {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text2);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: all .15s;
  }
  .speed-chip:hover { border-color: var(--accent); color: var(--text); }
  .speed-chip.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .speed-adj {
    padding: 3px 7px;
    border-radius: 4px;
    font-size: 13px;
    cursor: pointer;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-weight: 700;
    line-height: 1;
    transition: all .15s;
  }
  .speed-adj:hover { border-color: var(--accent); color: var(--accent); }
  .current-speed {
    font-size: 13px;
    font-weight: 700;
    color: var(--orange);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    min-width: 40px;
    text-align: center;
  }
  .sep { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }

  .chart-tabs {
    display: flex;
    gap: 4px;
    padding: 0 16px 12px;
  }
  .chart-tab {
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text2);
    transition: all .15s;
  }
  .chart-tab.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .chart-wrapper {
    padding: 0 16px 16px;
    height: 300px;
    position: relative;
  }
  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 400px;
    color: var(--text2);
    font-size: 14px;
    gap: 12px;
  }
  .placeholder .shortcut-hint {
    font-size: 12px;
    color: var(--text2);
    opacity: 0.6;
  }
  .task-tag {
    display: inline-block;
    background: rgba(124, 91, 245, 0.15);
    color: var(--accent2);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin: 2px;
  }
  .ep-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
  }
  .info-item {
    background: var(--surface2);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .info-item .label { font-size: 10px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
  .info-item .value { font-size: 16px; font-weight: 700; margin-top: 3px; }
  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text2);
  }
  .spinner {
    width: 24px; height: 24px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .6s linear infinite;
    margin-right: 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .help-modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 1000;
    align-items: center;
    justify-content: center;
  }
  .help-modal-overlay.show { display: flex; }
  .help-modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    max-width: 480px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
  }
  .help-modal h3 { font-size: 16px; margin-bottom: 16px; }
  .help-modal table { width: 100%; border-collapse: collapse; }
  .help-modal td {
    padding: 5px 0;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
  .help-modal td:first-child { color: var(--text2); width: 50%; }
  .help-modal kbd {
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
  .toast {
    position: fixed;
    top: 60px;
    left: 50%;
    transform: translateX(-50%) translateY(-20px);
    background: var(--surface);
    border: 1px solid var(--accent);
    color: var(--text);
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    z-index: 200;
    opacity: 0;
    transition: all .25s;
    pointer-events: none;
  }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

  @media (max-width: 900px) {
    .videos-grid { grid-template-columns: 1fr; }
    .sidebar { width: 220px; min-width: 220px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>LeRobot Dataset Visualizer</h1>
  <span class="badge" id="robotType">-</span>
  <div class="stats">
    <div>Episodes: <span id="totalEpisodes">-</span></div>
    <div>Frames: <span id="totalFrames">-</span></div>
    <div>FPS: <span id="fps">-</span></div>
  </div>
  <button class="help-btn" onclick="toggleHelp()" title="Keyboard shortcuts (?)">?</button>
</div>

<div id="toast" class="toast"></div>

<div class="help-modal-overlay" id="helpModal" onclick="if(event.target===this)toggleHelp()">
  <div class="help-modal">
    <h3>Keyboard Shortcuts</h3>
    <table>
      <tr><td>Play / Pause</td><td><kbd>Space</kbd></td></tr>
      <tr><td>Previous frame</td><td><kbd>&larr;</kbd></td></tr>
      <tr><td>Next frame</td><td><kbd>&rarr;</kbd></td></tr>
      <tr><td>Back 1 second</td><td><kbd>Shift</kbd> + <kbd>&larr;</kbd></td></tr>
      <tr><td>Forward 1 second</td><td><kbd>Shift</kbd> + <kbd>&rarr;</kbd></td></tr>
      <tr><td>Jump to start</td><td><kbd>Home</kbd></td></tr>
      <tr><td>Jump to end</td><td><kbd>End</kbd></td></tr>
      <tr><td>Speed down</td><td><kbd>[</kbd></td></tr>
      <tr><td>Speed up</td><td><kbd>]</kbd></td></tr>
      <tr><td>Reset speed (1x)</td><td><kbd>\\</kbd></td></tr>
      <tr><td>Previous episode</td><td><kbd>P</kbd></td></tr>
      <tr><td>Next episode</td><td><kbd>N</kbd></td></tr>
      <tr><td>Toggle help</td><td><kbd>?</kbd></td></tr>
    </table>
  </div>
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">
      <h2>Episodes</h2>
      <span class="ep-count" id="epCount">-</span>
    </div>
    <div class="filter-bar">
      <select id="taskFilter">
        <option value="all">All Tasks</option>
      </select>
      <input type="text" id="searchInput" placeholder="Search episode...">
    </div>
    <div class="ep-list" id="epList"></div>
  </div>

  <div class="main-content" id="mainContent">
    <div class="placeholder">
      <div>Select an episode from the sidebar to start</div>
      <div class="shortcut-hint">Press <kbd style="background:var(--surface2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;font-size:12px">?</kbd> for keyboard shortcuts</div>
    </div>
  </div>
</div>

<script>
let appInfo = null;
let episodes = [];
let filteredEpisodes = [];
let currentEp = null;
let actionChart = null;
let currentChartType = 'action';
let playbackSpeed = 1;
let timelineRAF = null;
let isPlaying = false;

const SPEEDS = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 8, 10];
const SPEED_CHIPS = [0.25, 0.5, 1, 2, 4, 8];

const COLORS = [
  '#4f8ff7','#7c5bf5','#34d399','#fb923c','#f87171','#a78bfa','#38bdf8',
  '#fbbf24','#f472b6','#2dd4bf','#818cf8','#c084fc','#e879f9',
];

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 1200);
}

function toggleHelp() {
  document.getElementById('helpModal').classList.toggle('show');
}

async function init() {
  const [infoRes, epsRes] = await Promise.all([
    fetch('/api/info').then(r => r.json()),
    fetch('/api/episodes').then(r => r.json()),
  ]);
  appInfo = infoRes;
  episodes = epsRes;

  document.getElementById('robotType').textContent = appInfo.robot_type;
  document.getElementById('totalEpisodes').textContent = appInfo.total_episodes;
  document.getElementById('totalFrames').textContent = appInfo.total_frames.toLocaleString();
  document.getElementById('fps').textContent = appInfo.fps;

  const taskFilter = document.getElementById('taskFilter');
  appInfo.tasks.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.task_index;
    opt.textContent = t.task;
    taskFilter.appendChild(opt);
  });
  taskFilter.addEventListener('change', renderEpisodeList);
  document.getElementById('searchInput').addEventListener('input', renderEpisodeList);

  renderEpisodeList();
  setupKeyboard();
}

function renderEpisodeList() {
  const filter = document.getElementById('taskFilter').value;
  const search = document.getElementById('searchInput').value.toLowerCase();
  const list = document.getElementById('epList');
  list.innerHTML = '';

  const taskMap = {};
  appInfo.tasks.forEach(t => taskMap[t.task_index] = t.task);

  filteredEpisodes = [];

  episodes.forEach(ep => {
    const taskNames = (ep.tasks || []).map(t => typeof t === 'string' ? t : taskMap[t] || `Task ${t}`);
    const taskStr = taskNames.join(', ');

    if (filter !== 'all') {
      const fidx = parseInt(filter);
      const hasTask = (ep.tasks || []).some(t => {
        if (typeof t === 'string') return appInfo.tasks.find(at => at.task === t && at.task_index === fidx);
        return t === fidx;
      });
      if (!hasTask) return;
    }

    const epStr = `Episode ${ep.episode_index} ${taskStr}`;
    if (search && !epStr.toLowerCase().includes(search)) return;

    filteredEpisodes.push(ep);

    const div = document.createElement('div');
    div.className = 'ep-item' + (currentEp === ep.episode_index ? ' active' : '');
    div.dataset.epIdx = ep.episode_index;
    div.innerHTML = `
      <div class="ep-id">Episode ${ep.episode_index}</div>
      <div class="ep-meta">
        <span>${ep.length} frames</span>
        <span>${(ep.length / appInfo.fps).toFixed(1)}s</span>
      </div>
    `;
    div.addEventListener('click', () => selectEpisode(ep.episode_index));
    list.appendChild(div);
  });

  document.getElementById('epCount').textContent = filteredEpisodes.length;
}

function getAdjacentEpisode(dir) {
  if (currentEp === null || filteredEpisodes.length === 0) return null;
  const idx = filteredEpisodes.findIndex(e => e.episode_index === currentEp);
  const next = idx + dir;
  if (next < 0 || next >= filteredEpisodes.length) return null;
  return filteredEpisodes[next].episode_index;
}

function navigateEpisode(dir) {
  const target = getAdjacentEpisode(dir);
  if (target !== null) {
    selectEpisode(target);
  } else {
    showToast(dir < 0 ? 'Already at first episode' : 'Already at last episode');
  }
}

async function selectEpisode(epIdx) {
  currentEp = epIdx;
  isPlaying = false;
  if (timelineRAF) { cancelAnimationFrame(timelineRAF); timelineRAF = null; }
  renderEpisodeList();

  const activeItem = document.querySelector(`.ep-item[data-ep-idx="${epIdx}"]`);
  if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });

  const main = document.getElementById('mainContent');
  main.innerHTML = `<div class="loading"><div class="spinner"></div>Loading episode ${epIdx}...</div>`;

  const ep = episodes.find(e => e.episode_index === epIdx);
  const taskMap = {};
  appInfo.tasks.forEach(t => taskMap[t.task_index] = t.task);
  const taskNames = (ep.tasks || []).map(t => typeof t === 'string' ? t : taskMap[t] || `Task ${t}`);

  const dataRes = await fetch(`/api/episode/${epIdx}/data`).then(r => r.json());
  const videoKeys = appInfo.video_keys || [];
  const duration = (ep.length / appInfo.fps).toFixed(1);

  const hasPrev = getAdjacentEpisode(-1) !== null;
  const hasNext = getAdjacentEpisode(1) !== null;

  main.innerHTML = `
    <div class="card">
      <div class="card-header">
        Episode ${epIdx}
        <div class="nav-btns">
          <button onclick="navigateEpisode(-1)" ${hasPrev ? '' : 'disabled'} title="Previous episode (P)">&larr; Prev</button>
          <button onclick="navigateEpisode(1)" ${hasNext ? '' : 'disabled'} title="Next episode (N)">Next &rarr;</button>
        </div>
      </div>
      <div class="card-body">
        <div style="margin-bottom:10px">${taskNames.map(t => `<span class="task-tag">${t}</span>`).join('')}</div>
        <div class="ep-info-grid">
          <div class="info-item"><div class="label">Frames</div><div class="value">${ep.length}</div></div>
          <div class="info-item"><div class="label">Duration</div><div class="value">${duration}s</div></div>
          <div class="info-item"><div class="label">FPS</div><div class="value">${appInfo.fps}</div></div>
          <div class="info-item"><div class="label">Action Dim</div><div class="value">${dataRes.action_names.length}</div></div>
          <div class="info-item"><div class="label">State Dim</div><div class="value">${dataRes.state_names.length}</div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">Camera Views</div>
      <div class="card-body">
        <div class="videos-grid" id="videosGrid">
          ${videoKeys.map(vk => `
            <div class="video-cell" id="cell_${vk.replace(/\./g,'_')}">
              <span class="video-label">${vk.split('.').pop()}</span>
              <span class="frame-overlay" id="overlay_${vk.replace(/\./g,'_')}">0:00.000 | F0</span>
              <video id="video_${vk.replace(/\./g,'_')}" preload="auto" loop muted playsinline>
                <source src="/api/episode/${epIdx}/video/${vk}" type="video/mp4">
              </video>
            </div>
          `).join('')}
        </div>

        <div class="transport-bar">
          <div class="timeline-row">
            <span class="time-display" id="timeDisplay">0:00.000 / ${duration}s</span>
            <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="1000" value="0">
            <span class="time-display" id="frameDisplay" style="min-width:60px;text-align:right">F0/${ep.length}</span>
          </div>
          <div class="controls-row">
            <button class="btn icon-btn secondary" onclick="stepFrame(-1)" title="Previous frame (&larr;)">&lsaquo;</button>
            <button class="btn" id="playBtn" onclick="togglePlay()" title="Play/Pause (Space)">&#9654; Play</button>
            <button class="btn icon-btn secondary" onclick="stepFrame(1)" title="Next frame (&rarr;)">&rsaquo;</button>
            <div class="sep"></div>
            <div class="speed-bar">
              <span class="speed-label">Speed:</span>
              <button class="speed-adj" onclick="adjustSpeed(-1)" title="Slower ([)">-</button>
              <span class="current-speed" id="speedDisplay">1x</span>
              <button class="speed-adj" onclick="adjustSpeed(1)" title="Faster (])">+</button>
              <div class="sep"></div>
              ${SPEED_CHIPS.map(s => `<button class="speed-chip${s===1?' active':''}" data-speed="${s}" onclick="setSpeed(${s})">${s}x</button>`).join('')}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">Trajectories</div>
      <div class="chart-tabs" id="chartTabs">
        <div class="chart-tab active" data-type="action" onclick="switchChart('action')">Action</div>
        <div class="chart-tab" data-type="state" onclick="switchChart('state')">Observation State</div>
        <div class="chart-tab" data-type="joint_pos" onclick="switchChart('joint_pos')">Joint Positions</div>
        <div class="chart-tab" data-type="ee_pos" onclick="switchChart('ee_pos')">End Effector</div>
      </div>
      <div class="chart-wrapper">
        <canvas id="trajectoryChart"></canvas>
      </div>
    </div>
  `;

  window._epData = dataRes;
  window._epLength = ep.length;
  currentChartType = 'action';
  playbackSpeed = 1;
  renderChart(dataRes, 'action');
  setupVideoSync(videoKeys);
}

function setupVideoSync(videoKeys) {
  const videos = document.querySelectorAll('#videosGrid video');
  const slider = document.getElementById('timelineSlider');
  if (!slider || videos.length === 0) return;

  videos.forEach(v => { v.playbackRate = playbackSpeed; });

  let seeking = false;

  slider.addEventListener('input', () => {
    seeking = true;
    const pct = slider.value / 1000;
    videos.forEach(v => {
      if (v.duration) v.currentTime = pct * v.duration;
    });
    updateTimeDisplay();
    updateChartCursor();
  });
  slider.addEventListener('change', () => { seeking = false; });

  function tick() {
    if (!seeking && videos[0] && videos[0].duration) {
      const pct = videos[0].currentTime / videos[0].duration;
      slider.value = Math.round(pct * 1000);
      updateTimeDisplay();
      updateChartCursor();
    }
    timelineRAF = requestAnimationFrame(tick);
  }
  timelineRAF = requestAnimationFrame(tick);
}

function updateTimeDisplay() {
  const videos = document.querySelectorAll('#videosGrid video');
  const v = videos[0];
  if (!v || !v.duration) return;

  const cur = v.currentTime;
  const dur = v.duration;
  const fps = appInfo.fps;
  const frame = Math.round(cur * fps);
  const totalFrames = window._epLength || Math.round(dur * fps);

  const fmt = (t) => {
    const m = Math.floor(t / 60);
    const s = (t % 60).toFixed(3);
    return `${m}:${s.padStart(6, '0')}`;
  };

  const td = document.getElementById('timeDisplay');
  const fd = document.getElementById('frameDisplay');
  if (td) td.textContent = `${fmt(cur)} / ${fmt(dur)}`;
  if (fd) fd.textContent = `F${frame}/${totalFrames}`;

  videos.forEach(vv => {
    const id = vv.id.replace('video_', 'overlay_');
    const ov = document.getElementById(id);
    if (ov) ov.textContent = `${fmt(vv.currentTime)} | F${Math.round(vv.currentTime * fps)}`;
  });
}

function updateChartCursor() {
  if (!actionChart) return;
  const videos = document.querySelectorAll('#videosGrid video');
  const v = videos[0];
  if (!v || !v.duration) return;

  const pct = v.currentTime / v.duration;
  const dataLen = actionChart.data.labels.length;
  const idx = Math.min(Math.round(pct * (dataLen - 1)), dataLen - 1);

  actionChart.setActiveElements([]);
  actionChart.tooltip.setActiveElements(
    actionChart.data.datasets.map((_, di) => ({ datasetIndex: di, index: idx })),
    { x: 0, y: 0 }
  );
  actionChart.update('none');
}

function togglePlay() {
  const videos = document.querySelectorAll('#videosGrid video');
  if (videos.length === 0) return;

  isPlaying = !isPlaying;
  const btn = document.getElementById('playBtn');

  if (isPlaying) {
    videos.forEach(v => v.play());
    if (btn) btn.innerHTML = '&#9646;&#9646; Pause';
  } else {
    videos.forEach(v => v.pause());
    if (btn) btn.innerHTML = '&#9654; Play';
  }
}

function stepFrame(dir) {
  const videos = document.querySelectorAll('#videosGrid video');
  const dt = 1 / appInfo.fps;
  videos.forEach(v => {
    v.pause();
    v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + dir * dt));
  });
  isPlaying = false;
  const btn = document.getElementById('playBtn');
  if (btn) btn.innerHTML = '&#9654; Play';
  setTimeout(updateTimeDisplay, 50);
}

function seekRelative(seconds) {
  const videos = document.querySelectorAll('#videosGrid video');
  videos.forEach(v => {
    v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + seconds));
  });
  setTimeout(updateTimeDisplay, 50);
}

function seekTo(position) {
  const videos = document.querySelectorAll('#videosGrid video');
  videos.forEach(v => {
    if (v.duration) v.currentTime = position === 'start' ? 0 : v.duration;
  });
  setTimeout(updateTimeDisplay, 50);
}

function setSpeed(speed) {
  playbackSpeed = speed;
  const videos = document.querySelectorAll('#videosGrid video');
  videos.forEach(v => { v.playbackRate = speed; });
  document.querySelectorAll('.speed-chip').forEach(c => {
    c.classList.toggle('active', parseFloat(c.dataset.speed) === speed);
  });
  const sd = document.getElementById('speedDisplay');
  if (sd) sd.textContent = `${speed}x`;
  showToast(`Speed: ${speed}x`);
}

function adjustSpeed(dir) {
  const curIdx = SPEEDS.indexOf(playbackSpeed);
  let nextIdx;
  if (curIdx === -1) {
    nextIdx = SPEEDS.findIndex(s => s >= playbackSpeed);
    if (nextIdx === -1) nextIdx = SPEEDS.length - 1;
    nextIdx += dir;
  } else {
    nextIdx = curIdx + dir;
  }
  nextIdx = Math.max(0, Math.min(SPEEDS.length - 1, nextIdx));
  setSpeed(SPEEDS[nextIdx]);
}

function setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (e.shiftKey) seekRelative(-1);
        else stepFrame(-1);
        break;
      case 'ArrowRight':
        e.preventDefault();
        if (e.shiftKey) seekRelative(1);
        else stepFrame(1);
        break;
      case 'Home':
        e.preventDefault();
        seekTo('start');
        break;
      case 'End':
        e.preventDefault();
        seekTo('end');
        break;
      case '[':
        e.preventDefault();
        adjustSpeed(-1);
        break;
      case ']':
        e.preventDefault();
        adjustSpeed(1);
        break;
      case '\\':
        e.preventDefault();
        setSpeed(1);
        break;
      case 'n': case 'N':
        e.preventDefault();
        navigateEpisode(1);
        break;
      case 'p': case 'P':
        e.preventDefault();
        navigateEpisode(-1);
        break;
      case '?':
        e.preventDefault();
        toggleHelp();
        break;
    }
  });
}

function switchChart(type) {
  currentChartType = type;
  document.querySelectorAll('.chart-tab').forEach(t => t.classList.toggle('active', t.dataset.type === type));
  renderChart(window._epData, type);
}

function renderChart(data, type) {
  const canvas = document.getElementById('trajectoryChart');
  if (!canvas) return;

  if (actionChart) { actionChart.destroy(); actionChart = null; }

  let datasets = [];
  let labels = data.timestamps.map(t => t.toFixed(2));

  if (type === 'action') {
    data.action_names.forEach((name, i) => {
      datasets.push({
        label: name,
        data: data.actions.map(a => a[i]),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      });
    });
  } else if (type === 'state') {
    data.state_names.forEach((name, i) => {
      datasets.push({
        label: name,
        data: data.states.map(s => s[i]),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      });
    });
  } else if (type === 'joint_pos') {
    const jointNames = data.state_names.slice(0, 7);
    jointNames.forEach((name, i) => {
      datasets.push({
        label: name,
        data: data.states.map(s => s[i]),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      });
    });
  } else if (type === 'ee_pos') {
    const eeNames = data.state_names.slice(7);
    eeNames.forEach((name, i) => {
      datasets.push({
        label: name,
        data: data.states.map(s => s[i + 7]),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      });
    });
  }

  const step = Math.max(1, Math.floor(labels.length / 200));
  const sampledLabels = labels.filter((_, i) => i % step === 0);
  const sampledDatasets = datasets.map(ds => ({
    ...ds,
    data: ds.data.filter((_, i) => i % step === 0),
  }));

  const verticalLinePlugin = {
    id: 'verticalLine',
    afterDraw(chart) {
      const active = chart.tooltip?.getActiveElements();
      if (active && active.length > 0) {
        const x = active[0].element.x;
        const yAxis = chart.scales.y;
        const ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, yAxis.top);
        ctx.lineTo(x, yAxis.bottom);
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(79, 143, 247, 0.5)';
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();
      }
    }
  };

  actionChart = new Chart(canvas, {
    type: 'line',
    data: { labels: sampledLabels, datasets: sampledDatasets },
    plugins: [verticalLinePlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#9ca0ab', usePointStyle: true, pointStyle: 'line', padding: 12, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor: '#2e3140',
          borderWidth: 1,
          callbacks: {
            title: (items) => items.length ? `t = ${items[0].label}s` : '',
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Time (s)', color: '#9ca0ab' },
          ticks: { color: '#9ca0ab', maxTicksLimit: 15, font: { size: 10 } },
          grid: { color: 'rgba(46,49,64,0.5)' },
        },
        y: {
          ticks: { color: '#9ca0ab', font: { size: 10 } },
          grid: { color: 'rgba(46,49,64,0.5)' },
        },
      },
      onClick(evt, elements) {
        if (elements.length > 0) {
          const idx = elements[0].index;
          const totalLabels = actionChart.data.labels.length;
          const pct = idx / (totalLabels - 1);
          const videos = document.querySelectorAll('#videosGrid video');
          videos.forEach(v => { if (v.duration) v.currentTime = pct * v.duration; });
          setTimeout(updateTimeDisplay, 50);
        }
      },
    },
  });
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot Dataset Web Visualizer")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/root/data/piper_dataset/2026-03-16-merge",
        help="Path to the LeRobot dataset directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_dir}")
    load_dataset(args.dataset_dir)
    print(f"Loaded {INFO['total_episodes']} episodes, {INFO['total_frames']} frames")
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
