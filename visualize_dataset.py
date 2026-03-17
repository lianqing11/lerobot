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

INDEX_HTML = """<!DOCTYPE html>
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
    padding: 16px 24px;
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
  .layout {
    display: flex;
    height: calc(100vh - 57px);
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
    gap: 20px;
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
  .card-body { padding: 16px; }
  .videos-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .video-container { position: relative; }
  .video-container .video-label {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,0,0,0.7);
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    z-index: 10;
  }
  .video-container video {
    width: 100%;
    border-radius: 8px;
    background: #000;
    display: block;
  }
  .video-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }
  .btn {
    padding: 6px 14px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    font-weight: 500;
    transition: opacity .15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn.secondary {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
  }
  .speed-select {
    padding: 4px 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    outline: none;
  }
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
    align-items: center;
    justify-content: center;
    height: 300px;
    color: var(--text2);
    font-size: 14px;
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
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
  }
  .info-item {
    background: var(--surface2);
    border-radius: 8px;
    padding: 12px;
  }
  .info-item .label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
  .info-item .value { font-size: 18px; font-weight: 700; margin-top: 4px; }
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
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">
      <h2>Episodes</h2>
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
    <div class="placeholder">Select an episode from the sidebar to start visualization</div>
  </div>
</div>

<script>
let appInfo = null;
let episodes = [];
let currentEp = null;
let actionChart = null;
let stateChart = null;
let currentChartType = 'action';

const COLORS = [
  '#4f8ff7','#7c5bf5','#34d399','#fb923c','#f87171','#a78bfa','#38bdf8',
  '#fbbf24','#f472b6','#2dd4bf','#818cf8','#c084fc','#e879f9',
];

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
}

function renderEpisodeList() {
  const filter = document.getElementById('taskFilter').value;
  const search = document.getElementById('searchInput').value.toLowerCase();
  const list = document.getElementById('epList');
  list.innerHTML = '';

  const taskMap = {};
  appInfo.tasks.forEach(t => taskMap[t.task_index] = t.task);

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

    const div = document.createElement('div');
    div.className = 'ep-item' + (currentEp === ep.episode_index ? ' active' : '');
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
}

async function selectEpisode(epIdx) {
  currentEp = epIdx;
  renderEpisodeList();

  const main = document.getElementById('mainContent');
  main.innerHTML = `<div class="loading"><div class="spinner"></div>Loading episode ${epIdx}...</div>`;

  const ep = episodes.find(e => e.episode_index === epIdx);
  const taskMap = {};
  appInfo.tasks.forEach(t => taskMap[t.task_index] = t.task);
  const taskNames = (ep.tasks || []).map(t => typeof t === 'string' ? t : taskMap[t] || `Task ${t}`);

  const dataRes = await fetch(`/api/episode/${epIdx}/data`).then(r => r.json());

  const videoKeys = appInfo.video_keys || [];

  main.innerHTML = `
    <div class="card">
      <div class="card-header">Episode ${epIdx} Overview</div>
      <div class="card-body">
        <div style="margin-bottom:12px">${taskNames.map(t => `<span class="task-tag">${t}</span>`).join('')}</div>
        <div class="ep-info-grid">
          <div class="info-item"><div class="label">Frames</div><div class="value">${ep.length}</div></div>
          <div class="info-item"><div class="label">Duration</div><div class="value">${(ep.length / appInfo.fps).toFixed(1)}s</div></div>
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
            <div class="video-container">
              <span class="video-label">${vk.split('.').pop()}</span>
              <video id="video_${vk.replace(/\\./g,'_')}" controls preload="auto" loop>
                <source src="/api/episode/${epIdx}/video/${vk}" type="video/mp4">
              </video>
            </div>
          `).join('')}
        </div>
        <div class="video-controls" style="margin-top:12px">
          <button class="btn" onclick="syncPlayAll()">Play All</button>
          <button class="btn secondary" onclick="syncPauseAll()">Pause All</button>
          <select class="speed-select" onchange="setPlaybackSpeed(this.value)">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
          </select>
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
  currentChartType = 'action';
  renderChart(dataRes, 'action');
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

  actionChart = new Chart(canvas, {
    type: 'line',
    data: { labels: sampledLabels, datasets: sampledDatasets },
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
        tooltip: { backgroundColor: '#1a1d27', borderColor: '#2e3140', borderWidth: 1 },
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
    },
  });
}

function syncPlayAll() {
  document.querySelectorAll('#videosGrid video').forEach(v => v.play());
}
function syncPauseAll() {
  document.querySelectorAll('#videosGrid video').forEach(v => v.pause());
}
function setPlaybackSpeed(speed) {
  document.querySelectorAll('#videosGrid video').forEach(v => v.playbackRate = parseFloat(speed));
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
