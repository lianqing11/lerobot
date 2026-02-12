"""
Web-based LeRobot dataset visualizer.

Features:
    - Episode list with instruction preview
    - Video playback for each camera view
    - Interactive trajectory plots (action & proprioception)
    - Instruction display

Usage:
    python scripts/visualize_dataset.py \
        --repo_id my_dataset \
        --root /path/to/local/dataset \
        --port 9090

    # Then open http://localhost:9090 in your browser.

Dependencies: flask, numpy, pandas, pyarrow
    pip install flask numpy pandas pyarrow
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, Response, jsonify, request, send_file

app = Flask(__name__)

# Global state filled in main()
DATASET = {}


def load_dataset(root: Path, repo_id: str):
    """Load all dataset metadata and parquet data."""
    # info.json
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)

    # tasks
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")

    # episode metadata
    ep_dir = root / "meta" / "episodes"
    ep_paths = sorted(ep_dir.glob("*/*.parquet"))
    if not ep_paths:
        raise FileNotFoundError(f"No episode metadata in {ep_dir}")
    ep_meta = pa.concat_tables([pq.read_table(p) for p in ep_paths]).to_pandas()

    # data parquet
    data_dir = root / "data"
    data_paths = sorted(data_dir.glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet in {data_dir}")
    df = pd.concat([pd.read_parquet(p) for p in data_paths], ignore_index=True)

    features = info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    image_keys = [k for k, v in features.items() if v["dtype"] == "image"]

    # Identify numeric feature keys (for trajectory plotting)
    numeric_keys = []
    for k, v in features.items():
        if v["dtype"] in ("float32", "float64", "int32", "int64") and k not in (
            "timestamp", "frame_index", "episode_index", "index", "task_index"
        ):
            numeric_keys.append(k)

    return {
        "root": root,
        "repo_id": repo_id,
        "info": info,
        "tasks_df": tasks_df,
        "ep_meta": ep_meta,
        "df": df,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "numeric_keys": numeric_keys,
    }


# ─── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/episodes")
def api_episodes():
    """Return list of episodes with basic info."""
    df = DATASET["df"]
    tasks_df = DATASET["tasks_df"]
    ep_meta = DATASET["ep_meta"]

    episodes = []
    for ep_idx in sorted(df["episode_index"].unique()):
        ep_frames = df[df["episode_index"] == ep_idx]
        task_idx = int(ep_frames["task_index"].iloc[0])
        matching = tasks_df[tasks_df["task_index"] == task_idx]
        instruction = str(matching.index[0]) if len(matching) > 0 else ""
        episodes.append({
            "episode_index": int(ep_idx),
            "num_frames": len(ep_frames),
            "instruction": instruction,
        })

    return jsonify({
        "repo_id": DATASET["repo_id"],
        "total_episodes": len(episodes),
        "fps": DATASET["info"]["fps"],
        "video_keys": DATASET["video_keys"],
        "image_keys": DATASET["image_keys"],
        "numeric_keys": DATASET["numeric_keys"],
        "episodes": episodes,
    })


@app.route("/api/episode/<int:ep_idx>")
def api_episode(ep_idx):
    """Return trajectory data for one episode."""
    df = DATASET["df"]
    ep_frames = df[df["episode_index"] == ep_idx].sort_values("frame_index")

    if len(ep_frames) == 0:
        return jsonify({"error": f"Episode {ep_idx} not found"}), 404

    tasks_df = DATASET["tasks_df"]
    task_idx = int(ep_frames["task_index"].iloc[0])
    matching = tasks_df[tasks_df["task_index"] == task_idx]
    instruction = str(matching.index[0]) if len(matching) > 0 else ""

    # Build trajectory data for each numeric key
    trajectories = {}
    for key in DATASET["numeric_keys"]:
        if key not in ep_frames.columns:
            continue
        values = ep_frames[key].values
        # values could be scalars or arrays (lists)
        first = values[0]
        if isinstance(first, (list, np.ndarray)):
            arr = np.stack(values)  # [T, D]
            trajectories[key] = {
                "shape": list(arr.shape),
                "data": arr.tolist(),
            }
        else:
            arr = np.array(values, dtype=float)
            trajectories[key] = {
                "shape": [len(arr)],
                "data": arr.tolist(),
            }

    timestamps = ep_frames["timestamp"].values.tolist() if "timestamp" in ep_frames.columns else list(range(len(ep_frames)))

    return jsonify({
        "episode_index": int(ep_idx),
        "instruction": instruction,
        "num_frames": len(ep_frames),
        "fps": DATASET["info"]["fps"],
        "timestamps": timestamps,
        "trajectories": trajectories,
        "video_keys": DATASET["video_keys"],
        "image_keys": DATASET["image_keys"],
    })


@app.route("/video/<int:ep_idx>/<path:cam_key>")
def serve_video(ep_idx, cam_key):
    """Serve an MP4 video file for a given episode and camera key."""
    root = DATASET["root"]
    info = DATASET["info"]
    ep_meta = DATASET["ep_meta"]

    if ep_idx >= len(ep_meta):
        return "Episode not found", 404

    ep = ep_meta.iloc[ep_idx]
    chunk_key = f"videos/{cam_key}/chunk_index"
    file_key = f"videos/{cam_key}/file_index"

    chunk_idx = int(ep.get(chunk_key, ep.get("data/chunk_index", 0)))
    file_idx = int(ep.get(file_key, ep.get("data/file_index", 0)))

    video_path_tpl = info.get("video_path", "")
    video_path = root / video_path_tpl.format(
        video_key=cam_key, chunk_index=chunk_idx, file_index=file_idx
    )

    if not video_path.exists():
        return f"Video not found: {video_path}", 404

    # For multi-episode MP4s, we need to trim. But for simplicity serve the whole
    # file and let the frontend seek to the right range.
    from_ts_key = f"videos/{cam_key}/from_timestamp"
    to_ts_key = f"videos/{cam_key}/to_timestamp"
    from_ts = float(ep.get(from_ts_key, 0.0))
    to_ts = float(ep.get(to_ts_key, 0.0))

    # Return the video with time range as headers so frontend can seek
    resp = send_file(str(video_path), mimetype="video/mp4")
    resp.headers["X-From-Timestamp"] = str(from_ts)
    resp.headers["X-To-Timestamp"] = str(to_ts)
    return resp


# ─── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeRobot Dataset Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --text2: #8b8fa3; --accent: #6366f1; --accent2: #818cf8;
    --green: #22c55e; --radius: 8px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); display:flex; height:100vh; overflow:hidden; }

  /* Sidebar */
  #sidebar { width: 340px; min-width: 340px; background: var(--surface); border-right: 1px solid var(--border);
             display:flex; flex-direction:column; }
  #sidebar-header { padding: 16px 20px; border-bottom: 1px solid var(--border); }
  #sidebar-header h2 { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
  #sidebar-header .meta { font-size: 12px; color: var(--text2); }
  #search { width:100%; padding: 8px 12px; margin-top: 10px; background: var(--bg); border: 1px solid var(--border);
            border-radius: var(--radius); color: var(--text); font-size: 13px; outline:none; }
  #search:focus { border-color: var(--accent); }
  #episode-list { flex:1; overflow-y:auto; padding: 6px; }
  .ep-item { padding: 10px 14px; border-radius: var(--radius); cursor:pointer; margin-bottom: 2px;
             transition: background 0.15s; }
  .ep-item:hover { background: rgba(99,102,241,0.1); }
  .ep-item.active { background: rgba(99,102,241,0.2); border-left: 3px solid var(--accent); }
  .ep-item .ep-id { font-size: 12px; font-weight: 600; color: var(--accent2); }
  .ep-item .ep-instr { font-size: 12px; color: var(--text2); margin-top: 2px;
                       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ep-item .ep-frames { font-size: 11px; color: var(--text2); opacity: 0.6; }

  /* Main content */
  #main { flex:1; overflow-y:auto; padding: 24px 32px; }
  #placeholder { display:flex; align-items:center; justify-content:center; height:100%;
                 color: var(--text2); font-size: 15px; }

  /* Instruction banner */
  #instruction-bar { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
                     padding: 14px 20px; margin-bottom: 20px; }
  #instruction-bar .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
                            color: var(--accent2); margin-bottom: 4px; }
  #instruction-bar .text { font-size: 15px; font-weight: 500; }

  /* Video grid */
  #video-section { margin-bottom: 24px; }
  #video-section h3 { font-size: 13px; font-weight: 600; color: var(--text2); margin-bottom: 10px;
                      text-transform: uppercase; letter-spacing: 0.5px; }
  #video-grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .video-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
                overflow: hidden; flex: 1; min-width: 320px; max-width: 640px; }
  .video-card .cam-label { padding: 8px 12px; font-size: 11px; font-weight: 600; color: var(--accent2);
                           border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.5px; }
  .video-card video { width: 100%; display: block; background: #000; }

  /* Trajectory section */
  #trajectory-section h3 { font-size: 13px; font-weight: 600; color: var(--text2); margin-bottom: 10px;
                           text-transform: uppercase; letter-spacing: 0.5px; }
  .traj-plot { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
               margin-bottom: 12px; overflow: hidden; }
  .traj-plot .plot-label { padding: 8px 12px; font-size: 11px; font-weight: 600; color: var(--accent2);
                           border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.5px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h2 id="dataset-title">LeRobot Dataset</h2>
    <div class="meta" id="dataset-meta"></div>
    <input type="text" id="search" placeholder="Search instructions..." oninput="filterEpisodes()">
  </div>
  <div id="episode-list"></div>
</div>

<div id="main">
  <div id="placeholder">Select an episode from the sidebar</div>
  <div id="content" style="display:none;">
    <div id="instruction-bar">
      <div class="label">Instruction</div>
      <div class="text" id="instruction-text"></div>
    </div>
    <div id="video-section">
      <h3>Camera Views</h3>
      <div id="video-grid"></div>
    </div>
    <div id="trajectory-section">
      <h3>Trajectories</h3>
      <div id="trajectory-plots"></div>
    </div>
  </div>
</div>

<script>
let allEpisodes = [];
let datasetInfo = {};
let currentEpIdx = null;

const COLORS = [
  '#6366f1','#22c55e','#f59e0b','#ef4444','#06b6d4','#ec4899',
  '#8b5cf6','#14b8a6','#f97316','#3b82f6','#a855f7','#10b981',
];

async function init() {
  const resp = await fetch('/api/episodes');
  const data = await resp.json();
  datasetInfo = data;
  allEpisodes = data.episodes;

  document.getElementById('dataset-title').textContent = data.repo_id || 'LeRobot Dataset';
  document.getElementById('dataset-meta').textContent =
    `${data.total_episodes} episodes · ${data.fps} fps · ${data.video_keys.length} cameras`;

  renderEpisodeList(allEpisodes);
}

function renderEpisodeList(episodes) {
  const container = document.getElementById('episode-list');
  container.innerHTML = '';
  for (const ep of episodes) {
    const div = document.createElement('div');
    div.className = 'ep-item' + (ep.episode_index === currentEpIdx ? ' active' : '');
    div.innerHTML = `
      <div class="ep-id">Episode ${ep.episode_index}</div>
      <div class="ep-instr">${ep.instruction || '(no instruction)'}</div>
      <div class="ep-frames">${ep.num_frames} frames</div>`;
    div.onclick = () => selectEpisode(ep.episode_index);
    container.appendChild(div);
  }
}

function filterEpisodes() {
  const q = document.getElementById('search').value.toLowerCase();
  const filtered = allEpisodes.filter(ep =>
    ep.instruction.toLowerCase().includes(q) || String(ep.episode_index).includes(q)
  );
  renderEpisodeList(filtered);
}

async function selectEpisode(epIdx) {
  currentEpIdx = epIdx;
  // Highlight in sidebar
  document.querySelectorAll('.ep-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.ep-item').forEach(el => {
    if (el.querySelector('.ep-id').textContent === `Episode ${epIdx}`) el.classList.add('active');
  });

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('content').style.display = 'block';

  const resp = await fetch(`/api/episode/${epIdx}`);
  const data = await resp.json();

  // Instruction
  document.getElementById('instruction-text').textContent = data.instruction || '(no instruction)';

  // Videos
  const videoGrid = document.getElementById('video-grid');
  videoGrid.innerHTML = '';
  for (const vk of data.video_keys) {
    const card = document.createElement('div');
    card.className = 'video-card';
    card.innerHTML = `<div class="cam-label">${vk}</div>`;
    const video = document.createElement('video');
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.style.width = '100%';

    // Fetch video with timestamp headers
    const videoUrl = `/video/${epIdx}/${vk}`;
    video.src = videoUrl;

    // After metadata loaded, seek to correct position for multi-episode mp4
    video.addEventListener('loadedmetadata', async () => {
      try {
        const headResp = await fetch(videoUrl, { method: 'HEAD' });
        const fromTs = parseFloat(headResp.headers.get('X-From-Timestamp') || '0');
        const toTs = parseFloat(headResp.headers.get('X-To-Timestamp') || '0');
        if (fromTs > 0 || toTs > 0) {
          video.currentTime = fromTs;
          // Auto-pause at toTs
          video.addEventListener('timeupdate', function handler() {
            if (video.currentTime >= toTs && toTs > 0) {
              video.currentTime = fromTs;
            }
          });
        }
      } catch(e) {}
      video.play();
    });

    card.appendChild(video);
    videoGrid.appendChild(card);
  }

  // Trajectories
  const trajContainer = document.getElementById('trajectory-plots');
  trajContainer.innerHTML = '';

  const timestamps = data.timestamps;

  for (const [key, traj] of Object.entries(data.trajectories)) {
    const wrapper = document.createElement('div');
    wrapper.className = 'traj-plot';
    wrapper.innerHTML = `<div class="plot-label">${key}</div>`;
    const plotDiv = document.createElement('div');
    plotDiv.style.height = '260px';
    wrapper.appendChild(plotDiv);
    trajContainer.appendChild(wrapper);

    const traces = [];
    if (traj.shape.length === 2) {
      // [T, D] - plot each dimension
      const D = traj.shape[1];
      for (let d = 0; d < D; d++) {
        traces.push({
          x: timestamps,
          y: traj.data.map(row => row[d]),
          name: `dim ${d}`,
          type: 'scatter',
          mode: 'lines',
          line: { color: COLORS[d % COLORS.length], width: 1.5 },
        });
      }
    } else {
      // [T] scalar
      traces.push({
        x: timestamps,
        y: traj.data,
        name: key,
        type: 'scatter',
        mode: 'lines',
        line: { color: COLORS[0], width: 1.5 },
      });
    }

    Plotly.newPlot(plotDiv, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#8b8fa3', size: 10 },
      margin: { l: 50, r: 20, t: 10, b: 36 },
      xaxis: { title: 'time (s)', gridcolor: '#2a2d3a', zerolinecolor: '#2a2d3a' },
      yaxis: { gridcolor: '#2a2d3a', zerolinecolor: '#2a2d3a' },
      legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
      showlegend: traj.shape.length === 2 && traj.shape[1] <= 12,
    }, { responsive: true, displayModeBar: false });
  }
}

init();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Web-based LeRobot dataset visualizer")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo id or local dataset name")
    parser.add_argument("--root", type=str, default=None,
                        help="Local dataset root. Defaults to ~/.cache/huggingface/lerobot/<repo_id>")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
    if not root.exists():
        print(f"Error: Dataset root not found: {root}")
        sys.exit(1)

    global DATASET
    print(f"Loading dataset from {root} ...")
    DATASET = load_dataset(root, args.repo_id)
    print(f"  {DATASET['info']['total_episodes']} episodes, {DATASET['info']['total_frames']} frames")
    print(f"  Video keys: {DATASET['video_keys']}")
    print(f"  Image keys: {DATASET['image_keys']}")
    print(f"  Numeric keys: {DATASET['numeric_keys']}")
    print(f"\nStarting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
