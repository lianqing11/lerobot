#!/usr/bin/env python3
"""
Converts ManiSkill HDF5 trajectory files to LeRobot v3.0 format.

Usage:
    python convert_maniskill_to_lerobot.py input.h5 output_dir --task-name "Pick cube"

For more information: https://github.com/huggingface/lerobot
"""

import json
import logging
import numpy as np
import pandas as pd
import cv2
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import os.path as osp
from multiprocessing import Pool, cpu_count
import functools
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_FPS = 30
DEFAULT_IMAGE_SIZE = "384x384"
DEFAULT_CHUNKS_SIZE = 1000

# Task descriptions for VLA training (smovla, pi0.5, etc.)
TASK_DESCRIPTIONS = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it.",
    "StackCube-v2": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCube-hard": "Pick up a source_color cube and stack it on top of a target_color cube and let go of the cube without it falling."
}


def get_task_description(task_name: str) -> str:
    """Get task description from task name, with fallback to task name if not found."""
    # Try exact match first
    if task_name in TASK_DESCRIPTIONS:
        return TASK_DESCRIPTIONS[task_name]
    
    # Try fuzzy matching (e.g., "StackCube" matches "StackCube-v1")
    for key in TASK_DESCRIPTIONS:
        if task_name in key or key.split('-')[0] in task_name:
            return TASK_DESCRIPTIONS[key]
    
    # Fallback to task name itself
    return task_name


def load_metadata(h5_file: Path) -> Dict[str, Any]:
    json_file = h5_file.with_suffix('.json')
    if json_file.exists():
        try:
            with open(json_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")
    return {}




def process_single_trajectory(args_tuple):
    """
    Worker function to process a single trajectory.
    Args:
        args_tuple: Tuple containing (traj_data, meta_path, rgb_cameras, state_dim)
    Returns:
        Dict containing episode data and rgb statistics
    """
    traj_data, meta_path, rgb_cameras, state_dim = args_tuple
    
    actions = traj_data['actions']
    episode_data = {'actions': actions}
    num_step = actions.shape[0]
    
    # Initialize rgb statistics dict
    rgb_stats = {}
    
    if rgb_cameras and 'obs' in traj_data:
        for camera_name in rgb_cameras:
            rgb = np.stack([np.load(osp.join(meta_path, traj_data['traj_key'], f"{camera_name}_rgb_{i}.npy"))
                for i in range(num_step)], axis=0)
            episode_data[f'rgb_{camera_name}'] = rgb[:len(actions)]
            
            # Calculate statistics from numpy array directly
            normalized_rgb = rgb.astype(np.float32) / 255.0
            pixels = normalized_rgb.reshape(-1, 3)
            
            # Sample pixels if too many to reduce memory usage
            if len(pixels) > 50000:
                indices = np.random.choice(len(pixels), 50000, replace=False)
                pixels = pixels[indices]
            
            rgb_stats[camera_name] = {
                'sum_vals': pixels.sum(axis=0).astype(np.float64),
                'sum_sq_vals': (pixels ** 2).sum(axis=0).astype(np.float64),
                'min_vals': pixels.min(axis=0).astype(np.float32),
                'max_vals': pixels.max(axis=0).astype(np.float32),
                'n_samples': len(pixels),
                'frame_count': len(rgb)
            }
    
    if state_dim and 'obs' in traj_data:
        qpos = traj_data['obs']['agent']['qpos']
        episode_data['robot_state'] = qpos[:len(actions)]
    
    episode_data['rgb_stats'] = rgb_stats
    return episode_data


def load_trajectory_from_h5(h5_file: Path) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
        
    episodes = []
    metadata = load_metadata(h5_file)
    meta_path = str(h5_file)[:-3]
    print(meta_path)
    with h5py.File(h5_file, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        
        if not traj_keys:
            raise ValueError(f"No trajectories found in {h5_file}. Expected keys starting with 'traj_'")
        
        first_traj = f[traj_keys[0]]
        actions = first_traj['actions'][:]
        action_dim = actions.shape[1]
        
        rgb_cameras = ["base_camera"]
        state_dim = None
        if 'obs' in first_traj and 'agent' in first_traj['obs'] and 'qpos' in first_traj['obs']['agent']:
            qpos = first_traj['obs']['agent']['qpos'][:]
            state_dim = qpos.shape[1]
        
        logger.info(f"Detected: action_dim={action_dim}, state_dim={state_dim}, cameras={rgb_cameras}")
        
        # Load all trajectory data first
        logger.info("Loading trajectory data...")
        all_traj_data = []
        for traj_key in tqdm(traj_keys, desc="Loading trajectories"):
            traj = f[traj_key]
            traj_data = {
                'traj_key': traj_key,
                'actions': traj['actions'][:],
                'obs': {}
            }
            
            if 'obs' in traj:
                if 'agent' in traj['obs'] and 'qpos' in traj['obs']['agent']:
                    traj_data['obs']['agent'] = {'qpos': traj['obs']['agent']['qpos'][:]}
            
            all_traj_data.append(traj_data)
        
        # Prepare arguments for multiprocessing with loaded data
        args_list = [(traj_data, meta_path, rgb_cameras, state_dim) for traj_data in all_traj_data]
        
        # Use multiprocessing to process trajectories in parallel
        num_processes = min(cpu_count(), len(traj_keys))
        logger.info(f"Processing {len(traj_keys)} trajectories using {num_processes} processes")
        
        with Pool(processes=num_processes) as pool:
            episodes = list(tqdm(
                pool.imap(process_single_trajectory, args_list),
                total=len(traj_keys),
                desc="Processing trajectories"
            ))
    
    info = {
        'action_dim': action_dim,
        'state_dim': state_dim,
        'rgb_cameras': rgb_cameras,
        'metadata': metadata
    }
    
    return episodes, info


def parse_image_size(size_str: str) -> Tuple[int, int]:
    if 'x' in size_str:
        parts = size_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid image size format: {size_str}. Expected 'WIDTHxHEIGHT' or 'SIZE'")
        width, height = int(parts[0]), int(parts[1])
    else:
        width = height = int(size_str)
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got: {width}x{height}")
    
    return width, height


def create_directory_structure(
    output_dir: str, 
    rgb_cameras: List[str], 
    num_episodes: int, 
    chunks_size: int = DEFAULT_CHUNKS_SIZE
) -> Path:
    base_path = Path(output_dir)
    num_chunks = (num_episodes + chunks_size - 1) // chunks_size
    
    for chunk_idx in range(num_chunks):
        (base_path / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        
        for camera_name in rgb_cameras:
            camera_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}"
            camera_path.mkdir(parents=True, exist_ok=True)

    (base_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    return base_path


def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result


def create_video_from_frames(
    frames: np.ndarray, 
    output_path: Path, 
    fps: int, 
    image_width: int, 
    image_height: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    target_size = (image_width, image_height)
    resized_frames = [resize_image_with_padding(frame, target_size) for frame in frames]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (image_width, image_height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def _create_video_worker(args_tuple):
    """Worker function for parallel video creation."""
    frames, output_path, fps, image_width, image_height = args_tuple
    create_video_from_frames(frames, output_path, fps, image_width, image_height)
    return output_path


def process_episode(
    episode_data: Dict[str, np.ndarray], 
    episode_idx: int, 
    has_state: bool, 
    fps: int, 
    task_index: int = 0, 
    task_name: str = "Unknown task"
) -> pd.DataFrame:
    actions = episode_data['actions']
    episode_length = actions.shape[0]
    timestamps = np.arange(episode_length, dtype=np.float32) / fps
    
    # Get task description for VLA models
    task_description = get_task_description(task_name)
    
    df_data = {
        'action': [row.tolist() for row in actions],
        'timestamp': timestamps,
        'frame_index': np.arange(episode_length, dtype=np.int64),
        'episode_index': np.full(episode_length, episode_idx, dtype=np.int64),
        'index': np.arange(episode_length, dtype=np.int64),
        'task_index': np.full(episode_length, task_index, dtype=np.int64),
        'task': [task_name] * episode_length,
        'language_instruction': [task_description] * episode_length,
        'language_embedding': [task_description] * episode_length  # For compatibility with some VLA models
    }
    
    if has_state and 'robot_state' in episode_data:
        df_data['observation.state'] = [row.tolist() for row in episode_data['robot_state']]
    
    column_order = ['action', 'observation.state', 'language_instruction', 'language_embedding',
                    'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task']
    
    df = pd.DataFrame(df_data)
    
    # Ensure task and language fields are stored as string
    for col in ['task', 'language_instruction', 'language_embedding']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df[[col for col in column_order if col in df.columns]]


def _merge_rgb_stats_for_camera(camera_name: str, all_episode_stats: List[Dict]) -> Dict[str, Any]:
    """Merge RGB statistics from all episodes for a single camera."""
    # Initialize accumulators
    total_sum_vals = np.zeros(3, dtype=np.float64)
    total_sum_sq_vals = np.zeros(3, dtype=np.float64)
    global_min_vals = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    global_max_vals = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    total_n_samples = 0
    total_frames = 0
    
    # Merge statistics from all episodes
    for episode_stats in all_episode_stats:
        if camera_name in episode_stats:
            stats = episode_stats[camera_name]
            total_sum_vals += stats['sum_vals']
            total_sum_sq_vals += stats['sum_sq_vals']
            global_min_vals = np.minimum(global_min_vals, stats['min_vals'])
            global_max_vals = np.maximum(global_max_vals, stats['max_vals'])
            total_n_samples += stats['n_samples']
            total_frames += stats['frame_count']
    
    # Calculate final statistics
    mean = total_sum_vals / total_n_samples if total_n_samples > 0 else np.zeros(3)
    variance = (total_sum_sq_vals / total_n_samples) - (mean ** 2) if total_n_samples > 0 else np.zeros(3)
    std = np.sqrt(np.maximum(variance, 0))  # Avoid negative variance due to numerical errors
    
    return {
        'mean': [[[float(mean[i])]] for i in range(3)],
        'std': [[[float(std[i])]] for i in range(3)],
        'max': [[[float(global_max_vals[i])]] for i in range(3)],
        'min': [[[float(global_min_vals[i])]] for i in range(3)],
        'count': [[[total_frames]]] * 3
    }


def calculate_statistics(
    all_dataframes: List[pd.DataFrame], 
    rgb_cameras: List[str],
    all_rgb_stats: List[Dict],
    has_state: bool
) -> Dict[str, Any]:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    stats = {}
    
    # Calculate action statistics
    actions = np.stack(combined_df['action'].values)
    stats['action'] = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'count': [len(actions)]
    }
    
    # Calculate state statistics
    if has_state and 'observation.state' in combined_df:
        states = np.stack(combined_df['observation.state'].values)
        stats['observation.state'] = {
            'mean': states.mean(axis=0).tolist(),
            'std': states.std(axis=0).tolist(),
            'max': states.max(axis=0).tolist(),
            'min': states.min(axis=0).tolist(),
            'count': [len(states)]
        }
    
    # Calculate RGB statistics from pre-calculated episode statistics
    logger.info(f"Merging RGB statistics for {len(rgb_cameras)} camera(s)")
    for camera_name in rgb_cameras:
        camera_stats = _merge_rgb_stats_for_camera(camera_name, all_rgb_stats)
        stats[f'observation.images.{camera_name}'] = camera_stats
    
    # Calculate metadata field statistics
    for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        values = combined_df[field].values
        stats[field] = {
            'mean': [float(values.mean())],
            'std': [float(values.std())],
            'max': [int(values.max())] if field != 'timestamp' else [float(values.max())],
            'min': [int(values.min())] if field != 'timestamp' else [float(values.min())],
            'count': [len(values)]
        }
    
    return stats


def create_meta_files(
    base_path: Path, 
    episode_lengths: List[int], 
    total_frames: int,
    action_dim: int, 
    state_dim: Optional[int], 
    rgb_cameras: List[str], 
    metadata: Dict[str, Any], 
    task_name: str, 
    chunks_size: int, 
    fps: int, 
    image_width: int, 
    image_height: int, 
    all_dataframes: List[pd.DataFrame],
    robot_type_override: Optional[str] = None
) -> None:
    num_chunks = (len(episode_lengths) + chunks_size - 1) // chunks_size
    
    episodes_data = []
    dataset_from_index = 0
    
    for ep_idx, (length, df) in enumerate(zip(episode_lengths, all_dataframes)):
        chunk_idx = ep_idx // chunks_size
        
        episode_meta = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_from_index + length,
            "tasks": [task_name],
            "length": length,
        }
        
        for camera_name in rgb_cameras:
            prefix = f"videos/observation.images.{camera_name}"
            episode_meta[f"{prefix}/chunk_index"] = chunk_idx
            episode_meta[f"{prefix}/file_index"] = ep_idx
            episode_meta[f"{prefix}/from_timestamp"] = float(df['timestamp'].iloc[0])
            episode_meta[f"{prefix}/to_timestamp"] = float(df['timestamp'].iloc[-1])
        
        actions = np.stack(df['action'].values)
        episode_meta["stats/action/min"] = actions.min(axis=0).tolist()
        episode_meta["stats/action/max"] = actions.max(axis=0).tolist()
        episode_meta["stats/action/mean"] = actions.mean(axis=0).tolist()
        episode_meta["stats/action/std"] = actions.std(axis=0).tolist()
        episode_meta["stats/action/count"] = [length]
        
        if state_dim and 'observation.state' in df:
            states = np.stack(df['observation.state'].values)
            episode_meta["stats/observation.state/min"] = states.min(axis=0).tolist()
            episode_meta["stats/observation.state/max"] = states.max(axis=0).tolist()
            episode_meta["stats/observation.state/mean"] = states.mean(axis=0).tolist()
            episode_meta["stats/observation.state/std"] = states.std(axis=0).tolist()
            episode_meta["stats/observation.state/count"] = [length]
        
        # Note: Per-episode RGB stats are skipped to save memory
        # Dataset-level RGB stats are computed from video files instead
        
        for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            values = df[field].values
            episode_meta[f"stats/{field}/min"] = [int(values.min())] if field != 'timestamp' else [float(values.min())]
            episode_meta[f"stats/{field}/max"] = [int(values.max())] if field != 'timestamp' else [float(values.max())]
            episode_meta[f"stats/{field}/mean"] = [float(values.mean())]
            episode_meta[f"stats/{field}/std"] = [float(values.std())]
            episode_meta[f"stats/{field}/count"] = [length]
        
        episode_meta["meta/episodes/chunk_index"] = 0
        episode_meta["meta/episodes/file_index"] = 0
        
        episodes_data.append(episode_meta)
        dataset_from_index += length
    
    episodes_df = pd.DataFrame(episodes_data)
    episodes_df.to_parquet(base_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_name])
    tasks_df.index.name = None
    tasks_df.to_parquet(base_path / "meta" / "tasks.parquet", index=True)
    
    # Determine robot type: use override if provided, otherwise auto-detect
    robot_type = "unknown"
    if robot_type_override:
        robot_type = robot_type_override
    elif metadata and 'env_info' in metadata:
        env_id = metadata['env_info'].get('env_id', 'unknown')
        robot_type = env_id.split('-')[0].lower() if '-' in env_id else 'unknown'
    
    features = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
            "fps": float(fps)
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)],
            "fps": float(fps)
        } if state_dim else {},
        "language_instruction": {"dtype": "string", "shape": [1], "names": None, "fps": float(fps)},
        "language_embedding": {"dtype": "string", "shape": [1], "names": None, "fps": float(fps)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(fps)}
    }
    
    # Remove empty observation.state if not present
    if not state_dim:
        del features["observation.state"]
    
    for camera_name in rgb_cameras:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": image_height,
                "video.width": image_width,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    data_files_size = sum(f.stat().st_size for f in (base_path / "data").rglob("*.parquet"))
    data_files_size_mb = int(data_files_size / (1024 * 1024))
    
    info_data = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_lengths) * len(rgb_cameras),
        "total_chunks": num_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "data_files_size_in_mb": data_files_size_mb,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features
    }
    
    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Convert ManiSkill HDF5 dataset to LeRobot v3.0 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_maniskill_to_lerobot.py trajectory.h5 ./output --task-name "Pick cube"
  
  # With all custom settings
  python convert_maniskill_to_lerobot.py trajectory.h5 ./output \\
    --task-name "Pick cube" \\
    --robot-type "panda" \\
    --fps 60 \\
    --image-size 1280x720 \\
    --chunks-size 500

For more information: https://github.com/huggingface/lerobot
        """
    )
    
    parser.add_argument('input_file', type=str,
                       help='Path to ManiSkill .h5 trajectory file')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for LeRobot dataset')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, metavar='N',
                       help=f'Video FPS (default: {DEFAULT_FPS})')
    parser.add_argument('--task-name', type=str, metavar='NAME',
                       help='Task description (default: auto-detected from metadata)')
    parser.add_argument('--chunks-size', type=int, default=DEFAULT_CHUNKS_SIZE, metavar='N',
                       help=f'Episodes per chunk (default: {DEFAULT_CHUNKS_SIZE})')
    parser.add_argument('--image-size', type=str, default=DEFAULT_IMAGE_SIZE, metavar='WxH',
                       help=f'Output image size as WIDTHxHEIGHT or single value for square (default: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--robot-type', type=str, metavar='NAME',
                   help='Robot type (default: auto-detected, e.g., "panda", "ur5")')
    
    args = parser.parse_args()
    
    if args.chunks_size <= 0:
        parser.error("--chunks-size must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    input_path = Path(args.input_file)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    
    try:
        logger.info(f"Loading trajectories from {input_path}")
        episodes, info = load_trajectory_from_h5(input_path)
        logger.info(f"Found {len(episodes)} episodes")
        
        task_name = args.task_name
        if not task_name and info['metadata'] and 'env_info' in info['metadata']:
            task_name = info['metadata']['env_info'].get('env_id', 'Unknown task')
        if not task_name:
            task_name = "Unknown task"
            logger.warning("No task name provided and couldn't auto-detect. Using 'Unknown task'")
        
        base_path = create_directory_structure(args.output_dir, info['rgb_cameras'], len(episodes), args.chunks_size)
        image_width, image_height = parse_image_size(args.image_size)
        
        all_dataframes = []
        all_rgb_stats = []  # Collect RGB statistics from episodes
        episode_lengths = []
        global_index = 0
        
        # Collect video creation tasks
        video_tasks = []
        
        for episode_idx, episode_data in enumerate(tqdm(episodes, desc="Processing episodes")):
            chunk_idx = episode_idx // args.chunks_size
            
            df = process_episode(episode_data, episode_idx, info['state_dim'] is not None, args.fps,
                               task_index=0, task_name=task_name)
            episode_length = len(df)
            df['index'] = range(global_index, global_index + episode_length)
            global_index += episode_length
            
            # Collect RGB statistics from this episode
            if 'rgb_stats' in episode_data:
                all_rgb_stats.append(episode_data['rgb_stats'])
                    
            # Collect video creation tasks instead of creating them immediately
            for camera_name in info['rgb_cameras']:
                rgb_key = f'rgb_{camera_name}'
                if rgb_key in episode_data:
                    video_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}" / f"file-{episode_idx:03d}.mp4"
                    video_tasks.append((episode_data[rgb_key], video_path, args.fps, image_width, image_height))

            all_dataframes.append(df)
            episode_lengths.append(episode_length)
        
        # Create videos in parallel
        if video_tasks:
            num_video_processes = min(cpu_count(), len(video_tasks), 8)  # Limit to 8 processes for I/O
            logger.info(f"Creating {len(video_tasks)} videos using {num_video_processes} processes")
            
            with Pool(processes=num_video_processes) as pool:
                list(tqdm(
                    pool.imap(_create_video_worker, video_tasks),
                    total=len(video_tasks),
                    desc="Creating videos"
                ))
        
        num_chunks = (len(episodes) + args.chunks_size - 1) // args.chunks_size
        logger.info(f"Saving data to {num_chunks} chunk(s)")
        
        for chunk_idx in range(num_chunks):
            start_ep = chunk_idx * args.chunks_size
            end_ep = min((chunk_idx + 1) * args.chunks_size, len(all_dataframes))
            
            chunk_dfs = all_dataframes[start_ep:end_ep]
            combined_df = pd.concat(chunk_dfs, ignore_index=True)
            
            # Force task and language columns to be string type for parquet
            for col in ['task', 'language_instruction', 'language_embedding']:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].astype('string')

            parquet_path = base_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
            
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            schema_fields = []
            for col in combined_df.columns:
                if col in ['task', 'language_instruction', 'language_embedding']:
                    schema_fields.append(pa.field(col, pa.string()))
                elif col in ['action', 'observation.state']:
                    schema_fields.append(pa.field(col, pa.list_(pa.float32())))
                elif col == 'timestamp':
                    schema_fields.append(pa.field(col, pa.float32()))
                elif col in ['frame_index', 'episode_index', 'index', 'task_index']:
                    schema_fields.append(pa.field(col, pa.int64()))
            
            schema = pa.schema(schema_fields)
            table = pa.Table.from_pandas(combined_df, schema=schema)
            pq.write_table(table, parquet_path)
        
        logger.info("Calculating statistics")
        stats = calculate_statistics(all_dataframes, info['rgb_cameras'], all_rgb_stats, info['state_dim'] is not None)
        with open(base_path / "meta" / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Creating metadata files")
        total_frames = sum(episode_lengths)
        create_meta_files(
            base_path, episode_lengths, total_frames,
            info['action_dim'], info['state_dim'], info['rgb_cameras'],
            info['metadata'], task_name, args.chunks_size, args.fps, 
            image_width, image_height, all_dataframes,
            robot_type_override=args.robot_type
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("Conversion completed successfully!")
        logger.info(f"{'='*80}")
        logger.info(f"Episodes: {len(episode_lengths)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Chunks: {num_chunks}")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())