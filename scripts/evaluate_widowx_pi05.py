#!/usr/bin/env python3
"""
SimplerEnv WidowX evaluation for PI05 (LeRobot).
Loads PI05Policy with its preprocessor/postprocessor pipeline,
outputs absolute EEF poses with rot_offset compensation.
"""
import argparse
import faulthandler
import logging
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

faulthandler.enable()

import numpy as np
import torch
from PIL import Image
from sapien.core import Pose

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("SAPIEN_RENDERER", "offscreen")

import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402
from simpler_env.utils.visualization import write_video  # noqa: E402

logger = logging.getLogger("evaluate_widowx_pi05")


# ---------------------------------------------------------------------------
# Geometry / action helpers
# ---------------------------------------------------------------------------

def get_eef_state_from_obs(obs: Dict) -> np.ndarray:
    """Extract 7-dim EEF state [x,y,z, euler_x,y,z, gripper] from env obs."""
    from scipy.spatial.transform import Rotation as R

    ee_pose_wrt_base = Pose(
        p=obs["agent"]["base_pose"][:3],
        q=obs["agent"]["base_pose"][3:],
    ).inv() * Pose(
        p=obs["extra"]["tcp_pose"][:3],
        q=obs["extra"]["tcp_pose"][3:],
    )
    euler = R.from_quat(ee_pose_wrt_base.q[[1, 2, 3, 0]]).as_euler("xyz")
    gripper = obs["agent"]["qpos"][-1]
    return np.concatenate([ee_pose_wrt_base.p, euler, [gripper]]).astype(np.float32)


def convert_action_for_env(
    target: np.ndarray,
    gripper_close_threshold: float,
) -> np.ndarray:
    """Convert model absolute output to env format (apply rot_offset, binarize gripper)."""
    rot_offset = np.array([0, math.pi / 2, 0], dtype=np.float32)
    euler = target[3:6].astype(np.float32)
    gripper = 1.0 if target[6] < gripper_close_threshold else -1.0
    return np.concatenate([target[:3], euler + rot_offset, [gripper]]).astype(np.float32)


# ---------------------------------------------------------------------------
# PI05 Agent
# ---------------------------------------------------------------------------

class Pi05WidowXAgent:
    def __init__(
        self,
        policy: torch.nn.Module,
        preprocessor: Any,
        postprocessor: Any,
        execute_steps: int = 0,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.execute_steps = execute_steps
        self.action_queue: deque = deque()
        self.current_state: Optional[np.ndarray] = None
        self._gripper_vals: List[float] = []

    def reset(self, instruction: str):
        self.action_queue.clear()
        self.instruction = instruction
        self.current_state = None
        self._gripper_vals = []
        self.policy.reset()

    def _predict_chunk(self, image: np.ndarray) -> np.ndarray:
        """Run PI05 preprocessor -> policy -> postprocessor, return (chunk_size, action_dim) numpy."""
        pil_img = Image.fromarray(image).resize((256, 256), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        obs = {
            "observation.images.top": img_tensor,
            "observation.state": torch.from_numpy(self.current_state).float(),
            "task": self.instruction,
        }
        batch = self.preprocessor(obs)
        action_chunk = self.policy.select_action(batch)
        actions_np = self.postprocessor(action_chunk).numpy()
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        return actions_np

    def step(self, image: np.ndarray, obs: dict, gripper_close_threshold: float) -> np.ndarray:
        self.current_state = get_eef_state_from_obs(obs)

        if not self.action_queue:
            actions = self._predict_chunk(image)
            keep = self.execute_steps if self.execute_steps > 0 else len(actions)
            self.action_queue.extend(actions[:keep])

        target_abs = np.array(self.action_queue.popleft(), dtype=np.float32)[:7]
        self._gripper_vals.append(float(target_abs[6]))
        return convert_action_for_env(target_abs, gripper_close_threshold)

    def gripper_summary(self) -> str:
        if not self._gripper_vals:
            return "no gripper data"
        arr = np.array(self._gripper_vals)
        return (
            f"gripper: mean={arr.mean():.3f} std={arr.std():.3f} "
            f"min={arr.min():.3f} max={arr.max():.3f} n={len(arr)}"
        )


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "cogact": {
        "widowx_spoon_on_towel":          {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_carrot_on_plate":          {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_stack_cube":               {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_put_eggplant_in_basket":   {"max_steps": 120, "gripper_close_threshold": 0.5},
    },
    "official": {
        "widowx_spoon_on_towel":          {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_carrot_on_plate":          {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_stack_cube":               {"max_steps": 120, "gripper_close_threshold": 0.5},
        "widowx_put_eggplant_in_basket":   {"max_steps": 120, "gripper_close_threshold": 0.5},
    },
}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    agent: Pi05WidowXAgent,
    task: str,
    output_dir: Path,
    episodes: int,
    seed_offset: int,
    save_video: bool,
    task_cfg: dict,
    gripper_threshold_override: Optional[float] = None,
) -> float:
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    gripper_th = gripper_threshold_override or task_cfg["gripper_close_threshold"]
    max_steps = task_cfg["max_steps"]
    successes = 0

    for episode in range(episodes):
        env = simpler_env.make(task, renderer_kwargs={"offscreen_only": True})
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode + seed_offset}})
        instruction = env.unwrapped.get_language_instruction()
        agent.reset(instruction)

        frames: List[np.ndarray] = []
        done = False

        for step_idx in range(max_steps):
            image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
            action = agent.step(image, obs, gripper_close_threshold=gripper_th)

            if step_idx == 0:
                logger.info("[%s ep%d] raw action sample: %s", task, episode, action)

            if not np.all(np.isfinite(action)):
                logger.error("Non-finite action at step %d: %s", step_idx, action)
                break

            obs, reward, done, _, _ = env.step(action)
            if save_video:
                frames.append(image.copy())
            if done:
                break

        successes += int(done)
        if save_video:
            video_path = task_dir / f"{task}_ep{episode}_success{int(done)}.mp4"
            write_video(str(video_path), frames, fps=10)
            print(f"Video saved: {video_path.resolve()}")

        logger.info(
            "[%s] ep=%d success=%s steps=%d/%d rate=%.3f | %s",
            task, episode, bool(done), step_idx + 1, max_steps,
            successes / (episode + 1), agent.gripper_summary(),
        )
        env.close()

    rate = successes / float(episodes)
    print(f"\n[{task}] Success rate: {successes}/{episodes} = {rate:.2%}")
    return rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("PI05 WidowX evaluation")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained_model dir")
    p.add_argument("--output_dir", type=str, default="eval_outputs/pi05_bridge_widowx")
    p.add_argument("--tasks", nargs="+",
                   default=["widowx_spoon_on_towel", "widowx_carrot_on_plate",
                            "widowx_stack_cube", "widowx_put_eggplant_in_basket"],
                   choices=["widowx_spoon_on_towel", "widowx_carrot_on_plate",
                            "widowx_stack_cube", "widowx_put_eggplant_in_basket"])
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--eval_setting", type=str, default="cogact", choices=["cogact", "official"])
    p.add_argument("--execute_steps", type=int, default=0,
                   help="Receding horizon: execute N steps per chunk (0=full chunk)")
    p.add_argument("--gripper_threshold", type=float, default=None,
                   help="Override gripper close threshold for all tasks")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    ckpt = Path(args.checkpoint)
    logger.info("Loading PI05 from %s", ckpt)

    policy = PI05Policy.from_pretrained(str(ckpt))
    policy.eval()
    logger.info("Model loaded on %s", policy.config.device)

    preprocessor = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    logger.info("Preprocessor & postprocessor loaded")

    agent = Pi05WidowXAgent(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        execute_steps=args.execute_steps,
    )

    cfgs = TASK_CONFIGS[args.eval_setting]
    out_dir = Path(args.output_dir)
    results: Dict[str, float] = {}

    for task in args.tasks:
        if task not in cfgs:
            logger.warning("Skipping unknown task: %s", task)
            continue
        results[task] = evaluate_task(
            agent=agent,
            task=task,
            output_dir=out_dir,
            episodes=args.episodes,
            seed_offset=args.seed_offset,
            save_video=args.save_video,
            task_cfg=cfgs[task],
            gripper_threshold_override=args.gripper_threshold,
        )

    print("\n" + "=" * 70)
    print("PI05 WidowX Evaluation Results")
    print("=" * 70)
    for t, sr in results.items():
        print(f"  {t:40s}: {sr:6.2%} ({int(sr * args.episodes)}/{args.episodes})")
    print("-" * 70)
    if results:
        avg = sum(results.values()) / len(results)
        total_s = sum(int(sr * args.episodes) for sr in results.values())
        total_e = len(results) * args.episodes
        print(f"  {'Overall':40s}: {avg:6.2%} ({total_s}/{total_e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
