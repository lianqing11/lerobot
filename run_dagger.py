#!/usr/bin/env python3
"""
DAgger (Dataset Aggregation) data collection for Piper robot.

Each episode is a single continuous recording:
  1. Press ENTER — Policy starts driving the Follower, Leader shadow-follows.
     Recording begins immediately.
  2. Press SPACE — Recording pauses, Leader enters teach mode.
     Human grabs the Leader arm and prepares to demonstrate.
  3. Press SPACE — Recording resumes, human controls the robot via Leader.
  4. Press SPACE — Control returns to Policy, Leader resumes shadowing.
     Recording continues.
  5. Steps 2-4 can repeat as many times as needed within one episode.
  6. Press Right Arrow — Episode ends and is saved.

Both policy-driven frames and human-intervention frames are recorded as one
episode, which is the standard DAgger / HG-DAgger paradigm.
Preparation pauses (step 2) do not count toward episode duration.

Usage:
  python run_dagger.py --config configs/piper_record_policy.yaml
  python run_dagger.py --config configs/piper_record.yaml --policy.path /path/to/ckpt
"""
from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say

logger = logging.getLogger(__name__)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Terminal keyboard listener — works over SSH, no X / pynput needed.
#   SPACE        ->  toggle intervention (Policy <-> Teleop)
#   Right arrow  ->  exit current episode early
#   Left arrow   ->  discard & re-record current episode
#   ESC or 'q'   ->  stop entire recording session
# ---------------------------------------------------------------------------
def _init_keyboard_listener() -> tuple[threading.Thread | None, dict]:
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "waiting_for_enter": False,
        "start_episode": False,
        "dagger_state": "policy",  # "policy" | "preparing" | "human"
        "_quit": False,
    }

    if not sys.stdin.isatty():
        logger.warning("stdin is not a TTY — keyboard controls disabled.")
        return None, events

    def _reader():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not events.get("_quit"):
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch == " ":
                        cur = events.get("dagger_state", "policy")
                        if cur == "policy":
                            events["dagger_state"] = "preparing"
                            print(f"\n[KEY] SPACE -> {YELLOW}PAUSED — Grab the Leader arm, press SPACE to start teleop{RESET}")
                        elif cur == "preparing":
                            events["dagger_state"] = "human"
                            print(f"\n[KEY] SPACE -> {YELLOW}HUMAN CONTROL — Recording resumed{RESET}")
                        elif cur == "human":
                            events["dagger_state"] = "policy"
                            print(f"\n[KEY] SPACE -> {GREEN}POLICY — Model controlling, recording resumed{RESET}")
                    
                    elif ch == "\x1b":
                        # Could be arrow key sequence (ESC [ X) or bare ESC
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            ch2 = os.read(fd, 1).decode("utf-8", errors="ignore")
                            if ch2 == "[" and select.select([sys.stdin], [], [], 0.05)[0]:
                                ch3 = os.read(fd, 1).decode("utf-8", errors="ignore")
                                if ch3 == "C":  # right arrow
                                    logger.info("[KEY] Right arrow -> finish current episode early")
                                    events["exit_early"] = True
                                elif ch3 == "D":  # left arrow
                                    logger.info("[KEY] Left arrow -> discard & re-record episode")
                                    events["rerecord_episode"] = True
                                    events["exit_early"] = True
                        else:
                            # bare ESC
                            logger.info("[KEY] ESC -> stop recording session")
                            events["stop_recording"] = True
                            events["exit_early"] = True
                    elif ch == "q":
                        logger.info("[KEY] 'q' -> stop recording session")
                        events["stop_recording"] = True
                        events["exit_early"] = True
                    elif ch in ("\r", "\n"):
                        if events.get("waiting_for_enter"):
                            logger.info("[KEY] ENTER -> start recording episode")
                            events["start_episode"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t, events


def _wait_for_enter(events: dict, message: str = "Press ENTER to start recording...") -> bool:
    """Block until the user presses Enter. Returns False if stop_recording is set."""
    if not sys.stdin.isatty():
        logger.warning("stdin is not a TTY; skipping ENTER gate and starting episode immediately.")
        return True
    events["waiting_for_enter"] = True
    events["start_episode"] = False
    print(f"\n{message}", flush=True)
    while not events["start_episode"] and not events["stop_recording"]:
        time.sleep(0.05)
    events["waiting_for_enter"] = False
    return not events["stop_recording"]


@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)
    # v0.5.0 video encoding parameters
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None


@dataclass
class PolicyConfig:
    """Minimal config to point at a pretrained policy checkpoint."""
    path: str = ""        # local dir or HF repo id
    device: str = "cuda"
    use_amp: bool = False


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PolicyConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    display_window_steps: int = 30
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self) -> None:
        if self.teleop is None and self.policy is None:
            raise ValueError("Provide at least `teleop:` or `policy:` in the YAML to control the robot.")


def _control_loop(
    *,
    robot,
    events: dict,
    dataset: LeRobotDataset | None,
    dataset_features: dict,
    fps: int,
    duration_s: float,
    single_task: str,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    teleop=None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    dagger_mode: bool = False,
) -> None:
    """Unified control loop with DAgger intervention support.

    When *dagger_mode* is True the loop begins under policy control and
    records every frame.  Pressing SPACE toggles between policy and human
    teleop (intervention).  The Leader arm mode is switched automatically
    so the human can grab the arm and take over seamlessly.
    """
    has_policy = policy is not None and preprocessor is not None and postprocessor is not None
    if has_policy:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    start_t = time.perf_counter()
    pause_total = 0.0
    _pause_start: float | None = None
    prev_state = events.get("dagger_state", "policy")

    while True:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break
        if events["stop_recording"]:
            break

        dagger_state = events.get("dagger_state", "policy")

        # Handle state transitions (hardware mode switching)
        if dagger_mode and teleop is not None and dagger_state != prev_state:
            if dagger_state == "preparing":
                _leader_switch_to_teach(teleop)
                _pause_start = time.perf_counter()
            elif dagger_state == "human":
                if _pause_start is not None:
                    pause_total += time.perf_counter() - _pause_start
                    _pause_start = None
            elif dagger_state == "policy":
                _leader_switch_to_pos(teleop)
                if has_policy:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
            prev_state = dagger_state

        # During preparation pause: hold position, don't record
        if dagger_mode and dagger_state == "preparing":
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / fps - dt_s)
            if time.perf_counter() - start_t - pause_total >= duration_s:
                return
            continue

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        use_teleop = (
            (dagger_mode and dagger_state == "human" and teleop is not None)
            or (not dagger_mode and not has_policy and teleop is not None)
        )

        action_values = None
        if use_teleop:
            raw_action = teleop.get_action()
            action_values = teleop_action_processor((raw_action, obs))
        elif has_policy:
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action_values = make_robot_action(action_tensor, dataset_features)
        elif teleop is not None:
            raw_action = teleop.get_action()
            action_values = teleop_action_processor((raw_action, obs))

        if action_values is None:
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / fps - dt_s)
            if time.perf_counter() - start_t - pause_total >= duration_s:
                return
            continue

        robot_action_to_send = robot_action_processor((action_values, obs))
        robot.send_action(robot_action_to_send)

        # Shadow Leader when policy is active so the human can see the
        # arm trajectory and grab it when they need to intervene
        if dagger_mode and dagger_state == "policy" and teleop is not None:
            try:
                av_dict = {k: (v.item() if hasattr(v, "item") else float(v))
                           for k, v in action_values.items()}
                _send_action_to_leader(
                    teleop, av_dict,
                    use_degrees=getattr(robot.config, "use_degrees", True),
                )
            except Exception as e:
                logger.warning("[Leader] Shadow error: %s", e)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset_features, action_values, prefix=ACTION)
            is_intervention = np.array([1], dtype=np.int8) if (dagger_mode and dagger_state == "human") else np.array([0], dtype=np.int8)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task, "intervention": is_intervention})

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)

        if time.perf_counter() - start_t - pause_total >= duration_s:
            return


def _prompt_instruction(default: str) -> str:
    """Prompt user for a task instruction. Returns default if input is empty."""
    try:
        text = input(f"Enter instruction for this episode [{default}]: ").strip()
    except EOFError:
        text = ""
    return text if text else default


def _read_leader_joint_stream(teleop: Any) -> tuple[float, float] | None:
    try:
        iface = getattr(teleop, "_iface", None)
        if iface is None or not hasattr(iface, "GetArmJointMsgs"):
            return None
        msg = iface.GetArmJointMsgs()
        return float(getattr(msg, "time_stamp", 0.0)), float(getattr(msg, "Hz", 0.0))
    except Exception:
        return None


def _red(msg: str) -> str:
    return f"{RED}{msg}{RESET}"


def _reset_leader_feedback_path(teleop: Any) -> None:
    iface = getattr(teleop, "_iface", None)
    if iface is None:
        return

    # Best-effort: resume + reset motion state + re-enter joint CAN mode.
    try:
        iface.EmergencyStop(0x02)
    except Exception:
        pass
    try:
        iface.MotionCtrl_1(0x02, 0x00, 0x00)
    except Exception:
        pass
    try:
        spd = max(0, min(100, int(getattr(getattr(teleop, "config", None), "ctrl_speed", 10))))
        iface.MotionCtrl_2(0x01, 0x01, spd, 0x00)
    except Exception:
        pass
    try:
        enter_drag = bool(getattr(getattr(teleop, "config", None), "enter_drag_teach", True))
        if enter_drag:
            iface.MotionCtrl_1(0x00, 0x00, 0x01)
    except Exception:
        pass
    time.sleep(0.2)



def _enable_leader_teach_mode(teleop):
    """Enable drag-teach (compliance) mode on the leader arm (full reset path)."""
    _reset_leader_feedback_path(teleop)
    logger.info("[Leader] Enabled Teach Mode (Compliant)")


def _leader_switch_to_teach(teleop):
    """Position-control → drag-teach without power interruption.

    Drag-teach keeps motors energised with gravity compensation; the arm
    holds its current position and does not drop.
    """
    iface = getattr(teleop, "_iface", None)
    if iface is None:
        return
    try:
        iface.MotionCtrl_1(0x00, 0x00, 0x01)
        logger.info("[Leader] → Teach Mode (gravity comp, compliant)")
    except Exception as e:
        logger.warning("[Leader] Failed to enter teach mode: %s", e)


def _leader_switch_to_pos(teleop):
    """Drag-teach → position-control so ``set_joint_positions_deg`` works.

    Calls ``EnablePiper`` + ``EnableArm`` first — required for the arm to
    accept position commands after having been in drag-teach (read-only) mode.
    """
    iface = getattr(teleop, "_iface", None)
    if iface is None:
        return
    try:
        iface.EnablePiper()
        if hasattr(iface, "EnableArm"):
            iface.EnableArm(7, 0x02)
        iface.MotionCtrl_1(0x00, 0x00, 0x00)  # exit drag-teach
        spd = 30
        if hasattr(teleop.config, "ctrl_speed"):
            spd = max(0, min(100, int(teleop.config.ctrl_speed)))
        for _ in range(3):
            iface.MotionCtrl_2(0x01, 0x01, spd, 0x00)
            time.sleep(0.05)
        logger.info("[Leader] → Position Mode (shadow-following)")
    except Exception as e:
        logger.warning("[Leader] Failed to enter position mode: %s", e)


def _send_action_to_leader(teleop, action: dict, use_degrees: bool = True):
    """Send joint positions to Leader for shadow-following (mirrors Piper.send_action)."""
    iface = getattr(teleop, "_iface", None)
    if iface is None:
        return

    offsets = getattr(teleop, "_cal_offsets_deg", None) or [0.0] * 6
    joint_signs = getattr(teleop.config, "joint_signs", None) or [1] * 6

    joints_hw_deg: list[float] = []
    for i in range(1, 7):
        idx = i - 1
        val = action.get(f"joint_{i}.pos")
        if val is None:
            val = action.get(f"joint_{i}")
        if val is None:
            return
        cmd_deg = float(val)
        raw_oriented = cmd_deg + float(offsets[idx])
        deg_hw = raw_oriented * float(joint_signs[idx])
        joints_hw_deg.append(deg_hw)

    g_val = action.get("gripper.pos", action.get("gripper"))
    gripper_mm: float | None = None
    if g_val is not None:
        gripper_mm = float(g_val) if use_degrees else None

    try:
        if hasattr(iface, "set_joint_positions_deg"):
            iface.set_joint_positions_deg(joints_hw_deg, gripper_mm)
        else:
            joints_int = [int(d * 1000) for d in joints_hw_deg]
            iface.JointCtrl(*joints_int)
            if gripper_mm is not None:
                iface.GripperCtrl(int(gripper_mm * 1000), 2000, 0x01, 0x00)
    except Exception as e:
        logger.warning("[Leader] Send Error: %s", e)



def _read_follower_enable_status(robot: Any) -> list[bool] | None:
    try:
        iface = getattr(robot, "_iface", None)
        if iface is None:
            return None
        # PiperSDKInterface -> iface.piper.GetArmEnableStatus()
        piper = getattr(iface, "piper", None)
        if piper is not None and hasattr(piper, "GetArmEnableStatus"):
            values = piper.GetArmEnableStatus()
            if isinstance(values, (list, tuple)):
                return [bool(v) for v in values]
            return None
        # Direct SDK interface fallback
        if hasattr(iface, "GetArmEnableStatus"):
            values = iface.GetArmEnableStatus()
            if isinstance(values, (list, tuple)):
                return [bool(v) for v in values]
            return None
    except Exception:
        return None
    return None


def _log_collect_precheck(robot: Any, teleop: Any | None) -> None:
    logger.info("采集前状态检查开始。")

    if teleop is not None:
        leader_stream = _read_leader_joint_stream(teleop)
        if leader_stream is None:
            logger.info("Leader 状态读取: 当前 teleop 不提供关节反馈检查接口，跳过。")
        else:
            ts, hz = leader_stream
            logger.info("Leader 关节反馈帧: ts=%.3f hz=%.2f", ts, hz)
            if ts <= 0.0 or hz <= 0.0:
                logger.warning(_red("检测到 Leader 无关节反馈帧，正在自动执行重置恢复流程..."))
                _reset_leader_feedback_path(teleop)
                leader_stream = _read_leader_joint_stream(teleop)
                ts2, hz2 = (leader_stream if leader_stream is not None else (0.0, 0.0))
                logger.info("Leader 重置后关节反馈帧: ts=%.3f hz=%.2f", ts2, hz2)
                if ts2 <= 0.0 or hz2 <= 0.0:
                    logger.error(_red("Leader 重置后仍无关节反馈帧。"))
                    logger.error(_red("后续操作建议："))
                    logger.error(_red("1) 在上位机将 Leader 切回“可反馈关节状态”（不要处于仅控制帧模式）。"))
                    logger.error(_red("2) 仅重启 Leader 电源（断电重上电）。"))
                    logger.error(_red("3) 运行 `bash run_diagnose_follower.sh leader` 确认反馈恢复后再采集。"))
                    raise RuntimeError(
                        "Leader 未提供关节反馈帧（GetArmJointMsgs 的 ts/hz 为 0）。"
                        "自动重置失败，请按红字提示人工处理后重试。"
                    )

    follower_enable = _read_follower_enable_status(robot)
    if follower_enable is None:
        logger.info("Follower 电机使能状态读取失败或不可用，跳过。")
    else:
        states = ", ".join(
            f"关节{i + 1}:{'使能' if v else '未使能'}" for i, v in enumerate(follower_enable[:6])
        )
        logger.info("Follower 电机使能: %s", states)
        if len(follower_enable) >= 6 and not all(follower_enable[:6]):
            logger.warning("Follower 存在未使能关节，采集期间可能不跟随。")

    logger.info("采集前状态检查完成。")


def record(cfg: RecordConfig, *, instruction_mode: str = "config") -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Check for Dagger requirements
    if teleop is None and cfg.policy is None:
        raise ValueError("Config must provide teleop or policy.")
    
    if teleop is None and cfg.policy is not None:
        logger.warning(f"{YELLOW}Warning: No teleop configured. Dagger intervention will not be possible (Policy Only).{RESET}")
    
    if cfg.policy is None and teleop is not None:
        logger.warning(f"{YELLOW}Warning: No policy configured. Running in pure Teleop mode.{RESET}")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot.connect()
    if teleop is not None:
        teleop.connect()
    _log_collect_precheck(robot, teleop)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # DAgger intervention label: 0 = policy-driven frame, 1 = human-intervention frame.
    # Declared at dataset creation so it's stored as a regular parquet column.
    dataset_features["intervention"] = {"dtype": "int8", "shape": (1,), "names": None}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
    else:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            features=dataset_features,
            root=cfg.dataset.root,
            robot_type=robot.name,
            use_videos=cfg.dataset.video,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    if hasattr(robot, "cameras") and len(getattr(robot, "cameras", {})) > 0:
        dataset.start_image_writer(
            num_processes=cfg.dataset.num_image_writer_processes,
            num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load policy if configured
    policy_model: PreTrainedPolicy | None = None
    preprocessor = None
    postprocessor = None

    def _log_mem(tag: str) -> None:
        """Log CPU RAM and GPU VRAM usage at a given checkpoint."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_used = vm.used / 1024 ** 3
            ram_total = vm.total / 1024 ** 3
            ram_free = vm.available / 1024 ** 3
            logging.info("[mem/%s] RAM: %.1f / %.1f GB used, %.1f GB free", tag, ram_used, ram_total, ram_free)
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    lines = {k: v for k, v in (l.split(":", 1) for l in f if ":" in l)}
                total = int(lines["MemTotal"].split()[0]) / 1024 ** 2
                free = int(lines["MemAvailable"].split()[0]) / 1024 ** 2
                logging.info("[mem/%s] RAM: %.1f GB total, %.1f GB available", tag, total, free)
            except Exception:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(dev) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(dev) / 1024 ** 3
                total_vram = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
                logging.info("[mem/%s] GPU VRAM: alloc=%.1f GB, reserved=%.1f GB, total=%.1f GB",
                             tag, alloc, reserved, total_vram)
        except Exception:
            pass

    if cfg.policy is not None and cfg.policy.path:
        logging.info("[policy] Loading config from: %s", cfg.policy.path)
        _log_mem("before_policy_load")

        # Force offline loading to avoid network issues (Hugging Face)
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy.path, local_files_only=True)
        logging.info("[policy] Config loaded. policy_type=%s", getattr(policy_cfg, "policy_type", "?"))
        policy_cfg.device = cfg.policy.device
        policy_cfg.use_amp = cfg.policy.use_amp
        policy_cfg.pretrained_path = Path(cfg.policy.path)

        # Large VLA models (e.g. PI05/PaliGemma ~3B params) create their sub-model skeletons
        # in float32 by default, which requires ~17 GB of CPU RAM just for initialisation.
        # Overriding to bfloat16 halves the peak to ~8.8 GB, matching the checkpoint on disk.
        if hasattr(policy_cfg, "dtype") and policy_cfg.dtype == "float32":
            logging.info(
                "[policy] Overriding dtype float32 → bfloat16 to halve CPU RAM during model init "
                "(checkpoint is already stored in bfloat16)."
            )
            policy_cfg.dtype = "bfloat16"
        logging.info("[policy] dtype=%s device=%s use_amp=%s",
                     getattr(policy_cfg, "dtype", "?"), cfg.policy.device, cfg.policy.use_amp)

        logging.info("[policy] Calling make_policy() — this allocates the model skeleton on CPU...")
        _log_mem("before_make_policy")
        policy_model = make_policy(policy_cfg, ds_meta=dataset.meta)
        logging.info("[policy] make_policy() done.")
        _log_mem("after_make_policy")

        logging.info("[policy] Loading pretrained weights from checkpoint...")
        _log_mem("before_pre_post_processors")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=cfg.policy.path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )
        _log_mem("after_pre_post_processors")
        logging.info("[policy] Preprocessor/postprocessor ready.")
        logging.info("Policy loaded from %s (device=%s)", cfg.policy.path, cfg.policy.device)

    # Keyboard listener (terminal-based, works over SSH — no X/pynput needed)
    listener, events = _init_keyboard_listener()
    has_policy = policy_model is not None
    has_teleop = teleop is not None

    logging.info("=" * 60)
    if has_policy and has_teleop:
        logging.info("DAgger Data Collection")
        logging.info("  ENTER 开始 episode → Policy 自动运行并录制")
        logging.info(f"  {GREEN}Policy 模式{RESET}: 模型控制机器人, Leader 跟随, 录制中")
        logging.info(f"  {YELLOW}SPACE (1){RESET}: 暂停录制, Leader 进入示教模式, 人类准备接管")
        logging.info(f"  {YELLOW}SPACE (2){RESET}: 恢复录制, 人类通过 Leader 操控机器人")
        logging.info(f"  {GREEN}SPACE (3){RESET}: 归还控制权, Policy 继续运行, 录制继续")
        logging.info("  准备阶段不计入 episode 时长")
    elif has_teleop:
        logging.info("Pure Teleop Data Collection")
    else:
        logging.info("Policy-Only Data Collection")
    logging.info("  ENTER        ->  start episode")
    logging.info("  SPACE        ->  cycle: policy → prepare → human → policy")
    logging.info("  Right arrow  ->  finish current episode")
    logging.info("  Left arrow   ->  discard & re-record current episode")
    logging.info("  ESC or 'q'   ->  stop entire session")
    logging.info("=" * 60)

    loop_kwargs = dict(
        robot=robot,
        events=events,
        dataset_features=dataset.features,
        fps=cfg.dataset.fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                single_task = (
                    _prompt_instruction(cfg.dataset.single_task)
                    if instruction_mode == "episode"
                    else cfg.dataset.single_task
                )
                ep_label = f"[Episode {dataset.num_episodes + 1}/{cfg.dataset.num_episodes}]"

                if not _wait_for_enter(
                    events,
                    f"{ep_label} Press ENTER to start episode...",
                ):
                    break

                events["dagger_state"] = "policy"

                if has_policy and has_teleop:
                    # DAgger: policy starts, human can intervene with SPACE
                    _leader_switch_to_pos(teleop)
                    log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                    print(
                        f"{ep_label} {GREEN}Recording (Policy running)...{RESET}  "
                        f"SPACE=takeover  Right Arrow=finish"
                    )
                    _control_loop(
                        **loop_kwargs,
                        dataset=dataset,
                        duration_s=float(cfg.dataset.episode_time_s),
                        single_task=single_task,
                        teleop=teleop,
                        policy=policy_model,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dagger_mode=True,
                    )
                elif has_teleop:
                    _leader_switch_to_teach(teleop)
                    log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                    print(
                        f"{ep_label} {YELLOW}Recording (Teleop)...{RESET}  "
                        f"Right Arrow=finish"
                    )
                    _control_loop(
                        **loop_kwargs,
                        dataset=dataset,
                        duration_s=float(cfg.dataset.episode_time_s),
                        single_task=single_task,
                        teleop=teleop,
                    )
                else:
                    log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                    print(
                        f"{ep_label} {GREEN}Recording (Policy)...{RESET}  "
                        f"Right Arrow=finish"
                    )
                    _control_loop(
                        **loop_kwargs,
                        dataset=dataset,
                        duration_s=float(cfg.dataset.episode_time_s),
                        single_task=single_task,
                        policy=policy_model,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )

                # Episode ended — reset DAgger state to "policy" so the next episode
                # always starts in automatic mode regardless of how this one ended.
                # Also put the Leader back into compliance (teach) mode so it's safe
                # to handle during the environment-reset period before the next ENTER.
                events["dagger_state"] = "policy"
                if teleop is not None and has_policy:
                    _leader_switch_to_teach(teleop)
                    logger.info("[episode] Leader switched to teach mode for environment reset.")

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        # Signal the keyboard reader thread to quit
        events["_quit"] = True
        try:
            if teleop is not None:
                teleop.disconnect()
        finally:
            robot.disconnect()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Dagger collection for Piper from a YAML config."
    )
    ap.add_argument("--config", required=True, help="Path to a RecordConfig YAML (see configs/)")
    ap.add_argument(
        "--prefix",
        type=str,
        help="Dataset prefix (saves to ./datasets/{prefix}). If not provided, uses dataset.root from config.",
    )
    ap.add_argument(
        "--instruction-mode",
        choices=["config", "episode", "select"],
        help="instruction mode",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to tasks.yaml vocabulary file",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and print the config (do not connect to devices).",
    )
    args, draccus_overrides = ap.parse_known_args(argv)

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.exists():
        ap.error(f"config not found: {cfg_path}")

    register_third_party_plugins()
    cfg = draccus.parse(config_class=RecordConfig, config_path=cfg_path, args=draccus_overrides)

    if args.prefix:
        cfg.dataset.root = Path("/home/qing/projects/clawvla/datasets") / args.prefix
        logging.info(f"Using dataset root: {cfg.dataset.root}")

    if args.dry_run:
        print(cfg)
        return 0

    record(cfg, instruction_mode=args.instruction_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())