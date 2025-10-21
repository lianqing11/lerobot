# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field
from typing import Any

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    observation_height: int = 480
    observation_width: int = 640
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    observation_height: int = 384
    observation_width: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class ImagePreprocessingConfig:
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class RewardClassifierConfig:
    """Configuration for reward classification."""

    pretrained_path: str | None = None
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class InverseKinematicsConfig:
    """Configuration for inverse kinematics processing."""

    urdf_path: str | None = None
    target_frame_name: str | None = None
    end_effector_bounds: dict[str, list[float]] | None = None
    end_effector_step_sizes: dict[str, float] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    display_cameras: bool = False


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    use_gripper: bool = True
    gripper_penalty: float = 0.0


@dataclass
class ResetConfig:
    """Configuration for environment reset behavior."""

    fixed_reset_joint_positions: Any | None = None
    reset_time_s: float = 5.0
    control_time_s: float = 20.0
    terminate_on_success: bool = True


@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_mode: str = "gamepad"
    observation: ObservationConfig | None = None
    image_preprocessing: ImagePreprocessingConfig | None = None
    gripper: GripperConfig | None = None
    reset: ResetConfig | None = None
    inverse_kinematics: InverseKinematicsConfig | None = None
    reward_classifier: RewardClassifierConfig | None = None
    max_gripper_pos: float | None = 100.0


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: HILSerlProcessorConfig = field(default_factory=HILSerlProcessorConfig)

    name: str = "real_robot"

    @property
    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    fps: int = 30
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/agentview_image": f"{OBS_IMAGES}.image",
            "pixels/robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v2"  # add all tasks
    fps: int = 80
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    multitask_eval: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}",
            "pixels/top": f"{OBS_IMAGE}",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }


@EnvConfig.register_subclass("maniskill")
@dataclass
class ManiSkillEnv(EnvConfig):
    task: str = "StackCube-v1"  # Default task
    fps: int = 30
    episode_length: int = 50
    obs_mode: str = "state"  # Can be "state", "rgb", "rgbd", "pointcloud", "state_dict"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"
    sim_backend: str = "auto"  # "cpu" or "gpu"
    robot_uids: str = "panda"  # Robot type
    camera_width: int = 128
    camera_height: int = 128
    num_envs: int = 1  # Number of parallel environments for GPU simulation
    eval_tasks: list[str] | None = None  # Additional tasks to evaluate on (e.g., ["StackCube-pertube"])
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "base_camera": f"{OBS_IMAGES}.base_camera",
        }
    )

    def __post_init__(self):
        # Update action dimension based on control mode
        if "pd_ee_delta_pose" in self.control_mode:
            action_dim = 8  # 7 DoF (position + quaternion) + 1 gripper
        elif "pd_joint_delta_pos" in self.control_mode:
            action_dim = 9  # 8 joints + 1 gripper for Panda
        else:
            action_dim = 8  # Default

        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))

        # Set up observation features based on obs_mode
        if self.obs_mode == "state":
            # State observation (qpos)
            state_dim = 9  # Panda has 9 joints (7 arm + 2 gripper)
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        
        elif self.obs_mode in ["rgb", "rgbd"]:
            # Visual observation
            self.features["base_camera"] = PolicyFeature(
                type=FeatureType.VISUAL, 
                shape=(self.camera_height, self.camera_width, 3)
            )
        
        elif self.obs_mode == "state_dict":
            # State dict observation (more detailed)
            state_dim = 9
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    @property
    def gym_kwargs(self) -> dict:
        kwargs = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "render_mode": self.render_mode,
            "robot_uids": self.robot_uids,
            "sim_backend": self.sim_backend,
        }
        
        # Add camera configuration for visual observations
        # if self.obs_mode in ["rgb", "rgbd", "pointcloud"]:
        #     kwargs["camera_width"] = self.camera_width
        #     kwargs["camera_height"] = self.camera_height
        
        # Add num_envs for GPU simulation
        if self.sim_backend == "gpu":
            kwargs["num_envs"] = self.num_envs
        
        return kwargs
