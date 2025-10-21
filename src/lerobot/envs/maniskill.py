#!/usr/bin/env python

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
"""ManiSkill environment wrapper for LeRobot."""

import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict
from lerobot.envs.custom_maniskill_envs import *


# Import ManiSkill to register all environments
try:
    import mani_skill.envs  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ManiSkill is not installed. Please install it with: "
        "pip install mani_skill"
    ) from e


def create_maniskill_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict,
    env_cls: type,
    eval_tasks: list[str] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Create ManiSkill vectorized environments.
    
    Args:
        task: Main task name (e.g., "StackCube-v2")
        n_envs: Number of parallel environments
        gym_kwargs: Additional kwargs for gym.make
        env_cls: Vector environment class (SyncVectorEnv or AsyncVectorEnv)
        eval_tasks: Additional tasks to evaluate on (e.g., ["StackCube-pertube"])
                   If None, only creates environment for the main task
    
    Returns:
        Dictionary mapping suite name to task_id to vectorized environment
        Format: {suite_name: {task_id: vec_env}}
        Example: {"maniskill": {0: vec_env_v2, 1: vec_env_pertube}}
    """
    
    def _make_one_task(task_name: str, env_id: int):
        """Create a factory function for a single task with deterministic env_id."""
        def _make():
            env = gym.make(task_name, **gym_kwargs)
            env = ManiSkillWrapper(env, env_id=env_id)
            return env
        return _make
    
    suite_name = "maniskill"
    envs_dict = {}
    
    # Create environment for main task
    vec_env_main = env_cls(
        [_make_one_task(task, env_id=i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    envs_dict[task] = vec_env_main
    # Create environments for additional eval tasks
    if eval_tasks:
        for task_id, eval_task in enumerate(eval_tasks, start=1):
            vec_env = env_cls(
                [_make_one_task(eval_task, env_id=i) for i in range(n_envs)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
            envs_dict[eval_task] = vec_env
    return {suite_name: envs_dict}


class ManiSkillWrapper(gym.ObservationWrapper):
    """
    Wrapper to make ManiSkill environments compatible with LeRobot's evaluation pipeline.
    
    This wrapper:
    - Handles observation transformations
    - Adds success tracking to info dict
    - Ensures consistent observation/action space formats
    - Provides deterministic cube_order for StackCube-pertube based on env_id
    """
    
    def __init__(self, env: gym.Env, env_id: int = 0):
        super().__init__(env)
        
        # Store env_id for deterministic initialization
        self.env_id = env_id
        
        # Store observation mode for processing
        self.obs_mode = getattr(env.unwrapped, "obs_mode", "state")
        
        # Build observation space based on obs_mode
        self.observation_space = self._build_observation_space_from_mode()
        
        # Add render_fps to metadata if not present
        if "render_fps" not in self.metadata:
            # ManiSkill default is 30 fps (can be overridden by sim_freq)
            self.metadata["render_fps"] = 30
        
    def observation(self, obs):
        """Transform observation to dict format."""
        return self._process_observation(obs)
    
    def _process_action(self, action: Any) -> Any:
        """
        Process action to match ManiSkill's expected format.
        
        Args:
            action: Action from the policy (numpy array or tensor)
            
        Returns:
            Processed action for ManiSkill
        """
        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # For pd_ee_delta_pose, ManiSkill expects 7D actions (3 pos + 4 quat)
        # LeRobot policies output 8D actions (7D pose + 1D gripper)

        # ManiSkill expects actions with shape (1, action_dim) - add batch dimension
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action.reshape(1, -1)
        
        # Convert to torch tensor if ManiSkill expects it
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        return action
    
    def _to_numpy(self, value: Any) -> np.ndarray:
        """
        Convert various types to numpy arrays.
        
        Args:
            value: Value to convert (can be torch.Tensor, np.ndarray, or other)
            
        Returns:
            Numpy array
        """
        if isinstance(value, torch.Tensor):
            # Move to CPU if on CUDA and convert to numpy
            return value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value
        else:
            # Try to convert to numpy
            return np.asarray(value)
    
    def _build_observation_space_from_mode(self) -> gym.spaces.Dict:
        """
        Build observation space based on observation mode.
        
        Returns:
            A gym.spaces.Dict matching the expected processed observation structure
        """
        original_space = self.env.observation_space
        
        if self.obs_mode == "state":
            # For state mode, wrap the original space
            return gym.spaces.Dict({
                "agent_pos": original_space
            })
        
        elif self.obs_mode == "state_dict":
            # For state_dict mode, extract qpos from agent
            if isinstance(original_space, gym.spaces.Dict) and "agent" in original_space.spaces:
                agent_space = original_space.spaces["agent"]
                if isinstance(agent_space, gym.spaces.Dict) and "qpos" in agent_space.spaces:
                    qpos_space = agent_space.spaces["qpos"]
                else:
                    # Fallback to original agent space
                    qpos_space = agent_space
                
                spaces = {"agent_pos": qpos_space}
                
                # Add extra spaces if they exist
                if "extra" in original_space.spaces:
                    extra_space = original_space.spaces["extra"]
                    if isinstance(extra_space, gym.spaces.Dict):
                        for key, value in extra_space.spaces.items():
                            spaces[f"extra_{key}"] = value
                
                return gym.spaces.Dict(spaces)
            else:
                # Fallback: wrap original space
                return gym.spaces.Dict({"agent_pos": original_space})
        
        elif self.obs_mode in ["rgb", "rgbd"]:
            # For visual modes, extract camera spaces from sensor_data
            spaces = {}
            
            # Add agent_pos space (LeRobot's preprocessing expects this)
            if isinstance(original_space, gym.spaces.Dict) and "agent" in original_space.spaces:
                agent_space = original_space.spaces["agent"]
                if isinstance(agent_space, gym.spaces.Dict) and "qpos" in agent_space.spaces:
                    qpos_space = agent_space.spaces["qpos"]
                    # ManiSkill sometimes includes a batch dimension of 1, remove it
                    if isinstance(qpos_space, gym.spaces.Box) and qpos_space.shape[0] == 1:
                        spaces["agent_pos"] = gym.spaces.Box(
                            low=qpos_space.low.squeeze(0),
                            high=qpos_space.high.squeeze(0),
                            shape=qpos_space.shape[1:],
                            dtype=qpos_space.dtype
                        )
                    else:
                        spaces["agent_pos"] = qpos_space
                else:
                    spaces["agent_pos"] = agent_space
            
            # Extract camera spaces under "pixels" dict
            camera_spaces = {}
            if isinstance(original_space, gym.spaces.Dict) and "sensor_data" in original_space.spaces:
                sensor_data_space = original_space.spaces["sensor_data"]
                if isinstance(sensor_data_space, gym.spaces.Dict):
                    for camera_name, camera_space in sensor_data_space.spaces.items():
                        if isinstance(camera_space, gym.spaces.Dict) and "rgb" in camera_space.spaces:
                            rgb_space = camera_space.spaces["rgb"]
                            # ManiSkill sometimes includes a batch dimension of 1, remove it
                            if isinstance(rgb_space, gym.spaces.Box) and rgb_space.shape[0] == 1:
                                # Remove the first dimension
                                camera_spaces[camera_name] = gym.spaces.Box(
                                    low=rgb_space.low.squeeze(0),
                                    high=rgb_space.high.squeeze(0),
                                    shape=rgb_space.shape[1:],
                                    dtype=rgb_space.dtype
                                )
                            else:
                                camera_spaces[camera_name] = rgb_space
            
            # If we couldn't extract camera spaces, create a default one
            if not camera_spaces:
                # Default camera space (128x128x3 RGB image)
                camera_spaces["base_camera"] = gym.spaces.Box(
                    low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
                )
            
            # Add pixels as a Dict space containing all cameras
            spaces["pixels"] = gym.spaces.Dict(camera_spaces)
            
            # Ensure agent_pos exists
            if "agent_pos" not in spaces:
                # Default agent_pos space (9 joints for Panda)
                spaces["agent_pos"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
                )
            
            return gym.spaces.Dict(spaces)
        
        else:
            # Default: wrap original space
            if isinstance(original_space, gym.spaces.Dict):
                return original_space
            else:
                return gym.spaces.Dict({"observation": original_space})
    
    def reset(self, **kwargs):
        """Reset the environment with deterministic cube_order for pertube tasks."""
        # For StackCube-pertube, add deterministic cube_order based on env_id
        task_name = getattr(self.env.unwrapped, "spec", None)
        if task_name:
            task_id = getattr(task_name, "id", "")
            if "pertube" in task_id.lower():
                # Set deterministic cube_order based on env_id
                # This ensures each environment gets a different but consistent configuration
                if "options" not in kwargs or kwargs["options"] is None:
                    kwargs["options"] = {}
                if "cube_order" not in kwargs.get("options", {}):
                    # Use env_id as the cube_order for deterministic placement
                    kwargs["options"]["cube_order"] = [self.env_id * 25]
        
        obs, info = super().reset(**kwargs)
        
        # Initialize success tracking
        if "success" not in info:
            info["success"] = False
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Process action to match ManiSkill's expected format
        action_processed = self._process_action(action)
        obs, reward, terminated, truncated, info = super().step(action_processed)
        
        # Extract success from info if available
        if "success" not in info:
            # Try to get success from environment
            if hasattr(self.env.unwrapped, "evaluate"):
                eval_info = self.env.unwrapped.evaluate()
                info["success"] = eval_info.get("success", False)
            else:
                info["success"] = False
        
        # Add is_success for compatibility
        info["is_success"] = info["success"]
        
        return obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs: Any) -> Dict[str, Any]:
        """
        Process observations to ensure compatibility with LeRobot.
        
        ManiSkill returns different observation formats based on obs_mode:
        - "state": flat array
        - "state_dict": nested dict with agent/extra keys
        - "rgb", "rgbd": dict with sensor_data and images
        
        This method converts them to LeRobot's expected format.
        """
        if self.obs_mode == "state":
            # Flat state observation
            if isinstance(obs, (np.ndarray, torch.Tensor)):
                return {"agent_pos": self._to_numpy(obs)}
            elif isinstance(obs, dict) and "agent" in obs:
                # Sometimes state mode still returns dict
                return {"agent_pos": self._to_numpy(obs["agent"]["qpos"])}
            else:
                return {"agent_pos": self._to_numpy(obs)}
        
        elif self.obs_mode == "state_dict":
            # State dict observation
            processed_obs = {}
            
            # Extract robot state
            if "agent" in obs:
                processed_obs["agent_pos"] = self._to_numpy(obs["agent"]["qpos"])
            
            # Extract extra information if needed
            if "extra" in obs:
                for key, value in obs["extra"].items():
                    processed_obs[f"extra_{key}"] = self._to_numpy(value)
            
            return processed_obs
        
        elif self.obs_mode in ["rgb", "rgbd"]:
            # Visual observation
            processed_obs = {}
            
            # Extract agent state - LeRobot's preprocess_observation expects this
            if "agent" in obs:
                if isinstance(obs["agent"], dict) and "qpos" in obs["agent"]:
                    agent_qpos = self._to_numpy(obs["agent"]["qpos"])
                    # Remove batch dimension if it's 1 (ManiSkill sometimes adds this)
                    if agent_qpos.ndim == 2 and agent_qpos.shape[0] == 1:
                        agent_qpos = agent_qpos.squeeze(0)
                    processed_obs["agent_pos"] = agent_qpos
                else:
                    agent_data = self._to_numpy(obs["agent"])
                    # Remove batch dimension if it's 1
                    if agent_data.ndim == 2 and agent_data.shape[0] == 1:
                        agent_data = agent_data.squeeze(0)
                    processed_obs["agent_pos"] = agent_data
            if "extra" in obs:
                for key, value in obs["extra"].items():
                    processed_obs[f"extra_{key}"] = self._to_numpy(value)
            
            # Extract images from sensor_data
            # LeRobot's preprocess_observation expects images under "pixels" key
            pixels = {}
            if "sensor_data" in obs:
                for camera_name, camera_data in obs["sensor_data"].items():
                    if "rgb" in camera_data:
                        # ManiSkill returns images in (H, W, C) format, but sometimes with a batch dim
                        # Convert to numpy if it's a tensor
                        rgb_np = self._to_numpy(camera_data["rgb"])
                        # Remove batch dimension if it's 1 (ManiSkill sometimes adds this)
                        if rgb_np.ndim == 4 and rgb_np.shape[0] == 1:
                            rgb_np = rgb_np.squeeze(0)
                        pixels[camera_name] = rgb_np
            
            if pixels:
                processed_obs["pixels"] = pixels
            
            return processed_obs
        
        else:
            # Default: return as-is
            return obs if isinstance(obs, dict) else {"observation": obs}

