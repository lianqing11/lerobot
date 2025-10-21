#!/usr/bin/env python
"""
Test script to verify ManiSkill integration with LeRobot.

Usage:
    python test_maniskill_integration.py
"""

import sys
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.configs import ManiSkillEnv
from lerobot.envs.factory import make_env


def test_maniskill_env_config():
    """Test ManiSkill environment configuration."""
    print("=" * 80)
    print("Testing ManiSkill Environment Configuration")
    print("=" * 80)
    
    # Create ManiSkill config for StackCube-v1
    config = ManiSkillEnv(
        task="StackCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        fps=30,
        episode_length=50,
    )
    
    print(f"\n✓ Config created successfully")
    print(f"  - Task: {config.task}")
    print(f"  - Observation mode: {config.obs_mode}")
    print(f"  - Control mode: {config.control_mode}")
    print(f"  - FPS: {config.fps}")
    print(f"  - Episode length: {config.episode_length}")
    print(f"\n✓ Features:")
    for key, value in config.features.items():
        print(f"  - {key}: shape={value.shape}, type={value.type}")
    
    print(f"\n✓ Gym kwargs:")
    for key, value in config.gym_kwargs.items():
        print(f"  - {key}: {value}")
    
    return config


def test_env_creation(config):
    """Test environment creation."""
    print("\n" + "=" * 80)
    print("Testing Environment Creation")
    print("=" * 80)
    
    try:
        # Create environment
        envs = make_env(config, n_envs=1, use_async_envs=False)
        
        print(f"\n✓ Environment created successfully")
        print(f"  - Suite: {list(envs.keys())}")
        print(f"  - Task IDs: {list(envs['maniskill'].keys())}")
        
        # Get the vectorized environment
        vec_env = envs['maniskill'][0]
        print(f"  - Number of parallel envs: {vec_env.num_envs}")
        print(f"  - Observation space: {vec_env.single_observation_space}")
        print(f"  - Action space: {vec_env.single_action_space}")
        
        return vec_env
    
    except Exception as e:
        print(f"\n✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_env_interaction(vec_env):
    """Test basic environment interaction."""
    print("\n" + "=" * 80)
    print("Testing Environment Interaction")
    print("=" * 80)
    
    try:
        # Reset environment
        obs, info = vec_env.reset()
        print(f"\n✓ Environment reset successfully")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Info keys: {list(info.keys())}")
        
        for key, value in obs.items():
            print(f"  - obs['{key}'] shape: {value.shape}")
        
        # Take a random action
        action = vec_env.action_space.sample()
        print(f"\n✓ Sampled random action: shape={action.shape}")
        
        obs, reward, terminated, truncated, info = vec_env.step(action)
        print(f"\n✓ Step executed successfully")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        print(f"  - Success: {info.get('success', [False])[0]}")
        
        # Run a few more steps
        print(f"\n✓ Running 10 steps...")
        for i in range(10):
            action = vec_env.action_space.sample()
            obs, reward, terminated, truncated, info = vec_env.step(action)
            if terminated[0] or truncated[0]:
                print(f"  - Episode ended at step {i+1}")
                break
        
        print(f"\n✓ Environment interaction test passed!")
        
        # Close environment
        vec_env.close()
        print(f"✓ Environment closed")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Environment interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ManiSkill Integration Test Suite")
    print("=" * 80)
    
    # Test 1: Configuration
    config = test_maniskill_env_config()
    
    # Test 2: Environment creation
    vec_env = test_env_creation(config)
    if vec_env is None:
        print("\n" + "=" * 80)
        print("✗ Tests FAILED: Could not create environment")
        print("=" * 80)
        return False
    
    # Test 3: Environment interaction
    success = test_env_interaction(vec_env)
    
    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✓ All tests PASSED!")
        print("\nYou can now use ManiSkill environments in LeRobot:")
        print("\n  # Example: Evaluate a policy on StackCube-v1")
        print("  lerobot-eval \\")
        print("      --policy.path=your_model \\")
        print("      --env.type=maniskill \\")
        print("      --env.task=StackCube-v1 \\")
        print("      --env.obs_mode=state \\")
        print("      --eval.n_episodes=10")
        print("\n  # Example: Train a policy on StackCube-v1")
        print("  lerobot-train \\")
        print("      --dataset.repo_id=your_dataset \\")
        print("      --env.type=maniskill \\")
        print("      --env.task=StackCube-v1 \\")
        print("      --policy.type=act \\")
        print("      --eval_freq=5000")
    else:
        print("✗ Some tests FAILED")
    print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

