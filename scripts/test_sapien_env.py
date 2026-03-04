#!/usr/bin/env python3
"""Minimal test: only SAPIEN + simpler_env, skip Pi0.5 loading.
Env var setup copied exactly from X-VLA evaluate_widowx.py.
"""
import os

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("SAPIEN_RENDERER", "offscreen")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

print("1. importing simpler_env (includes sapien) ...")
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
print("2. simpler_env imported OK")

print("3. making env ...")
env = simpler_env.make("widowx_spoon_on_towel", renderer_kwargs={"offscreen_only": True})
print("4. env created OK")

obs, _ = env.reset()
base_env = env.unwrapped if hasattr(env, "unwrapped") else env
instruction = base_env.get_language_instruction()
print(f"5. instruction: {instruction}")

image = get_image_from_maniskill2_obs_dict(base_env, obs)
print(f"6. image shape: {image.shape}")

env.close()
print("ALL PASSED")
