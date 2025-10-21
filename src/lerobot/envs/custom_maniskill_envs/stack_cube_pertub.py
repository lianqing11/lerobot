from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
import random

def build_grid(x_min=-0.15, x_max=0.15, y_min=-0.15, y_max=0.15, n=3):
    """Return an (n*n, 2) array of grid coordinates (x, y).
    The order is row-major over y (from min to max) then x.
    """
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return grid

def placement_from_index_distinct_2(idx: int):
    """
    idx: 0..n-1 的编号
    返回: [i, j] 两个格子的索引位置
    """
    total = 9 * 8  # 9个位置选第一个，剩下8个选第二个
    if idx >= total:
        idx = idx % total

    d1 = idx // 8
    d2 = idx % 8

    pool = list(range(9))
    i = pool.pop(d1)
    j = pool.pop(d2)
    return [i, j]

def placement_from_index_distinct_3(idx: int):
    """
    positions: shape (9, 2) 的坐标数组
    idx: 0..503 的编号（共 504 种 = 9*8*7）
    返回：{'A': (x,y), 'B': (x,y), 'C': (x,y)} 映射
    """
    total = 9*8*7
    if idx >= total:
        idx =  idx % total

    # 混合进制解码：第一位基数9，第二位基数8，第三位基数7（每次都排除已选）
    base1, base2, base3 = 9, 8, 7
    d1 = idx // (base2*base3)
    r1 = idx %  (base2*base3)
    d2 = r1 // base3
    d3 = r1 %  base3

    # 可选索引列表，按升序，逐步剔除
    pool = list(range(9))
    i = pool.pop(d1)        # 第1个位置索引
    j = pool.pop(d2)        # 第2个位置索引（在剩下8个里选）
    k = pool.pop(d3)        # 第3个位置索引（在剩下7个里选）

    return [i, j, k]


@register_env("StackCube-v2", max_episode_steps=200)
### In stackCube V2, we make sure that the sensor view is the same as the human view
class StackCubeEnvV2(StackCubeEnv):
    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [CameraConfig("base_camera", pose, 384, 384, 1, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 384, 384, 1, 0.01, 100)

@register_env("StackCube-light", max_episode_steps=200)
class StackCubeEnvLight(StackCubeEnvV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

@register_env("StackCube-pertube", max_episode_steps=200)
class StackCubeEnvPertube(StackCubeEnvV2):
    """
    **Task Description:**o
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, num_extra_cubes=1, *args, **kwargs,
    ):
        # 在调用父类初始化前设置所有需要的属性
        self.extra_cube_colors = [
            [0, 0, 1, 1],  # 蓝色
            [1, 1, 0, 1],  # 黄色
            [0, 0, 0, 1],  # 黑色
            [1, 1, 1, 1],  # 白色
        ]
        self.num_extra_cubes = num_extra_cubes  # 控制额外方块的数量
        super().__init__(*args, **kwargs)


    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        
        # 创建额外的方块
        self.extra_cubes = []
        for i in range(min(self.num_extra_cubes, len(self.extra_cube_colors))):
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=self.extra_cube_colors[i],
                name=f"cube_{i}",
                initial_pose=sapien.Pose(p=[0, i * 0.1, 0.1]),
            )
            self.extra_cubes.append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        cube_order = options.get("cube_order", [0])
        with torch.device(self.device):
            b = len(cube_order)
            self.table_scene.initialize(cube_order)

            # 定义九宫格的位置（3x3网格）
            grid_positions = build_grid(n=3)
            grid_positions = torch.tensor(grid_positions, device=self.device)
            
            # 计算需要多少个位置（主要的两个方块加上额外的方块）
            total_cubes = 2 + len(self.extra_cubes)
            
            # 为每个批次随机选择不重复的位置
            all_positions = []
            for idx in cube_order:
                if total_cubes == 2:
                    positions = placement_from_index_distinct_2(idx)
                else:
                    positions = placement_from_index_distinct_3(idx)
                positions = positions[:total_cubes]  # 只取需要的数量
                if total_cubes > 3:
                    # 如果需要更多位置，从剩余位置中随机选择
                    for idx in range(total_cubes - 3):
                        remaining_indices = list(set(range(9)) - set(positions))
                        additional_positions = np.random.choice(remaining_indices, total_cubes - 3, replace=False).tolist()
                        positions.extend(additional_positions)
                all_positions.append(grid_positions[positions])
            
            all_positions = torch.stack(all_positions)
            
            # 设置所有方块的初始高度
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            
            # 设置主方块A的位置
            xyz[:, :2] = all_positions[:, 0]  # 使用第一个选定的位置
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # 设置主方块B的位置
            xyz[:, :2] = all_positions[:, 1]  # 使用第二个选定的位置
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            # 设置额外方块的位置
            for i, cube in enumerate(self.extra_cubes):
                xyz[:, :2] = all_positions[:, i + 2]  # 使用后续的选定位置
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False,
                )
                cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

def set_actor_color(actor, color):
    """
    Set the base color for all render shapes of a ManiSkill Actor (wrapper over sapiens.Entity).
    `color` can be [r,g,b] or [r,g,b,a], values in [0,1].
    """
    # normalize to RGBA list
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().tolist()
    rgba = list(color) + [1.0] if len(color) == 3 else list(color)

    # Iterate all per-env entities managed by this Actor
    for ent in actor._objs:
        rbc = ent.find_component_by_type(sapien.render.RenderBodyComponent)
        if rbc is None:
            continue

        # SAPIEN versions differ: try both attributes
        shapes = []
        if hasattr(rbc, "shapes"):
            shapes = rbc.shapes
        elif hasattr(rbc, "get_render_shapes"):
            shapes = rbc.get_render_shapes()

        for shp in shapes:
            # Get material (API differs slightly by version)
            mat = None
            if hasattr(shp, "material"):
                mat = shp.material
            elif hasattr(shp, "get_material"):
                mat = shp.get_material()

            if mat is None:
                continue

            # Set color (again, API name differs slightly by version)
            if hasattr(mat, "set_base_color"):
                mat.set_base_color(rgba)
            elif hasattr(mat, "base_color"):
                mat.base_color = rgba

@register_env("StackCube-hard", max_episode_steps=400)
class StackCubeEnvHard(StackCubeEnvV2):
    CUBE_COLORS = {
        "red":          [1.0, 0.0, 0.0],
        "green":        [0.0, 1.0, 0.0],
        "blue":         [0.0, 0.0, 1.0],
        "yellow":       [1.0, 1.0, 0.0],
        "cyan":         [0.0, 1.0, 1.0],
        "magenta":      [1.0, 0.0, 1.0],
        "orange":       [1.0, 0.5, 0.0],
        "purple":       [0.5, 0.0, 1.0],
        "lime":         [0.7, 1.0, 0.0],
        "pink":         [1.0, 0.6, 0.8],
        "brown":        [0.6, 0.3, 0.0],
        "gray":         [0.5, 0.5, 0.5],
        "white":        [1.0, 1.0, 1.0],
        "black":        [0.0, 0.0, 0.0],
        "teal":         [0.0, 0.5, 0.5],
        "navy":         [0.0, 0.0, 0.5],
    }
    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, 
        max_num_extra_cube=3, **kwargs
    ):
        self.num_extra_cubes = max_num_extra_cube
        super().__init__(*args, **kwargs)

    def get_info(self):
        info = super().get_info()
        info['source_color_name'] = getattr(self, 'source_color_name', 'red')
        info['target_color_name'] = getattr(self, 'target_color_name', 'green')
        return info
    def _load_scene(self, options: dict):
        if "source_color_name" in options and "target_color_name" in options:
            source_color_name = options['source_color_name']
            target_color_name = options['target_color_name']
            assert source_color_name != target_color_name, "Source and target colors must be different."
        else:
            source_color_name = random.choice(list(self.CUBE_COLORS.keys()))
            available_colors = list(self.CUBE_COLORS.keys())
            available_colors.remove(source_color_name)
            target_color_name = random.choice(available_colors)
        # target_color_name = options.get("target_color", "green")
        
        self.source_color_name = source_color_name
        self.target_color_name = target_color_name
        assert source_color_name != target_color_name, "Source and target colors must be different."
        if source_color_name not in self.CUBE_COLORS or target_color_name not in self.CUBE_COLORS:
            raise ValueError(f"Invalid cube colors: {source_color_name}, {target_color_name}")
        source_color = self.CUBE_COLORS[source_color_name]
        target_color = self.CUBE_COLORS[target_color_name]
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=source_color + [1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=target_color + [1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        # 创建额外的方块
        self.extra_cubes = []
        available_colors = list(self.CUBE_COLORS.keys())
        available_colors.remove(source_color_name)
        available_colors.remove(target_color_name)
        num_extra_cube = random.randint(0, self.num_extra_cubes)
        for i in range(num_extra_cube):
            color = random.choice(available_colors)
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=self.CUBE_COLORS[color] + [1],
                name=f"extra_cube_{i}",
                initial_pose=sapien.Pose(p=[i * 0.2, 0, 0.1]),
            )
            self.extra_cubes.append(cube)
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            region = [[-0.12, -0.2], [0.12, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.04, 0.04], device=self.device)) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            # 随机放置额外方块
            for cube in self.extra_cubes:
                extra_xy = xy + sampler.sample(radius, 100, verbose=False)
                xyz[:, :2] = extra_xy
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False,
                )
                cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))




@register_env("StackCube-random-extra", max_episode_steps=200)
class StackCubeEnvRandomExtraCube(StackCubeEnvPertube):
    """
    **Task Description:**
    Extended from StackCubeEnvPertube with random colored extra cubes.
    
    **Additional Features:**
    - Probability parameter to control the chance of including extra cubes
    - Random color selection for extra cubes from predefined color palette
    - Same stacking task as the parent class
    """
    
    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, 
        num_extra_cubes=1, extra_cube_prob=1.0, **kwargs
    ):
        # 设置额外方块出现的概率参数 (0-1之间)
        self.extra_cube_prob = max(0.0, min(1.0, extra_cube_prob))  # 确保在0-1范围内
        self.max_extra_cubes = num_extra_cubes  # 保存最大额外方块数量
        
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, 
                         num_extra_cubes=num_extra_cubes, **kwargs)

    
    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # 创建主要的两个方块
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        
        # 根据概率决定是否创建额外的方块
        self.extra_cubes = []
        if np.random.random() < self.extra_cube_prob and self.max_extra_cubes > 0:
            # 随机决定实际创建的额外方块数量 (1 到 max_extra_cubes)
            actual_extra_cubes = np.random.randint(1, self.max_extra_cubes + 1)
            
            for i in range(min(actual_extra_cubes, len(self.extra_cube_colors))):
                # 随机选择颜色
                random_color = self.extra_cube_colors[np.random.randint(0, len(self.extra_cube_colors))]
                cube = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=random_color,
                    name=f"cube_{i}",
                    initial_pose=sapien.Pose(p=[0, i * 0.1, 0.1]),
                )
                self.extra_cubes.append(cube)
        
        # 更新当前episode的额外方块数量，用于位置分配
        self.num_extra_cubes = len(self.extra_cubes)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # 设置额外方块的位置
            for i, cube in enumerate(self.extra_cubes):
                # 为每个额外方块使用相同的采样策略找到不碰撞的位置
                extra_cube_xy = xy + sampler.sample(radius, 100, verbose=False)
                xyz[:, :2] = extra_cube_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))


@register_env("StackCube-nine", max_episode_steps=200)
class StackCubeEnvNine(StackCubeEnvPertube):
    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, 
        num_extra_cubes=0, **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, num_extra_cubes=num_extra_cubes, **kwargs)
