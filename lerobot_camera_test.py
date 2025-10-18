#!/usr/bin/env python3
"""
使用 LeRobot OpenCVCamera 类的实时摄像头测试脚本
这个脚本展示了如何使用 LeRobot 项目中的摄像头类
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.utils.errors import DeviceNotConnectedError, ConnectionError
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您在 LeRobot 项目根目录中运行此脚本")
    sys.exit(1)

def test_lerobot_camera(camera_index=0):
    """
    使用 LeRobot OpenCVCamera 类测试摄像头
    
    Args:
        camera_index (int): 摄像头索引
    """
    print(f"正在使用 LeRobot OpenCVCamera 测试摄像头 {camera_index}...")
    
    try:
        # 创建摄像头配置
        config = OpenCVCameraConfig(
            index_or_path=camera_index,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,  # OpenCV 默认使用 BGR
            rotation=Cv2Rotation.NO_ROTATION
        )
        
        # 创建摄像头实例
        camera = OpenCVCamera(config)
        
        # 连接摄像头
        print("正在连接摄像头...")
        camera.connect(warmup=True)
        
        if not camera.is_connected:
            print("错误：摄像头连接失败")
            return False
        
        print(f"摄像头已成功连接！")
        print(f"分辨率: {camera.width}x{camera.height}")
        print(f"帧率: {camera.fps} FPS")
        print("按 'q' 键退出，按 's' 键保存当前帧")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # 同步读取一帧
                frame = camera.read()
                
                frame_count += 1
                
                # 在画面上添加信息
                cv2.putText(frame, f"LeRobot Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 计算实际帧率
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    actual_fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示画面
                cv2.imshow('LeRobot OpenCV Camera Test', frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    filename = f"lerobot_captured_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"已保存帧到: {filename}")
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            # 断开摄像头连接
            camera.disconnect()
            cv2.destroyAllWindows()
            
            # 显示统计信息
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                avg_fps = frame_count / elapsed_time
                print(f"\n统计信息:")
                print(f"总帧数: {frame_count}")
                print(f"运行时间: {elapsed_time:.2f} 秒")
                print(f"平均帧率: {avg_fps:.2f} FPS")
        
        return True
        
    except ConnectionError as e:
        print(f"连接错误: {e}")
        print("请检查摄像头是否正确连接")
        return False
    except DeviceNotConnectedError as e:
        print(f"设备未连接错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

def test_async_camera(camera_index=0):
    """
    测试异步摄像头读取
    
    Args:
        camera_index (int): 摄像头索引
    """
    print(f"正在测试异步摄像头读取...")
    
    try:
        # 创建摄像头配置
        config = OpenCVCameraConfig(
            index_or_path=camera_index,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR
        )
        
        # 创建摄像头实例
        camera = OpenCVCamera(config)
        
        # 连接摄像头
        camera.connect(warmup=True)
        
        print("异步摄像头测试开始...")
        print("按 'q' 键退出")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # 异步读取一帧
                frame = camera.async_read(timeout_ms=1000)
                
                frame_count += 1
                
                # 在画面上添加信息
                cv2.putText(frame, f"Async Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 计算实际帧率
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    actual_fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示画面
                cv2.imshow('LeRobot Async Camera Test', frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            # 断开摄像头连接
            camera.disconnect()
            cv2.destroyAllWindows()
            
            # 显示统计信息
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                avg_fps = frame_count / elapsed_time
                print(f"\n异步测试统计信息:")
                print(f"总帧数: {frame_count}")
                print(f"运行时间: {elapsed_time:.2f} 秒")
                print(f"平均帧率: {avg_fps:.2f} FPS")
        
        return True
        
    except Exception as e:
        print(f"异步测试错误: {e}")
        return False

def find_lerobot_cameras():
    """
    使用 LeRobot 方法查找可用摄像头
    """
    print("使用 LeRobot 方法搜索可用摄像头...")
    
    try:
        cameras = OpenCVCamera.find_cameras()
        
        if not cameras:
            print("未找到可用的摄像头")
            return []
        
        print(f"找到 {len(cameras)} 个可用摄像头:")
        for i, cam in enumerate(cameras):
            profile = cam['default_stream_profile']
            print(f"摄像头 {i}: {cam['name']}")
            print(f"  ID: {cam['id']}")
            print(f"  后端: {cam['backend_api']}")
            print(f"  分辨率: {profile['width']}x{profile['height']}")
            print(f"  帧率: {profile['fps']} FPS")
            print()
        
        return cameras
        
    except Exception as e:
        print(f"搜索摄像头时出错: {e}")
        return []

if __name__ == "__main__":
    print("LeRobot OpenCV 摄像头测试工具")
    print("=" * 50)
    
    # 查找可用摄像头
    cameras = find_lerobot_cameras()
    
    if not cameras:
        print("没有找到可用的摄像头，退出程序")
        sys.exit(1)
    
    # 使用第一个找到的摄像头
    camera_index = cameras[0]['id']
    print(f"使用摄像头 {camera_index} 进行测试...")
    
    # 测试同步读取
    print("\n1. 测试同步摄像头读取")
    print("-" * 30)
    success1 = test_lerobot_camera(camera_index)
    
    if success1:
        print("\n2. 测试异步摄像头读取")
        print("-" * 30)
        success2 = test_async_camera(camera_index)
        
        if success2:
            print("\n所有测试完成！")
        else:
            print("\n异步测试失败")
    else:
        print("\n同步测试失败")
        sys.exit(1)
