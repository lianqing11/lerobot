#!/usr/bin/env python3
"""
简单的 OpenCV 实时摄像头测试脚本
使用基础的 cv2.VideoCapture 来实时显示摄像头画面
"""

import cv2
import sys
import time

def test_camera_basic(camera_index=0):
    """
    使用基础的 OpenCV 方法测试摄像头
    
    Args:
        camera_index (int): 摄像头索引，通常是 0
    """
    print(f"正在尝试打开摄像头 {camera_index}...")
    
    # 创建 VideoCapture 对象
    cap = cv2.VideoCapture(camera_index)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {camera_index}")
        print("请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 摄像头权限是否已授予")
        print("3. 摄像头索引是否正确")
        return False
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 获取实际设置的参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"摄像头已成功打开！")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps} FPS")
    print("按 'q' 键退出，按 's' 键保存当前帧")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                print("错误：无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 在画面上添加信息
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 计算实际帧率
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                actual_fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow('OpenCV Camera Test', frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存帧到: {filename}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 释放资源
        cap.release()
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

def find_available_cameras():
    """
    查找可用的摄像头
    """
    print("正在搜索可用的摄像头...")
    available_cameras = []
    
    for i in range(10):  # 检查前10个摄像头索引
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            
            camera_info = {
                'index': i,
                'width': width,
                'height': height,
                'fps': fps,
                'backend': backend
            }
            available_cameras.append(camera_info)
            print(f"摄像头 {i}: {width}x{height} @ {fps:.1f}fps (后端: {backend})")
        
        cap.release()
    
    if not available_cameras:
        print("未找到可用的摄像头")
    else:
        print(f"找到 {len(available_cameras)} 个可用摄像头")
    
    return available_cameras

if __name__ == "__main__":
    print("OpenCV 摄像头测试工具")
    print("=" * 50)
    
    # 查找可用摄像头
    cameras = find_available_cameras()
    
    if not cameras:
        print("没有找到可用的摄像头，退出程序")
        sys.exit(1)
    
    # 使用第一个找到的摄像头
    camera_index = cameras[0]['index']
    print(f"\n使用摄像头 {camera_index} 进行测试...")
    
    # 测试摄像头
    success = test_camera_basic(camera_index)
    
    if success:
        print("摄像头测试完成！")
    else:
        print("摄像头测试失败！")
        sys.exit(1)
