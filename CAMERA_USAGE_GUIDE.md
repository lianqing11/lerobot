# OpenCV 实时摄像头使用指南

本指南将帮助您在 macOS 上使用 OpenCV 实时打开摄像头，包括基础方法和 LeRobot 项目的高级用法。

## 目录
1. [准备工作](#准备工作)
2. [基础 OpenCV 摄像头测试](#基础-opencv-摄像头测试)
3. [LeRobot 摄像头测试](#lerobot-摄像头测试)
4. [常见问题解决](#常见问题解决)
5. [高级配置](#高级配置)

## 准备工作

### 1. 检查摄像头权限

在 macOS 上，您需要确保应用程序有摄像头访问权限：

#### 方法 1：通过系统偏好设置
1. 打开 **系统偏好设置** → **安全性与隐私** → **隐私**
2. 选择左侧的 **摄像头**
3. 确保您的终端应用（Terminal.app 或 iTerm2）被勾选
4. 如果使用 Python 解释器，确保 Python 也被勾选

#### 方法 2：通过终端命令
```bash
# 查看当前摄像头权限状态
sudo sqlite3 /Library/Application\ Support/com.apple.TCC/TCC.db "SELECT service, client, auth_value FROM access WHERE service='kTCCServiceCamera';"

# 重置摄像头权限（需要管理员权限）
sudo tccutil reset Camera
```

### 2. 查找可用摄像头

使用 LeRobot 提供的工具查找摄像头：

```bash
# 在项目根目录运行
lerobot-find-cameras opencv
```

或者使用我们提供的测试脚本：

```bash
python3 simple_camera_test.py
```

## 基础 OpenCV 摄像头测试

### 运行基础测试脚本

```bash
python3 simple_camera_test.py
```

这个脚本会：
- 自动搜索可用的摄像头
- 显示实时摄像头画面
- 显示帧率统计信息
- 支持保存当前帧（按 's' 键）
- 按 'q' 键退出

### 基础 OpenCV 代码示例

```python
import cv2

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(0)  # 0 是默认摄像头索引

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 实时显示摄像头画面
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取摄像头画面")
        break
    
    # 显示画面
    cv2.imshow('Camera', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
```

## LeRobot 摄像头测试

### 运行 LeRobot 测试脚本

```bash
python3 lerobot_camera_test.py
```

这个脚本会：
- 使用 LeRobot 的 OpenCVCamera 类
- 测试同步和异步摄像头读取
- 提供更高级的摄像头控制功能
- 支持多种颜色模式和旋转设置

### LeRobot 摄像头代码示例

```python
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# 创建摄像头配置
config = OpenCVCameraConfig(
    index_or_path=0,  # 摄像头索引
    fps=30,           # 帧率
    width=640,        # 宽度
    height=480,       # 高度
    color_mode=ColorMode.BGR,  # 颜色模式
    rotation=Cv2Rotation.NO_ROTATION  # 旋转
)

# 创建摄像头实例
camera = OpenCVCamera(config)

# 连接摄像头
camera.connect()

# 同步读取一帧
frame = camera.read()

# 异步读取一帧
async_frame = camera.async_read()

# 断开连接
camera.disconnect()
```

## 常见问题解决

### 1. 摄像头无法打开

**错误信息**: `Failed to open camera` 或 `无法打开摄像头`

**解决方案**:
- 检查摄像头是否正确连接
- 确认摄像头权限已授予
- 尝试不同的摄像头索引（0, 1, 2...）
- 重启应用程序或系统

### 2. 权限被拒绝

**错误信息**: 系统弹出权限请求对话框

**解决方案**:
- 点击"允许"授予权限
- 如果错过了对话框，在系统偏好设置中手动添加权限
- 重启应用程序

### 3. 画面卡顿或帧率低

**可能原因**:
- 摄像头驱动问题
- 系统资源不足
- 摄像头设置不当

**解决方案**:
- 降低分辨率或帧率
- 关闭其他占用摄像头的应用程序
- 更新摄像头驱动

### 4. 找不到摄像头

**解决方案**:
```bash
# 使用 LeRobot 工具查找摄像头
lerobot-find-cameras opencv

# 或使用系统命令（macOS）
system_profiler SPCameraDataType
```

### 5. 导入错误

**错误信息**: `ImportError: No module named 'lerobot'`

**解决方案**:
- 确保在 LeRobot 项目根目录中运行脚本
- 安装项目依赖：`pip install -e .`
- 检查 Python 路径设置

## 高级配置

### 1. 多摄像头支持

```python
# 同时使用多个摄像头
cameras = []
for i in range(3):  # 假设有3个摄像头
    config = OpenCVCameraConfig(index_or_path=i)
    camera = OpenCVCamera(config)
    camera.connect()
    cameras.append(camera)

# 同时读取多个摄像头画面
frames = []
for camera in cameras:
    frame = camera.read()
    frames.append(frame)
```

### 2. 自定义摄像头设置

```python
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=60,                    # 高帧率
    width=1920,               # 高分辨率
    height=1080,
    color_mode=ColorMode.RGB, # RGB 颜色模式
    rotation=Cv2Rotation.ROTATE_90,  # 旋转90度
    warmup_s=2.0              # 预热时间
)
```

### 3. 异步摄像头读取

```python
# 启动异步读取
camera.connect()

# 异步读取（非阻塞）
try:
    frame = camera.async_read(timeout_ms=1000)  # 1秒超时
    # 处理帧数据
except TimeoutError:
    print("读取超时")
```

### 4. 保存视频流

```python
import cv2

# 设置视频编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
```

## 性能优化建议

1. **合理设置分辨率**: 根据需求选择合适的分辨率，避免过高的分辨率影响性能
2. **控制帧率**: 设置合适的帧率，通常 30fps 已经足够
3. **使用异步读取**: 对于实时应用，使用异步读取可以提高响应性
4. **及时释放资源**: 使用完毕后及时调用 `disconnect()` 和 `release()`
5. **避免多线程冲突**: 在 macOS 上，OpenCV 默认使用单线程模式

## 故障排除检查清单

- [ ] 摄像头硬件连接正常
- [ ] 摄像头权限已授予
- [ ] 没有其他应用程序占用摄像头
- [ ] 摄像头索引正确
- [ ] Python 环境和依赖正确安装
- [ ] 系统资源充足
- [ ] 摄像头驱动正常

## 联系支持

如果遇到问题，可以：
1. 查看 LeRobot 项目文档
2. 检查 GitHub Issues
3. 运行诊断脚本获取详细信息

---

**注意**: 本指南基于 macOS 系统编写，其他操作系统可能需要不同的配置方法。
