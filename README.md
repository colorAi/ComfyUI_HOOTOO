HOOTOO_ImageTransform - 3D图像变换工具

HOOTOO_ImageTransform 是一个功能强大的3D图像变换工具节点，灵感来源于PT娃导。它能够对静态图像或默认生成的棋盘格进行复杂的动画变换，支持2D/3D变换、自定义背景和Mask输出。

✨ 核心特性
🎨 图像变换功能
2D/3D变换：支持平移、缩放和3D旋转(X/Y/Z轴)

透视效果：可调节焦距控制透视强度

动画生成：将单一图像生成为指定帧数的平滑动画序列

🖼️ 背景控制
自定义背景：支持外部图像/动画作为背景

内置选项：纯黑、纯白或棋盘格背景

无输入模式：自动生成大尺寸棋盘格作为操作对象

🎭 高级功能
Alpha通道支持：完整保留透明度信息

独立Mask输出：精确表示图像透明度，不含背景

固定输出尺寸：确保动画序列连贯性

🛠️ 接口说明
输入参数
| 参数名称               | 类型    | 说明                                                                 |
|------------------------|---------|----------------------------------------------------------------------|
| `frames`               | int     | 要生成的动画帧数                                                    |
| `output_width`         | int     | 输出图像宽度（无输入图像时生效）                                    |
| `output_height`        | int     | 输出图像高度（无输入图像时生效）                                    |
| `translate_x`          | float   | X轴最终平移量（第一帧为0，最后一帧为该值）                          |
| `scale_animate_start`  | float   | 动画第一帧的缩放比例                                                |
| `focal_length`         | float   | 3D透视变换的焦距（值越大透视效果越弱）                              |
| `bg_color`             | enum    | 背景选项：`"black"`/`"white"`/`"checker"`（默认：`"black"`）         |
|output_padding	float	   |         |图像内容与画布边缘的内边距                                        |
可选输入
background_image (IMAGE): 动画背景图像序列

输出
IMAGE (IMAGE): 变换后的RGBA图像序列

MASK (MASK): 纯Alpha通道序列

📌 使用提示
当不连接image输入时，节点会生成内部棋盘格

output_padding参数仅控制布局，不影响图像缩放

背景图像会自动缩放以适应输出尺寸

Mask输出完全透明(0值)的背景区域，便于合成




<img width="1589" alt="1" src="https://github.com/user-attachments/assets/3292ae57-779e-41b2-b03c-55c8127de0b0" />

https://github.com/user-attachments/assets/4b267a4e-91a0-452a-81f0-04e4d28dd417   

https://github.com/user-attachments/assets/53d08339-a055-40ff-8092-31488e2ee27f

https://github.com/user-attachments/assets/30d7daa4-0793-473d-ab52-9b8dc4a6fc8d

https://github.com/user-attachments/assets/5f5bac22-fb04-46bd-93d9-55a71af3c2e9

https://github.com/user-attachments/assets/837f2982-7314-47db-a9a7-a6423cdc2418



