HOOTOO_ImageTransform - 3D图像变换工具，灵感来源于PT娃导

HOOTOO_ImageTransform 是一个功能强大的3D图像变换工具节点，它能够对静态图像或默认生成的棋盘格进行复杂的动画变换，支持2D/3D变换、自定义背景和Mask输出。

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
| 参数分类       | 参数名               | 类型/范围                     | 默认值   | 说明                                                                 |
|----------------|----------------------|-------------------------------|----------|----------------------------------------------------------------------|
| **动画控制**   | `frames`             | INT [1, 9999]                | 10       | 动画总帧数（线性插值）                                               |
|                | `output_width`       | INT [64,4096] (step=8)       | 512      | 无输入图像时的输出宽度                                               |
|                | `output_height`      | INT [64,4096] (step=8)       | 512      | 无输入图像时的输出高度                                               |
| **2D变换**     | `translate_x`        | INT [-5000,5000]             | 0        | X轴平移量（像素，末帧达到设定值）                                    |
|                | `translate_y`        | INT [-5000,5000]             | 0        | Y轴平移量                                                            |
|                | `scale_animate_start`| FLOAT [0.01,10.0]            | 1.0      | 起始帧缩放比例                                                       |
|                | `scale_animate_end`  | FLOAT [0.01,10.0]            | 1.0      | 结束帧缩放比例                                                       |
| **3D变换**     | `rotate_x`           | FLOAT [-179,179] (step=0.5)  | 0.0      | X轴旋转角度（俯仰角）                                                |
|                | `rotate_y`           | FLOAT [-179,179] (step=0.5)  | 0.0      | Y轴旋转角度（偏航角）                                                |
|                | `rotate_z`           | FLOAT [-360,360] (step=0.5)  | 0.0      | Z轴旋转角度（平面旋转）                                              |
|                | `focal_length`       | FLOAT [10.0,5000.0]          | 1000.0   | 透视强度（值越小效果越强）                                           |
| **背景控制**   | `bg_color`           | black/white/checker          | black    | 无背景输入时的填充方案                                               |
|                | `output_padding`     | INT [0,500]                  | 0        | 图像内边距（像素）                                                   |

可选输入
background_image (IMAGE): 动画背景图像序列

输出
IMAGE (IMAGE): 变换后的RGB图像序列

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



