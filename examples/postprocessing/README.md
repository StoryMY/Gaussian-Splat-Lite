# 可选神经网络后处理

查看器的 Render options 中开启 PixRestore。默认关闭，首次开启才加载 ORT 和模型。
开启“512×512 验证模式”可固定渲染与捕获分辨率、使用方形相机视口；关闭后恢复窗口尺寸。
停止操作 450 ms 后捕获一次；相机、窗口或场景变化立即隐藏旧结果。
“重新处理”用于手动重复测量。关闭后释放会话和缓冲区。
切换渲染后端会关闭后处理，可在原生 WebGPU 下重新开启。

## 模型

将 PixRestore 导出的 compat 文件放到以下任一目录（模型不提交 Git）：

- `public/models/pixrestore-s-ft-d6-fp32-compat/`
- `public/models/pixrestore-s-ft-d6-fp16-compat/`

每个目录包含 `pixrestore-manifest.json`、`dino_encoder.onnx`、`pixel_dit.onnx`。
也可以通过 `?postprocessManifest=/models/custom/pixrestore-manifest.json` 指定清单。
远程清单及模型需要允许 CORS。默认 FP16，需要 shader-f16；不支持时可选择 FP32。

## 模块边界

| 模块 | 职责 |
| --- | --- |
| IdlePostProcessor | 空闲调度、串行执行、丢弃过期结果、生命周期；不依赖渲染器或 ORT |
| GslFrameSource | 唯一依赖 GSL / Three.js 的适配器：捕获并提供 GPUTexture |
| PixRestoreProcessor | GPU 预处理、DINO → Pixel DiT；不依赖 Three.js |
| GpuPresenter | 将 RGB GPUBuffer 显示到独立 Canvas，不回读 CPU |
| ortRuntime | 共用 GPUDevice，串行化 ORT 初始化、推理与释放 |
| neuralPanel | 查看器开关及状态，独立于核心库 |

替换渲染器：实现 source 的 `device / capture() / dispose()`，输入 frame 为
`{device, texture, overlay, width, height, colorSpace, release?}`，纹理原点左上。
colorSpace 表示 textureLoad 得到的是 `srgb` 还是 `linear-srgb`；输入按黑底合成。
overlay 为 presenter 使用的预乘 sRGB 辅助图层纹理，由 source 保留至下次捕获或销毁。

替换神经网络：实现 `initialize(device) / process(frame) / dispose()`。
现有 presenter 的结果协议为
`{device, buffer, width, height, dataType, rect, release()}`，
buffer 为 NCHW RGB、[-1,1]、sRGB，dataType 为 float32/float16；
rect 表示去补边的显示区域。其他输出协议可以同时替换 presenter。

source 捕获的纹理借用至 frame.release；processor 返回的结果由 controller 负责释放。
同一控制器最多一个任务，过期 GPU 工作不强行中断。
ORT 在应用中绑定单一 GPUDevice；Three.js 切换后端复用应用拥有的设备。
设备丢失后需刷新页面。

## 图像和计时

保持原画面比例，黑色补边到模型方形输入，显示时裁去补边，不把 16:9 拉伸成方形。
捕获最长边限制为 1024；GPU 双线性缩放分别生成 Pixel 和 DINO 输入。
此缩放与 PyTorch bicubic/浏览器 Canvas high-quality 不完全相同，不能据此进行逐元素对齐。
噪声采用 runner 相同的 seed 42 和 Box–Muller，在初始化时生成并上传一次。
推理使用已有 compat 导出，默认关闭 ORT 图优化。

状态耗时从捕获开始到网络 GPU 工作完成；不含首次加载、最终呈现，不是纯 GPU 时间。
正常推理不下载图像张量；只等待队列完成来提供有意义的耗时。

捕获使用 SplatCapture 的独立渲染，不包含查看器屏幕上的 TAA 历史。
网格、坐标轴和控制点不进入 PixRestore，按原始分辨率在增强后合成；遮挡依据原始高斯。
不承诺逐帧时序一致；第一版定位为停止移动后的静帧增强。

## 参考

- [Visionary GPU I/O 架构](https://ai4sports.opengvlab.com/help/modules/13-onnx/architecture/index.html)：
  共用设备、GPU I/O binding、串行推理的设计参考，未复制其实现。
- onnx-webgpu-runner：复用 PixRestore manifest、输入顺序、归一化、固定噪声与单步采样约定。

ORT 是查看器开发依赖，动态加载；未加入 GSL 核心库导出。

使用固定 ORT 1.30.0：1.29 的外部 GPUDevice 同步路径存在错误，见 [上游修复](https://github.com/microsoft/onnxruntime/pull/32259)。
升级后重启 Vite；旧依赖缓存存在时使用 `npm run dev -- --force`。
