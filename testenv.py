import torch
import onnxruntime as ort

print("--- PyTorch 环境检查 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch 调用的 CUDA 版本: {torch.version.cuda}")

print("\n--- ONNX Runtime 环境检查 ---")
print(f"ONNX Runtime 版本: {ort.__version__}")
providers = ort.get_available_providers()
print(f"当前可用的执行提供程序: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("🎉 恭喜！ONNX Runtime 已成功调用 GPU 加速。")
else:
    print("⚠️ 警告：未找到 CUDAExecutionProvider，当前正在使用 CPU 运行。")