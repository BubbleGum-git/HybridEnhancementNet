import time, torch, numpy as np, onnx, onnxruntime as ort
from thop import profile
from src.model import HybridEnhancementNet

H, W = 480, 640                       # your camera resolution
CKPT = r"O:\Project_files\ClearSight\HybridEnhancementNet-master\best_model.pth"

model = HybridEnhancementNet()
ckpt = torch.load(CKPT, map_location="cpu")
if isinstance(ckpt, dict):
    ckpt = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict(ckpt)
model.eval()

x = torch.randn(1, 3, H, W)
macs, params = profile(model, inputs=(x,), verbose=False)
print(f"Params: {params/1e6:.3f} M | GFLOPs: {2*macs/1e9:.2f}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
m, xi = model.to(dev), x.to(dev)
with torch.no_grad():
    for _ in range(20): m(xi)
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(100): m(xi)
    if dev == "cuda": torch.cuda.synchronize()
ms = (time.time() - t) * 10
print(f"{dev} latency: {ms:.1f} ms ({1000/ms:.0f} FPS)")
if dev == "cuda":
    print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

# ONNX export + correctness check
model.cpu()
torch.onnx.export(model, x, "enh.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"],
                  dynamo=False)
onnx.checker.check_model(onnx.load("enh.onnx"))
out_o = ort.InferenceSession("enh.onnx").run(None, {"input": x.numpy()})[0]
print("Max diff PyTorch vs ONNX:", np.abs(out_o - model(x).detach().numpy()).max())