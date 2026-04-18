import torch
import torchvision.models as models
import time

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)
ITERS = 10

model = model.to("xpu")
data = data.to("xpu")

for i in range(ITERS):
    start = time.time()
    with torch.no_grad():
        model(data)
        torch.xpu.synchronize()
    end = time.time()
    print(f"Inference time before torch.compile for iteration {i}: {(end-start)*1000} ms")

model = torch.compile(model)
for i in range(ITERS):
    start = time.time()
    with torch.no_grad():
        model(data)
        torch.xpu.synchronize()
    end = time.time()
    print(f"Inference time after torch.compile for iteration {i}: {(end-start)*1000} ms")

print("Execution finished")