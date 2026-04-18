import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to("xpu")
data = data.to("xpu")

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    d = d.to("xpu")
    # set dtype=torch.bfloat16 for BF16
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
        model(data)

print("Execution finished")