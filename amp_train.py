import torch
import torchvision

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

use_amp=True

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
train_len = len(train_loader)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scaler = torch.amp.GradScaler(device="xpu", enabled=use_amp)

model.train()
model = model.to("xpu")
criterion = criterion.to("xpu")

print(f"Initiating training")
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to("xpu")
    target = target.to("xpu")
    # set dtype=torch.bfloat16 for BF16
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=use_amp):
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if (batch_idx + 1) % 10 == 0:
        iteration_loss = loss.item()
        print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

print("Execution finished")