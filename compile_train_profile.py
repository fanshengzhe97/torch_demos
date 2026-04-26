import torch
import torchvision
import torch.profiler

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

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

# 采样5%的数据
sample_size = int(len(train_dataset) * 0.05)
indices = torch.randperm(len(train_dataset))[:sample_size]
train_dataset = torch.utils.data.Subset(train_dataset, indices)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)
train_len = len(train_loader)

model = torchvision.models.resnet18()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
model.train()
model = model.to("xpu")
criterion = criterion.to("xpu")
model = torch.compile(model)

print(f"Initiating training with torch compile")

# 生成profetto可可视化的trace文件，增加详细的设备活动捕获
# 尝试添加XPU活动类型
activities = [torch.profiler.ProfilerActivity.CPU]
# 检查是否有XPU活动类型
if hasattr(torch.profiler.ProfilerActivity, 'XPU'):
    activities.append(torch.profiler.ProfilerActivity.XPU)
elif hasattr(torch.profiler.ProfilerActivity, 'CUDA'):
    activities.append(torch.profiler.ProfilerActivity.CUDA)

with torch.profiler.profile(
    on_trace_ready=lambda prof: prof.export_chrome_trace('./profile_trace_xpu.json'),
    record_shapes=False,
    with_stack=False,
    activities=activities,
    profile_memory=True,
    with_flops=True
) as prof:
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # 只分析前10个batch
            break
        data = data.to("xpu")
        target = target.to("xpu")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 5 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
        prof.step()  # 记录每个step

print("Profiling completed")
# 打印profiler结果摘要
print(prof.key_averages().table(sort_by="cuda_time_total"))
# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "checkpoint.pth",
# )

print("Execution finished")