import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# ====================== 基础配置（所有参数保持原样不变） ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./skin_dataset"
BATCH_SIZE = 48
EPOCHS = 15
LR = 6e-5
NUM_CLASSES = 3
CLASS_NAMES = ["mel", "nv", "vasc"]

# ====================== 修改：训练集取消额外在线增强，仅标准化 ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====================== 类别权重（参数完全保留原始） ======================
class_nums = [58, 92, 26]
total_num = sum(class_nums)
base_weights = torch.tensor([total_num / cnt for cnt in class_nums], dtype=torch.float32)
base_weights[0] *= 3.0# 加强 mel 惩罚
class_weights = base_weights.to(DEVICE)
print(f"类别权重 mel:nv:vasc = {class_weights[0]:.2f} : {class_weights[1]:.2f} : {class_weights[2]:.2f}")

# ====================== 模型构建（冻结比例、全连接层完全不变） ======================
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

total_params = len(list(model.parameters()))
freeze_num = int(total_params * 0.70)
for idx, param in enumerate(model.parameters()):
    if idx < freeze_num:
        param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(64, NUM_CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1.8e-3)

# ====================== 训练日志 & 保存最优结果（训练逻辑原样） ======================
train_losses = []
test_losses = []
train_accs = []
test_accs = []

best_acc = 0.0
best_preds = []
best_labels = []

print("\n============ 训练开始 ============\n")

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    model.train()
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

        print(f"[Train] Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_acc = test_correct / test_total

    # 记录当前轮指标
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # 只保存【全局准确率最高】的一轮结果
    if test_acc > best_acc:
        best_acc = test_acc
        best_preds = all_preds.copy()
        best_labels = all_labels.copy()
        torch.save(model.state_dict(), "best_resnet.pth")
        print(f"✅ 当前最佳准确率更新: {best_acc:.4f}")

print(f"\n🎉 训练结束 | 全局最高测试准确率: {best_acc:.4f}")

# ====================== 绘制损失 & 准确率曲线 ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("train_curve.png")
plt.show()

# ====================== 混淆矩阵+报告（代码不变） ======================
print("\n========== 【准确率最高轮次】分类评估报告 ==========")
cm = confusion_matrix(best_labels, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Best Accuracy Epoch)")
plt.savefig("confusion_matrix_best.png")
plt.show()

print(classification_report(best_labels, best_preds, target_names=CLASS_NAMES))