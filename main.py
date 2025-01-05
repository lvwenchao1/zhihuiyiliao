import json
import os

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging
from torchinfo import summary
import random
import numpy as np


def set_seed(seed):
    """
    设置随机种子以确保结果可复现。
    :param seed: 整数，随机种子
    """
    random.seed(seed)  # 固定 Python 的随机种子
    np.random.seed(seed)  # 固定 NumPy 的随机种子
    torch.manual_seed(seed)  # 固定 PyTorch 的CPU随机种子
    torch.cuda.manual_seed(seed)  # 固定 PyTorch 的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 固定所有GPU的随机种子（如果有多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作的结果一致
    torch.backends.cudnn.benchmark = False  # 关闭自动优化以保证可复现性


# 示例用法
set_seed(42)  # 固定随机种子

# 定义路径
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"


# 定义数据集类
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = 0 if "NORMAL" in img_path else 1  # NORMAL: 0, PNEUMONIA: 1
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义一个函数读取图像路径并合并
def load_data(train_dir, val_dir):
    data = {"NORMAL": [], "PNEUMONIA": []}
    for category in data.keys():
        # 读取 train 和 val 下的所有图像路径
        train_images = glob.glob(os.path.join(train_dir, category, "*.jpeg"))
        val_images = glob.glob(os.path.join(val_dir, category, "*.jpeg"))
        # 合并路径
        data[category] = train_images + val_images
    return data


# 创建日志文件
logging.basicConfig(
    filename="training.log",  # 日志文件名
    level=logging.INFO,  # 日志等级
    format="%(asctime)s - %(message)s"  # 日志格式
)

# 用于记录每个 epoch 的指标
metrics_log = []


# 定义函数记录指标到日志和文件
def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, train_metrics, val_metrics):
    log_message = (
        f"Epoch {epoch + 1}:\n"
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
        f"Train ROC-AUC: {train_metrics['roc_auc']:.4f}, Train F1: {train_metrics['f1']:.4f}, "
        f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}\n"
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
        f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
        f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}\n"
    )
    logging.info(log_message)

    # 保存到 JSON 文件
    metrics_log.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    })


# 定义一个函数计算指标
def calculate_metrics(outputs, labels):
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    return {
        "roc_auc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 数据增强和预处理
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转，限制在±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 调整亮度和对比度
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 随机裁剪并调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

if __name__ == '__main__':
    # 加载数据
    data = load_data(train_dir, val_dir)

    # 划分训练集和验证集
    train_data = {"NORMAL": [], "PNEUMONIA": []}
    val_data = {"NORMAL": [], "PNEUMONIA": []}

    for category, images in data.items():
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        train_data[category] = train_images
        val_data[category] = val_images

    # 输出统计结果
    print("训练集:")
    print(f"  NORMAL: {len(train_data['NORMAL'])} images")
    print(f"  PNEUMONIA: {len(train_data['PNEUMONIA'])} images")

    print("验证集:")
    print(f"  NORMAL: {len(val_data['NORMAL'])} images")
    print(f"  PNEUMONIA: {len(val_data['PNEUMONIA'])} images")

    # 合并数据并创建 Dataset
    train_images = train_data["NORMAL"] + train_data["PNEUMONIA"]
    val_images = val_data["NORMAL"] + val_data["PNEUMONIA"]
    train_dataset = ChestXRayDataset(train_images, transform=train_transforms)
    val_dataset = ChestXRayDataset(val_images, transform=data_transforms)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
                              pin_memory=True
                              )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4,
                            pin_memory=True
                            )

    # 定义模型
    name = 'resnet18'
    model = models.resnet18(pretrained=True)
    model.eval()
    summary(model, input_size=(64, 3, 224, 224))
    model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层分类器，输出2类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 存储所有标签和预测结果用于计算训练指标
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 存储预测结果和标签
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
            all_train_probs.extend(torch.softmax(outputs, dim=1)[:].detach().cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        # 计算额外指标
        train_metrics = calculate_metrics(
            torch.tensor(all_train_probs),
            torch.tensor(all_train_labels)
        )
        train_metrics['acc'] = train_acc
        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 存储所有标签和预测结果用于计算验证指标
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 存储预测结果和标签
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(torch.softmax(outputs, dim=1)[:].detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_metrics = calculate_metrics(
            torch.tensor(all_val_probs),
            torch.tensor(all_val_labels)
        )
        val_metrics['acc'] = val_acc
        # 记录指标
        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, train_metrics, val_metrics)

        # 保存最佳模型
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(model.state_dict(), name + "best_model.pth")
            logging.info(f"Epoch {epoch + 1}: Best model saved!")

    # 将所有指标保存到 JSON 文件
    with open(name + "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print("Training complete!")
