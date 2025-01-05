# 定义测试集路径
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

from main import ChestXRayDataset, data_transforms, calculate_metrics, set_seed
test_dir = "chest_xray/test"
set_seed(42)
# 加载测试集
test_images = []
for category in ["NORMAL", "PNEUMONIA"]:
    test_images += glob.glob(os.path.join(test_dir, category, "*.jpeg"))

# 构建测试集 Dataset 和 DataLoader
test_dataset = ChestXRayDataset(test_images, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model_on_test_set(model, test_loader, criterion):
    """
    在测试集上评估模型性能，并输出相关指标。
    :param model: 训练好的模型
    :param test_loader: 测试集 DataLoader
    :param criterion: 损失函数
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # 存储所有标签和预测结果
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 存储预测结果和标签
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            all_test_probs.extend(torch.softmax(outputs, dim=1)[:].detach().cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = correct / total

    # 计算测试集的指标
    test_metrics = calculate_metrics(
        torch.tensor(all_test_probs),
        torch.tensor(all_test_labels)
    )
    test_metrics["loss"] = test_loss
    test_metrics["accuracy"] = test_acc

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}, Test F1: {test_metrics['f1']:.4f}, "
          f"Test Precision: {test_metrics['precision']:.4f}, Test Recall: {test_metrics['recall']:.4f}")

    # 将测试集结果写入日志
    logging.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, "
                 f"ROC-AUC: {test_metrics['roc_auc']:.4f}, F1: {test_metrics['f1']:.4f}, "
                 f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")

    return test_metrics


if __name__ == '__main__':
    # 加载最佳模型
    name = 'resnet18'
    # 定义模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层分类器，输出2类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_path = name + "best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    criterion = nn.CrossEntropyLoss()
    # 在测试集上评估模型
    print("Evaluating on Test Set...")
    test_metrics = evaluate_model_on_test_set(model, test_loader, criterion)

    # 将测试集指标保存到 JSON 文件
    with open(name + "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
