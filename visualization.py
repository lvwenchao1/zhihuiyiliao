import re
import matplotlib.pyplot as plt

# 日志文件内容
log_data = """
2025-01-05 12:33:46,159 - Epoch 1:
Train Loss: 0.1780, Train Accuracy: 0.9352, Train ROC-AUC: 0.9761, Train F1: 0.9562, Train Precision: 0.9592, Train Recall: 0.9533
Val Loss: 0.1068, Val Accuracy: 0.9618, Val ROC-AUC: 0.9911, Val F1: 0.9741, Val Precision: 0.9817, Val Recall: 0.9665

2025-01-05 12:33:47,353 - Epoch 1: Best model saved!
2025-01-05 12:35:31,019 - Epoch 2:
Train Loss: 0.1011, Train Accuracy: 0.9618, Train ROC-AUC: 0.9916, Train F1: 0.9743, Train Precision: 0.9724, Train Recall: 0.9762
Val Loss: 0.2132, Val Accuracy: 0.9436, Val ROC-AUC: 0.9942, Val F1: 0.9607, Val Precision: 0.9945, Val Recall: 0.9292

2025-01-05 12:37:09,121 - Epoch 3:
Train Loss: 0.0969, Train Accuracy: 0.9654, Train ROC-AUC: 0.9926, Train F1: 0.9767, Train Precision: 0.9759, Train Recall: 0.9775
Val Loss: 0.0884, Val Accuracy: 0.9675, Val ROC-AUC: 0.9939, Val F1: 0.9782, Val Precision: 0.9757, Val Recall: 0.9807

2025-01-05 12:37:10,663 - Epoch 3: Best model saved!
2025-01-05 12:39:05,720 - Epoch 4:
Train Loss: 0.0783, Train Accuracy: 0.9720, Train ROC-AUC: 0.9945, Train F1: 0.9812, Train Precision: 0.9816, Train Recall: 0.9807
Val Loss: 0.0898, Val Accuracy: 0.9694, Val ROC-AUC: 0.9962, Val F1: 0.9792, Val Precision: 0.9908, Val Recall: 0.9678

2025-01-05 12:39:06,969 - Epoch 4: Best model saved!
2025-01-05 12:41:06,186 - Epoch 5:
Train Loss: 0.0657, Train Accuracy: 0.9768, Train ROC-AUC: 0.9965, Train F1: 0.9844, Train Precision: 0.9845, Train Recall: 0.9842
Val Loss: 0.0641, Val Accuracy: 0.9761, Val ROC-AUC: 0.9972, Val F1: 0.9840, Val Precision: 0.9772, Val Recall: 0.9910

2025-01-05 12:41:07,387 - Epoch 5: Best model saved!
2025-01-05 12:43:10,618 - Epoch 6:
Train Loss: 0.0669, Train Accuracy: 0.9754, Train ROC-AUC: 0.9962, Train F1: 0.9834, Train Precision: 0.9823, Train Recall: 0.9845
Val Loss: 0.0963, Val Accuracy: 0.9685, Val ROC-AUC: 0.9958, Val F1: 0.9785, Val Precision: 0.9895, Val Recall: 0.9678

2025-01-05 12:45:14,453 - Epoch 7:
Train Loss: 0.0595, Train Accuracy: 0.9751, Train ROC-AUC: 0.9970, Train F1: 0.9833, Train Precision: 0.9826, Train Recall: 0.9839
Val Loss: 0.2771, Val Accuracy: 0.9351, Val ROC-AUC: 0.9933, Val F1: 0.9544, Val Precision: 0.9958, Val Recall: 0.9163

2025-01-05 12:47:03,636 - Epoch 8:
Train Loss: 0.0533, Train Accuracy: 0.9775, Train ROC-AUC: 0.9977, Train F1: 0.9849, Train Precision: 0.9852, Train Recall: 0.9845
Val Loss: 0.0894, Val Accuracy: 0.9733, Val ROC-AUC: 0.9966, Val F1: 0.9822, Val Precision: 0.9687, Val Recall: 0.9961

2025-01-05 12:48:34,572 - Epoch 9:
Train Loss: 0.0620, Train Accuracy: 0.9768, Train ROC-AUC: 0.9967, Train F1: 0.9844, Train Precision: 0.9852, Train Recall: 0.9836
Val Loss: 0.1799, Val Accuracy: 0.9446, Val ROC-AUC: 0.9929, Val F1: 0.9615, Val Precision: 0.9918, Val Recall: 0.9331

2025-01-05 12:50:04,537 - Epoch 10:
Train Loss: 0.0476, Train Accuracy: 0.9821, Train ROC-AUC: 0.9982, Train F1: 0.9879, Train Precision: 0.9884, Train Recall: 0.9874
Val Loss: 0.1509, Val Accuracy: 0.9456, Val ROC-AUC: 0.9939, Val F1: 0.9621, Val Precision: 0.9945, Val Recall: 0.9318


"""  # 这里省略了部分日志内容，请将完整日志粘贴到这里

# 解析日志函数
# 修改解析日志函数，保留最后一个 Best Epoch
def parse_logs(log_data):
    pattern = r"Epoch (\d+):\nTrain Loss: ([\d.]+), Train Accuracy: ([\d.]+), Train ROC-AUC: ([\d.]+), Train F1: ([\d.]+), Train Precision: ([\d.]+), Train Recall: ([\d.]+)\nVal Loss: ([\d.]+), Val Accuracy: ([\d.]+), Val ROC-AUC: ([\d.]+), Val F1: ([\d.]+), Val Precision: ([\d.]+), Val Recall: ([\d.]+)"
    matches = re.findall(pattern, log_data)

    results = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_roc_auc": [],
        "train_f1": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "best_epoch": None  # 保留最后一个 Best Epoch
    }

    for match in matches:
        results["epoch"].append(int(match[0]))
        results["train_loss"].append(float(match[1]))
        results["train_accuracy"].append(float(match[2]))
        results["train_roc_auc"].append(float(match[3]))
        results["train_f1"].append(float(match[4]))
        results["train_precision"].append(float(match[5]))
        results["train_recall"].append(float(match[6]))
        results["val_loss"].append(float(match[7]))
        results["val_accuracy"].append(float(match[8]))
        results["val_roc_auc"].append(float(match[9]))
        results["val_f1"].append(float(match[10]))
        results["val_precision"].append(float(match[11]))
        results["val_recall"].append(float(match[12]))

    # 查找最后一次最佳模型保存的 Epoch
    best_epochs = [int(m.group(1)) for m in re.finditer(r"Epoch (\d+): Best model saved!", log_data)]
    if best_epochs:
        results["best_epoch"] = best_epochs[-1]  # 保留最后一个 Best Epoch

    return results

# 解析日志
results = parse_logs(log_data)

# 可视化函数
# 修改后的可视化函数，增加 F1、Precision、Recall
def visualize_results(results):
    epochs = results["epoch"]
    best_epoch = results["best_epoch"]  # 最后一个最佳 Epoch

    # 绘制训练和验证 Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, results["val_loss"], label="Validation Loss", marker='o')
    if best_epoch:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f"Best Epoch {best_epoch}")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制训练和验证 Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results["train_accuracy"], label="Train Accuracy", marker='o')
    plt.plot(epochs, results["val_accuracy"], label="Validation Accuracy", marker='o')
    if best_epoch:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f"Best Epoch {best_epoch}")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制训练和验证 ROC-AUC
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results["train_roc_auc"], label="Train ROC-AUC", marker='o')
    plt.plot(epochs, results["val_roc_auc"], label="Validation ROC-AUC", marker='o')
    if best_epoch:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f"Best Epoch {best_epoch}")
    plt.title("Training and Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制 Validation F1、Precision 和 Recall
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results["val_f1"], label="Validation F1", marker='o')
    plt.plot(epochs, results["val_precision"], label="Validation Precision", marker='o')
    plt.plot(epochs, results["val_recall"], label="Validation Recall", marker='o')
    if best_epoch:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f"Best Epoch {best_epoch}")
    plt.title("Validation Metrics (F1, Precision, Recall)")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用可视化函数
visualize_results(results)

