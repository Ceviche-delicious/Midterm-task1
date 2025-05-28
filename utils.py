import torch
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import datetime
import random
import matplotlib.pyplot as plt

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_caltech101(root_path: str = './data',
                    test_size: float = 0.20,
                    val_size: float  = 0.10,
                    random_state: int = 42):

    random.seed(random_state); torch.manual_seed(random_state)

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size 必须 < 1")

    img_dir = os.path.join(root_path, '101_ObjectCategories')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"数据集路径不存在：{img_dir}")

    dataset = ImageFolder(
        root=img_dir,
        transform=train_transform,
        is_valid_file=lambda x: x.lower().endswith(('.jpg', '.png'))
    )

    classes = [d.name for d in os.scandir(img_dir) if d.is_dir()]
    classes.sort()                          

    train_idx, val_idx, test_idx = [], [], []

    for class_idx, _ in enumerate(classes):
        cls_samples = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == class_idx]

        train_val, cls_test = train_test_split(
            cls_samples,
            test_size=test_size,
            random_state=random_state,
            stratify=None        
        )

        rel_val_size = val_size / (1.0 - test_size)   
        cls_train, cls_val = train_test_split(
            train_val,
            test_size=rel_val_size,
            random_state=random_state,
            stratify=None
        )

        train_idx.extend(cls_train)
        val_idx.extend(cls_val)
        test_idx.extend(cls_test)

    train_subset = Subset(dataset, train_idx)

    val_subset = copy.deepcopy(Subset(dataset, val_idx))
    val_subset.dataset.transform = test_transform    

    test_subset = copy.deepcopy(Subset(dataset, test_idx))
    test_subset.dataset.transform = test_transform    

    return train_subset, val_subset, test_subset

def get_dataloaders(root_path: str = './data',
                            batch_size: int = 256,
                            test_size: float = 0.20,
                            val_size: float = 0.10,
                            random_state: int = 42,
                            num_workers: int = 0):
    """
    获取 Caltech-101 数据集的 DataLoader
    -------------------------------------------------
    Parameters
    ----------
    root_path    : Caltech‑101 解压根目录，内部应含 '101_ObjectCategories'
    batch_size   : 每个 batch 的样本数
    test_size    : 测试集占比（0–1 之间）
    val_size     : 验证集占比（0–1 之间，注意与 test_size 之和须 < 1）
    random_state : 随机种子，保证可复现

    Returns
    -------
    train_loader : 训练集 DataLoader
    val_loader   : 验证集 DataLoader
    test_loader  : 测试集 DataLoader
    """
    train_subset, val_subset, test_subset = load_caltech101(
        root_path, test_size, val_size, random_state)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_model(num_classes=101, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * y.size(0)
        _, pred_lbl = pred.max(1)
        correct += (pred_lbl == y).sum().item()
        total += y.size(0)
    return epoch_loss/total, correct/total

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss += criterion(pred, y).item() * y.size(0)
        _, pred_lbl = pred.max(1)
        correct += pred_lbl.eq(y).sum().item()
        total += y.size(0)
    return loss/total, correct/total

def train_model(model, optimizer, scheduler, criterion, exp_name="finetune", epochs=30, 
                batch_size=32, device="cuda:0", save_best=True):
    
    tr_loss_record = []
    val_loss_record = []
    tr_acc_record = []
    val_acc_record = []

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    writer = SummaryWriter(log_dir=f"runs/{exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    best_val_acc = 0

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # 记录损失和准确率
        tr_loss_record.append(tr_loss)
        val_loss_record.append(val_loss)
        tr_acc_record.append(tr_acc)
        val_acc_record.append(val_acc)

        # TensorBoard logs
        writer.add_scalar("Loss/Train", tr_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss, epoch)
        writer.add_scalar("Acc/Train",    tr_acc,  epoch)
        writer.add_scalar("Acc/Val",    val_acc,  epoch)

        print(f"[{epoch:02}/{epochs}] "
              f"Train L={tr_loss:.3f}|Acc={tr_acc:.3f} "
              f"Val L={val_loss:.3f}|Acc={val_acc:.3f}")

        # 保存最好模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_best:
                torch.save(model.state_dict(), f"{exp_name}_best.pth")

    writer.close()
    print(f'Best Val Acc: {best_val_acc:.4f}')

    return {
        "train_loss": tr_loss_record,
        "val_loss": val_loss_record,
        "train_acc": tr_acc_record,
        "val_acc": val_acc_record,
        "best_val_acc": best_val_acc,
    }

def train_search(model, optimizer, scheduler, criterion, epochs=10, 
                batch_size=128, device="cuda:0"):
    
    tr_loss_record = []
    val_loss_record = []
    tr_acc_record = []
    val_acc_record = []

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    best_val_acc = 0

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        tr_loss_record.append(tr_loss)
        val_loss_record.append(val_loss)
        tr_acc_record.append(tr_acc)
        val_acc_record.append(val_acc)

        print(f"[{epoch:02}/{epochs}] "
              f"Train L={tr_loss:.3f}|Acc={tr_acc:.3f} "
              f"Val L={val_loss:.3f}|Acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return {
        "train_loss": tr_loss_record,
        "val_loss": val_loss_record,
        "train_acc": tr_acc_record,
        "val_acc": val_acc_record,
        "best_val_acc": best_val_acc,
    }



def plot(record, name, hyperparams, save_path='training_plots_random'):
    os.makedirs(save_path, exist_ok=True)

    epochs = range(1, len(record['train_loss']) + 1)
    
    plt.figure(figsize=(8, 12))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, record['train_loss'], label='train loss')
    plt.plot(epochs, record['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and validation loss curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, record['train_acc'], label='train acc')
    plt.plot(epochs, record['val_acc'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and validation accuracy curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{name}_BS_{hyperparams["batch_size"]}_LR_{hyperparams["lr"]}_Step_Size_{hyperparams["step_size"]}_Gamma_{hyperparams["gamma"]}_Weight_Decay_{hyperparams["weight_decay"]}.png')
    plt.close()

