
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def show_prediction(model, dataset, idx, info, device, plt):
    img, label = dataset[idx]

    if hasattr(label, "__len__"):
        label = int(label[0])
    else:
        label = int(label)

    x = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5) + 0.5
    img_np = img_np.clip(0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.title(f"True: {info['label'][str(label)]}\nPred: {info['label'][str(pred)]}")
    plt.axis("off")
    plt.show()
