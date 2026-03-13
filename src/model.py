
import torch
import timm
from torch import nn

def create_model(num_classes, device):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes
    )
    model = model.to(device)
    return model

def get_loss_optimizer_scheduler_scaler(model, lr=3e-4, weight_decay=1e-4, t_max=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max
    )
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler, scaler
