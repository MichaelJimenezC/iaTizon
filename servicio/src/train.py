import torch, torch.nn as nn
from tqdm import tqdm
from .metrics import f1_macro

@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    total,n=0.0,0
    ys,ps=[],[]
    for x,y in loader:
        x,y=x.to(device, non_blocking=True),y.to(device, non_blocking=True)
        logits=model(x)
        loss=criterion(logits,y)
        total+=loss.item()*x.size(0); n+=x.size(0)
        ys.append(y.cpu()); ps.append(logits.argmax(1).cpu())
    import torch as _t
    y_true=_t.cat(ys); y_pred=_t.cat(ps)
    acc=(y_true==y_pred).float().mean().item()
    f1=f1_macro(y_true,y_pred,num_classes)
    return total/max(1,n),acc,f1

def train_loop(model, train_loader, val_loader, device, epochs, lr, weight_decay, step_size, gamma, amp, class_weights=None, ckpt_path="best_model.pth"):
    criterion=nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None), label_smoothing=0.05)
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler=torch.amp.GradScaler('cuda', enabled=amp)
    best_f1=-1.0; patience=10; bad=0
    for ep in range(epochs):
        model.train()
        total,n=0.0,0
        for i,(x,y) in enumerate(tqdm(train_loader, desc=f"epoch {ep:03d}")):
            x,y=x.to(device, non_blocking=True),y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits=model(x)
                loss=criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total+=loss.item()*x.size(0); n+=x.size(0)
        scheduler.step()
        vl,va,vf=evaluate(model, val_loader, device, criterion, model.fc.out_features)
        tr=total/max(1,n)
        print(f"train {tr:.4f} | val {vl:.4f} | acc {va:.4f} | f1 {vf:.4f}")
        if vf>best_f1+1e-5:
            best_f1=vf; bad=0
            torch.save({"model":model.state_dict()}, ckpt_path)
        else:
            bad+=1
            if bad>=patience:
                break
    print(f"best f1 {best_f1:.4f}")
