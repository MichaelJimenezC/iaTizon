import torch

@torch.no_grad()
def f1_macro(y_true, y_pred, num_classes: int):
    f1s=[]
    for c in range(num_classes):
        tp=((y_true==c)&(y_pred==c)).sum().item()
        fp=((y_true!=c)&(y_pred==c)).sum().item()
        fn=((y_true==c)&(y_pred!=c)).sum().item()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
        f1=2*p*r/(p+r+1e-9)
        f1s.append(f1)
    return sum(f1s)/num_classes
