import torch
import torch.nn as nn
import torch.nn.functional as F


def do_CL(X, Y):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, 0.1)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc


def dual_CL(X, Y):
    CL_loss_1, CL_acc_1 = do_CL(X, Y)
    CL_loss_2, CL_acc_2 = do_CL(Y, X)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2
