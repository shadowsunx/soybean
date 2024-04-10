import os
import torch
import time
import numpy as np


def get_matrix(f: list):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    accs = []
    c = 0
    for i, line in enumerate(f):
        if "val epoch:" in line:
            l: str = line.split("confusion_matrix: [[")[-1].split("]")[0].split(" ") # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            l = [a for a in l if a != '']
            # print(l)
            TN = float(l[0])
            FP = float(l[1])
        if '[' in line and ']]' in line:
            c += 1
            l: str = line.split('[ ')[1].split(']]')[0].split(' ')
            l = [a for a in l if a != '']
            # print(l)
            FN = float(l[0])
            TP = float(l[1])

            precision = float(TP) / float(FP + TP) if float(FP + TP) != 0 else 0  # precision
            recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0  # recall
            f1 = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            accs.append(dict(iou=iou)) 
            accs.append(dict(f1=f1)) 
            accs.append(dict(recall=recall)) 
            accs.append(dict(precision=precision)) 
            # print(accs)
            # exit(0)
    accs = accs[:151 * 4]

    accs = dict(iou=[a['iou'] for a in accs if 'iou' in a], f1=[a['f1'] for a in accs if 'f1' in a],
                recall=[a['recall'] for a in accs if 'recall' in a], precision=[a['precision'] for a in accs if 'precision' in a])
    x_axis = range(len(accs['iou']))  
    return x_axis, accs


def get_acc(f: list, split_ema=True):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    emaaccs = None
    accs = []
    for i, line in enumerate(f):
        if "val epoch:" in line:
            l: str = line.split("miou:")[-1].strip(" ").split(",")[0] # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            accs.append(dict(miou=float(l))) 
            accs = accs[:151]
    accs = dict(miou=[a['miou'] for a in accs])
    x_axis = range(len(accs['miou']))  
    return x_axis, accs


def get_loss(f: list, x1e=torch.tensor(list(range(0, 1253, 10))).view(1, -1) / 1253, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    losses = []
    for i, line in enumerate(f):
        if "train: epoch" in line:
            l = line.split("loss:")[1].strip(" ").split(",")[0]  # [20.4382, (12.4844)]
            losses.append(float(l))
            losses = losses[:18*150]

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]

    return x_axis, losses


def draw_fig(data: list, xlim=(0, 301), ylim=(68, 84), xstep=None,ystep=None, save_path="./show.jpg"):
    assert isinstance(data[0], dict)
    from matplotlib import pyplot as plot
    fig, ax = plot.subplots(dpi=300, figsize=(24, 10))
    for d in data:
        length = min(len(d['x']), len(d['y']))
        x_axis = d['x'][:length]
        y_axis = d['y'][:length]
        label = d['label']
        ax.plot(x_axis, y_axis, label=label)
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.legend()

    ax.set_xlabel('Epoch', fontsize=30)
    # ax.set_ylabel('IoU', fontsize=30)
    ax.legend(fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

    if xstep is not None:
        plot.xticks(torch.arange(xlim[0], xlim[1], xstep).tolist())
    if ystep is not None:
        plot.yticks(torch.arange(ylim[0], ylim[1], ystep).tolist())
    plot.grid()
    # plot.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot.savefig(save_path)


def main_vssm():
    showpath = os.path.join(os.path.dirname(__file__), "./show/log")
    
    vssmdtiny = "results/2242-422-crack/log/train.info.log"
    vssmdtiny1 = "results/2242-422-crack/2222-2221.log"

    x, accs = get_acc(vssmdtiny, split_ema=False)
    x1, accs1 = get_acc(vssmdtiny1, split_ema=False)
    print(f"Max: {max(accs['miou'])} epoch: {accs['miou'].index(max(accs['miou'])) + 1}")
    # print(f"accs: {accs['acc1'][-1]}")
    # print(f"emaaccs: {emaaccs['acc1'][-1]}")
    lx, losses = get_loss(vssmdtiny, x1e=torch.tensor(list(range(0, 173, 10))).view(1, -1) / 173, scale=1)
    lx1, losses1 = get_loss(vssmdtiny1, x1e=torch.tensor(list(range(0, 173, 10))).view(1, -1) / 173, scale=1)
    vssmdtiny = dict(xaxis=x, accs=accs, loss_xaxis=lx, losses=losses)
    vssmdtiny1 = dict(xaxis=x1, accs=accs1, loss_xaxis=lx1, losses=losses1)

    x_m, info_m = get_matrix('results/2242-422-crack/log/train.info.log')
    vssmdtiny_m = dict(xaxis=x_m, info=info_m)

    # if True:
    #     draw_fig(data=[
    #         dict(x=vssmdtiny['xaxis'], y=vssmdtiny['accs']['miou'], label="VMamba-UNet"),
    #         # ======================================================================
    #         dict(x=vssmdtiny1['xaxis'], y=vssmdtiny1['accs']['miou'], label="VMamba-UNet"),
    #     ], xlim=(0, 150), ylim=(0, 1), xstep=20, ystep=0.1, save_path=f"{showpath}/miou.jpg")

    if True:
        draw_fig(data=[
            dict(x=vssmdtiny_m['xaxis'], y=vssmdtiny_m['info']['iou'], label="iou"),
            dict(x=vssmdtiny_m['xaxis'], y=vssmdtiny_m['info']['precision'], label="precision"),
            dict(x=vssmdtiny_m['xaxis'], y=vssmdtiny_m['info']['recall'], label="recall"),
            dict(x=vssmdtiny_m['xaxis'], y=vssmdtiny_m['info']['f1'], label="f1"),
        ], xlim=(0, 150), ylim=(0, 1), xstep=20, ystep=0.1, save_path=f"{showpath}/all.jpg")

    # if True:
    #     draw_fig(data=[
    #         dict(x=vssmdtiny['loss_xaxis'], y=vssmdtiny['losses'], label="VMamba-UNet"),
    #         dict(x=vssmdtiny1['loss_xaxis'], y=vssmdtiny1['losses'], label="VMamba-UNet"),
    #     ], xlim=(0, 150), ylim=(0, 1.8), xstep=20, ystep=0.2, save_path=f"{showpath}/loss.jpg")



main_vssm()
