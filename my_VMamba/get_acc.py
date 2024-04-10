import os
import torch


def get_acc(f: list, split_ema=True):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    emaaccs = None
    accs = []
    for i, line in enumerate(f):
        if "* Acc" in line:
            l: str = line.split("INFO")[-1].strip(" ").split(" ") # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            accs.append(dict(acc1=float(l[2]), acc5=float(l[4]))) 
    accs = dict(acc1=[a['acc1'] for a in accs], acc5=[a['acc5'] for a in accs])
    if split_ema:
        emaaccs = dict(acc1=[a for i, a in enumerate(accs['acc1']) if i % 2 == 1], 
                       acc5=[a for i, a in enumerate(accs['acc5']) if i % 2 == 1])
        accs = dict(acc1=[a for i, a in enumerate(accs['acc1']) if i % 2 == 0], 
                       acc5=[a for i, a in enumerate(accs['acc5']) if i % 2 == 0])
    x_axis = range(len(accs['acc1']))  
    return x_axis, accs, emaaccs


def get_loss(f: list, x1e=torch.tensor(list(range(0, 1253, 10))).view(1, -1) / 1253, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    avglosses = []
    losses = []
    for i, line in enumerate(f):
        if "Train: [" in line and ("loss" in line):
            l = line.split("loss")[1].strip(" ").split(" ")[:2]  # [20.4382, (12.4844)]
            losses.append(float(l[0]))
            avglosses.append(float(l[1].split(")")[0].strip("()")))

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]
    avglosses = [l * scale for l in avglosses]

    return x_axis, losses, avglosses


def draw_fig(data: list, xlim=(0, 301), ylim=(68, 84), xstep=None,ystep=None, save_path="./show.jpg"):
    assert isinstance(data[0], dict)
    from matplotlib import pyplot as plot
    fig, ax = plot.subplots(dpi=300, figsize=(24, 8))
    for d in data:
        length = min(len(d['x']), len(d['y']))
        x_axis = d['x'][:length]
        y_axis = d['y'][:length]
        label = d['label']
        ax.plot(x_axis, y_axis, label=label)
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.legend()
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
    
    vssmdtiny = "classification/tmp/vssm_tiny/20240318152312/log_rank0.txt"

    x, accs, emaaccs = get_acc(vssmdtiny, split_ema=False)
    print(f"epoch: {len(accs['acc1']) - 1}")
    print(f"accs: {accs['acc1'][-1]}")
    # print(f"emaaccs: {emaaccs['acc1'][-1]}")
    lx, losses, avglosses = get_loss(vssmdtiny, x1e=torch.tensor(list(range(0, 250, 10))).view(1, -1) / 250, scale=1)
    vssmdtiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    if True:
        draw_fig(data=[
            dict(x=vssmdtiny['xaxis'], y=vssmdtiny['accs']['acc1'], label="vssmdtiny"),
            # ======================================================================
            # dict(x=vssmdtiny['xaxis'], y=vssmdtiny['emaaccs']['acc1'], label="vssmdtiny_ema"),
        ], xlim=(0, 300), ylim=(0, 100), xstep=10, ystep=5, save_path=f"{showpath}/acc_vssmd.jpg")

    if True:
        draw_fig(data=[
            dict(x=vssmdtiny['loss_xaxis'], y=vssmdtiny['avglosses'], label="vssmdtiny"),
        ], xlim=(0, 300), ylim=(0, 7), xstep=10, ystep=0.5, save_path=f"{showpath}/loss_vssmd.jpg")


main_vssm()
