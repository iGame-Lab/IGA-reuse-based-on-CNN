import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

from model.UNet import UNet
from model.UNet3Plus import UNet3Plus, UNet3PlusISSA
from trainTest.loadDataset import loadOneData
from loss.myLossFunc import TestPlot

input_size = 128

def Plot (args):
    
    print(args.data_select + args.equation_select + " ploting data" + str(args.test_index) + "...")

    if args.net_select == "unet":
        model = UNet()
    elif args.net_select == "unet3plus":
        model = UNet3Plus()
    elif args.net_select == "unet3plusISSA":
        model = UNet3PlusISSA()
    model.to(args.device)
    params = torch.load(args.model_path)["state_dict"]
    model.load_state_dict(params, False)

    test_data = loadOneData(args.data_select, args.equation_select, mode="test", index=args.test_index)

    plot_func = TestPlot(args.device, test_data["patch_num"], test_data["order_vector"], test_data["knot_vector"], test_data["ctrlp_num"], args.sample_num)
   
    dir = os.path.join(args.writer_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    model.eval()
    
    input, target, mapRecord = test_data["inputs"], test_data["output"], test_data["mapRecord"]
    input = torch.tensor(input, dtype=torch.float32, device=args.device)
    target = torch.tensor(target, dtype=torch.float32, device=args.device)
    mapRecord = torch.tensor(mapRecord, dtype=torch.long, device=args.device)

    pred = model(input)

    np.savetxt(dir + "/pred_" + str(args.test_index) + ".txt", pred[0][0].cpu().detach().numpy())

    true_sol, pred_sol, _, _ = plot_func(mapRecord, target, pred)

    true_sol = true_sol[0][0].cpu().detach().numpy().tolist()
    pred_sol = pred_sol[0][0].cpu().detach().numpy().tolist()

    plt.figure()
    num = 20
    minmin = min([min(true_sol), min(true_sol)])
    maxmax = max([max(pred_sol), max(pred_sol)])
    true_sol.append(minmin)
    true_sol.append(maxmax)
    pred_sol.append(minmin)
    pred_sol.append(maxmax)
    weights_true = np.ones_like(true_sol) / float(len(true_sol))
    true_n, true_bins, _ = plt.hist(true_sol, bins=num, weights=weights_true, histtype='step')
    weights_net = np.ones_like(pred_sol) / float(len(pred_sol))
    pred_n, pred_bins, _ = plt.hist(pred_sol, bins=num, weights=weights_net, histtype='step')

    plt.figure()
    true = []
    for i in range(0, num):
        true.append((true_bins[i] + true_bins[i + 1]) / 2.0)
    plt.plot(true, true_n, lw=1, label="ground truth")
    pred = []
    for i in range(0, num):
        pred.append((pred_bins[i] + pred_bins[i + 1]) / 2.0)
    plt.plot(pred, pred_n, lw=1, label="Prediction")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Solution", fontsize=16)
    plt.ylabel("Statistic", fontsize=16)

    plt.savefig(dir + "/pred_" + str(args.test_index) + ".png")
