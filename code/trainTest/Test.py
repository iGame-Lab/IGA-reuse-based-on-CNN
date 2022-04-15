import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from model.UNet import UNet
from model.UNet3Plus import UNet3Plus, UNet3PlusISSA
from trainTest.loadDataset import loadDataset
from loss.myLossFunc import TestLoss
from loss.PDEError import PDETestError

def Test(args):

    print(args.data_select + args.equation_select + " testing...")

    if args.net_select == "unet":
        model = UNet()
    elif args.net_select == "unet3plus":
        model = UNet3Plus()
    elif args.net_select == "unet3plusISSA":
        model = UNet3PlusISSA()
    model.to(args.device)
    params = torch.load(args.model_path)["state_dict"]
    model.load_state_dict(params, False)

    test_dataset = loadDataset(args.data_select, args.equation_select, mode="test", size=args.size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    test_len = len(test_loader)
    
    test_error = TestLoss(args.data_select, args.device, test_dataset.patch_num, test_dataset.order_vector, test_dataset.knot_vector, test_dataset.ctrlp_num, test_dataset.boundary, args.sample_num)
    test_pde_error = PDETestError(args.device, test_dataset.patch_num, test_dataset.order_vector, test_dataset.knot_vector, test_dataset.ctrlp_num, test_dataset.boundary, args.sample_num, args.pde_type)
    
    test_coefs_loss = 0.
    test_sol_loss = 0.
    test_sol_relate_loss = 0.
    test_pde_loss = 0.

    model.eval()
    for i, vailationData in enumerate(test_loader, 0):
        input, target, mapRecord = vailationData
        input, target, mapRecord = input.to(args.device), target.to(args.device), mapRecord.to(args.device)
        pred = model(input)

        loss1, loss2, loss3 = test_error(mapRecord, target, pred)
        loss4 = test_pde_error(input, mapRecord, pred)
        test_coefs_loss += loss1.item()
        test_sol_loss += loss2.item()
        test_sol_relate_loss += loss3.item()
        test_pde_loss += loss4.item()

    test_results = {}
    test_results['test_coefs_loss'] = test_coefs_loss / test_len
    test_results['test_sol_loss'] = test_sol_loss / test_len
    test_results['test_sol_relate_loss'] = test_sol_relate_loss / test_len
    test_results['test_pde_loss'] = test_pde_loss / test_len
    test_results = str(test_results)
    file = open(args.writer_path,'w')
    file.writelines(test_results)
    file.close()
