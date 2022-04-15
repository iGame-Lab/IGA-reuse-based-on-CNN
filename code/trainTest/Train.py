import torch
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import sys
import logging
import math

from model.UNet import UNet
from model.UNet3Plus import UNet3Plus, UNet3PlusISSA
from trainTest.loadDataset import loadDataset
from loss.myLossFunc import TotalLoss, TestLoss
from loss.PDEError import PDETestError

def Train(args):

    print(args.data_select + args.equation_select + " training...")
    
    sys.stdout.flush()

    # writer = SummaryWriter(args.runs_dir)

    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger(args.data_select + args.equation_select)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s' % (args.writer_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    if args.net_select == "unet":
        model = UNet()
    elif args.net_select == "unet3plus":
        model = UNet3Plus()
    elif args.net_select == "unet3plusISSA":
        model = UNet3PlusISSA()
    model.to(args.device)

    train_dataset = loadDataset(args.data_select, args.equation_select, mode="train", size=args.size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_dataset = loadDataset(args.data_select, args.equation_select, mode="validation", size=args.size)
    validation_loader = DataLoader(validation_dataset, batch_size=1)
    train_len = len(train_loader)
    validation_len = len(validation_loader)

    train_loss_func = TotalLoss(args.device, train_dataset.patch_num, train_dataset.order_vector, train_dataset.knot_vector, train_dataset.ctrlp_num, args.w1, args.w2, args.sample_num)
    validation_error = TestLoss(args.data_select, args.device, validation_dataset.patch_num, validation_dataset.order_vector, validation_dataset.knot_vector, validation_dataset.ctrlp_num, validation_dataset.boundary, args.sample_num)
    validation_pde_error = PDETestError(args.device, validation_dataset.patch_num, validation_dataset.order_vector, validation_dataset.knot_vector, validation_dataset.ctrlp_num, validation_dataset.boundary, args.sample_num, args.pde_type)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0)
    epochs = math.ceil(args.iterations / train_len )
    log_string('epochs: %d' % epochs)

    for epoch in range(0, epochs):

        model.train()
        for i, traindata in enumerate(train_loader, 0):
            input, target, mapRecord = traindata
            input, target, mapRecord = input.to(args.device), target.to(args.device), mapRecord.to(args.device)
            model.zero_grad()
            output = model(input)
            loss, loss1, loss2 = train_loss_func(input, mapRecord, target, output)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                # writer.add_scalar("train_coefs_loss", loss1.item(), epoch * train_len + i)
                # writer.add_scalar("train_sol_loss", loss2.item(), epoch * train_len + i)

                log_string('epoch: %d iter: %d train_total_loss: %.4f train_coefs_loss: %.4f train_sol_loss: %.4f' % (epoch, i, loss, loss1, loss2))

        model.eval()
        validation_coefs_loss = 0.
        validation_sol_loss = 0.
        validation_sol_relate_loss = 0.
        validation_pde_loss = 0.
        for _, vailationData in enumerate(validation_loader, 0):
            input, target, mapRecord = vailationData
            input, target, mapRecord = input.to(args.device), target.to(args.device), mapRecord.to(args.device)
            pred = model(input)
            loss1, loss2, loss3 = validation_error(mapRecord, target, pred)
            loss4 = validation_pde_error(input, mapRecord, pred)
            validation_coefs_loss += loss1.item()
            validation_sol_loss += loss2.item()
            validation_sol_relate_loss += loss3.item()
            validation_pde_loss += loss4.item()

        # writer.add_scalar("validation_coefs_loss", validation_coefs_loss / validation_len, epoch)
        # writer.add_scalar("validation_sol_loss", validation_sol_loss / validation_len, epoch)
        # writer.add_scalar("validation_sol_relate_loss", validation_sol_relate_loss / validation_len, epoch)
        # writer.add_scalar("validation_pde_loss", validation_pde_loss / validation_len, epoch)

        log_string('epoch: %d  validation_coefs_loss: %.4f  validation_sol_loss: %.4f  validation_sol_relate_loss: %.4f validation_pde_loss: %.4f'
                   % (epoch, validation_coefs_loss / validation_len, validation_sol_loss / validation_len,  
                   validation_sol_relate_loss / validation_len, validation_pde_loss / validation_len))

    # writer.close()

    torch.save({'epoch': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()},
                args.model_path)
