from email.policy import default
import torch
import argparse

from trainTest.Train import Train
from trainTest.Test import Test
from trainTest.Plot import Plot

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', help="Choosing which python file to run")
    parser.add_argument('--net_select', default='unet3plusISSA', help="Chossing network")
    parser.add_argument('--w1', type=float, default=1., help="The weigth of coefficients loss of total loss")
    parser.add_argument('--w2', type=float, default=1., help="The weigth of numerical solution loss of total loss")
    parser.add_argument('--sample_num', type=int, default=10, help="Sample number of numerical solution on each direction of each patch")
    parser.add_argument('--learning_rate', type=float, default=0.0006)
    parser.add_argument('--model_path', default='../myresults/models/test.pth', help="The path of saving model")
    parser.add_argument('--writer_path', default='../myresults/preds/test.txt', help="The path of saving logging, testing errors and predications")
    # parser.add_argument('--runs_dir', default='../results/runs/test')
    parser.add_argument('--iterations', type=int, default=40000, help="The number of iterations for network training")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_select', default='hole', help="Selecting geometry, flower, hole or human")
    parser.add_argument('--equation_select', default='f1', help="Selecting right-hand side function of poisson equation, f1 or f2")
    parser.add_argument('--pde_type', type=int, default=1, help="The type of right-hand side function, 1 corresponds to f1 and 2 corresponds to f2")
    parser.add_argument('--size', type=int, default=128, help="The input size of network")
    parser.add_argument('--test_index', type=int, default=0, help="The test index of plot mode")
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        Train(args)
    elif args.mode == "test":
        Test(args)
    elif args.mode == "plot":
        Plot(args)