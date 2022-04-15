import torch
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
from os import listdir, path

BASE_DIR = path.abspath(path.join(path.dirname(__file__), "../.."))

class loadDataset(Dataset):
    def __init__(self, data_select="human", equation_select="f1", mode="train", size=128, input_channel=3, output_channel=1):
        super(loadDataset, self).__init__()

        info_file = open(path.join(BASE_DIR, "data", data_select, "info.txt"))
        info_list = []
        for line in info_file:
            info_list.append(line.split())
        self.patch_num = int(info_list[0][0])
        self.order_vector = np.array(info_list[1:self.patch_num + 1]).astype(np.int32)
        self.ctrlp_num = np.array(info_list[self.patch_num + 1:self.patch_num * 2  + 1]).astype(np.int32)
        self.knot_vector = np.array(info_list[self.patch_num * 2 + 1:self.patch_num * 4 + 1]).reshape(self.patch_num, 2, -1).astype(np.float32)
        self.boundary = np.array(info_list[self.patch_num * 4 + 1:self.patch_num * 5 + 1]).astype(np.int32)

        data_dir = path.join(BASE_DIR, "data", data_select, "poisson" + equation_select, mode)
        dataFiles = listdir(data_dir)
        filesLen = len(dataFiles)
        self.input = np.empty((filesLen, input_channel, size, size))
        if data_select == "human":
            self.mapRecord = np.empty((filesLen, 375, 4))
        elif data_select == "hole":
            self.mapRecord = np.empty((filesLen, 484, 4))
        elif data_select == "flower":
            self.mapRecord = np.empty((filesLen, 605, 4))
        self.target = np.empty((filesLen, output_channel, size, size))
        for i in range(filesLen):
            tempData = np.load(data_dir + "/" + dataFiles[i])
            self.input[i] = tempData["inputs"]
            self.target[i] = tempData["output"]
            self.mapRecord[i] = tempData["mapRecord"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return torch.tensor(self.input[idx], dtype=torch.float32, requires_grad=True).clone(), \
               torch.tensor(self.target[idx], dtype=torch.float32, requires_grad=True).clone(), \
               torch.tensor(self.mapRecord[idx], dtype=torch.long).clone()


def loadOneData (data_select="human", equation_select="f1", mode="test", index=0):
        info_file = open(path.join(BASE_DIR, "data", data_select, "info.txt"))
        info_list = []
        for line in info_file:
            info_list.append(line.split())
        testData = {}
        patch_num = int(info_list[0][0])
        testData["patch_num"] = int(info_list[0][0])
        testData["order_vector"] = np.array(info_list[1:patch_num + 1]).astype(np.int32)
        testData["ctrlp_num"] = np.array(info_list[patch_num + 1:patch_num * 2  + 1]).astype(np.int32)
        testData["knot_vector"] = np.array(info_list[patch_num * 2 + 1:patch_num * 4 + 1]).reshape(patch_num, 2, -1).astype(np.float32)
        testData["boundary"] = np.array(info_list[patch_num * 4 + 1:patch_num * 5 + 1]).astype(np.int32)

        data_dir = path.join(BASE_DIR, "data", data_select, "poisson" + equation_select, mode)
        
        tempData = np.load(data_dir + "/data" + str(index) + ".npz")
        testData["inputs"] = np.array([tempData["inputs"]])
        testData["output"] = np.array([tempData["output"]])
        testData["mapRecord"] = np.array([tempData["mapRecord"]])

        return testData


