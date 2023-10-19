import torch_geometric as pyg
import torch
from torch_geometric.datasets import MoleculeNet
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import one_hot, scatter
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.nn.conv import GATv2Conv, GATConv, TransformerConv
from torch_geometric.nn.models import MLP
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, HybridizationType
import os
import matplotlib.pyplot as plt

dataset = MoleculeNet(root="Lipo", name="Lipo")
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.linear1 = nn.Linear(16,1)
        self.out = nn.Linear(32, 1)
        #self.conv3 = GCNConv(32, dataset.num_classes) #num_classes:ラベルの数
    #バッチノルム(正則化)
    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        # Dropout:一定割合のノードを不活性化(0になる)させ、過学習を緩和する。pはゼロになるノードの確率で、0.5がデフォルト。
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_add_pool(x, batch) #これが必要やった
        #x = F.dropout(x, p=0.2, training=self.training) # 取ってみる
        x = self.out(x)
        return x

class GCN_N(torch.nn.Module):
    def __init__(self, layer:int, dim=32, dataset=dataset):
        super().__init__()
        self.layer = layer
        self.dataset = dataset
        self.dim = dim
        self.conv1 = GCNConv(self.dataset.num_node_features, self.dim, improved=True)
        self.convn = GCNConv(self.dim, self.dim, improved=True)
        self.out = pyg.nn.Linear(self.dim, 1)

    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for i in range(2, self.layer + 1):
            x = self.convn(x, edge_index)
            x = F.relu(x)
        x = pyg.nn.global_add_pool(x, batch) 
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.out(x)
        return x

class GATv2_N(torch.nn.Module):
    def __init__(self, layer:int, dim=32):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.conv1 = GATv2Conv(dataset.num_node_features, self.dim)
        self.convn = GATv2Conv(self.dim, self.dim)
        self.out = pyg.nn.Linear(self.dim, 1)

    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        for i in range(2, self.layer + 1):
            x = self.convn(x, edge_index, edge_attr)
            x = F.relu(x)
        x = pyg.nn.global_add_pool(x, batch) 
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.out(x)
        return x

class GAT_N(torch.nn.Module):
    def __init__(self, layer:int, dim=32):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.conv1 = GATConv(dataset.num_node_features, self.dim)
        self.convn = GATConv(self.dim, self.dim)
        self.out = pyg.nn.Linear(self.dim, 1)

    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        for i in range(2, self.layer + 1):
            x = self.convn(x, edge_index, edge_attr)
            x = F.relu(x)
        x = pyg.nn.global_add_pool(x, batch) 
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.out(x)
        return x

class trans_N(torch.nn.Module):
    def __init__(self, layer:int, dim=32):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.conv1 = GCNConv(dataset.num_node_features, self.dim, improved=True)
        self.convn = GCNConv(self.dim, self.dim, improved=True)
        self.out = pyg.nn.Linear(self.dim, 1)

    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr[0])
        x = F.relu(x)
        for i in range(2, self.layer + 1):
            x = self.convn(x, edge_index, edge_attr[0])
            x = F.relu(x)
        x = pyg.nn.global_add_pool(x, batch) 
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.out(x)
        return x

def split(model, layer, dim):
    # データの分割(total: 130831)
    num_train, num_val = int(len(dataset)*0.6), int(len(dataset)*0.2)
    num_test = len(dataset) - (num_train + num_val)
    batch_size = 32

    # 乱数の固定
    device = torch.device("cpu")
    seed = 0
    pyg.seed_everything(seed=seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    """
    train_set, valid_set, test_set = random_split(dataset, [num_train, num_val, num_test])

    #Dataloaderの生成
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=pyg.seed_everything(seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, worker_init_fn=pyg.seed_everything(seed))
    test_loader = DataLoader(test_set, batch_size=batch_size, worker_init_fn=pyg.seed_everything(seed))

    model = model(layer=layer,dim=dim)
    # 損失関数
    criterion = F.mse_loss
    # Optimizerの初期化
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

def train(max_epoch):
    for epoch in range(max_epoch):
    # train
        model.train()
        train_loss = 0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to("cpu")
            optimizer.zero_grad()
            prediction = model(batch)
            loss = criterion(prediction, batch.y[:, target_idx].unsqueeze(1))
            loss.backward()
            train_loss += loss.item()
            total_graphs += batch.num_graphs
            optimizer.step()
        train_loss /=  len(train_loader) # 損失の平均(batchあたり) #平均を取ってからルート
        train_loss = sqrt(train_loss)
    
    # validation
        model.eval()
        valid_loss = 0
        total_graphs = 0
        with torch.inference_mode(): # 自動微分無効。torch.no_grad()よりさらに高速化
            for batch in valid_loader:
                prediction = model(batch)
                #loss = torch.sqrt(criterion(prediction, batch.y[:, target_idx].unsqueeze(1)))
                loss = criterion(prediction, batch.y[:, target_idx].unsqueeze(1))
                valid_loss += loss.item()
                total_graphs += batch.num_graphs
        valid_loss /= len(valid_loader)
        valid_loss = sqrt(valid_loss)

        print(f"Epoch {epoch+1} | train_loss:{train_loss}, valid_loss:{valid_loss}")
