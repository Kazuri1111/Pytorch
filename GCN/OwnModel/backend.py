import pandas as pd
import pickle
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import rdkit.Chem.AllChem as AllChem
import torch
import torch_geometric as pyg
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import one_hot, scatter
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import QM9
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.models import MLP
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import one_hot, scatter
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import time
import pandas as pd
import os
import os.path as osp
import pickle
import sys
import shutil
from typing import Callable, List, Optional
import tqdm
from math import sqrt as sqrt
import backend
from torchmetrics.regression import R2Score, MeanSquaredError
from sklearn.preprocessing import StandardScaler #標準化(P.116-)

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        #self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.dataset = dataset
        self.conv1 = GCNConv(self.dataset.num_node_features, 32)
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
    def __init__(self, dataset, layer=3, dim=64):
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

class GCN3(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN3, self).__init__()
        hidden_layer = 64
        self.conv1 =  GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.out = nn.Linear(64, 1)
    
    def forward(self, data):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.out(x)
        return x

def seed_worker(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702

    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

class MyQM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_reduce=None):
        super().__init__(root, transform, pre_transform, pre_reduce)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.pre_reduce = pre_reduce

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return "data_v3.pt"
    
    def download(self):
        pass
    
    def process(self):
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        #回帰ターゲット
        df = pd.read_csv("qm9_dataset.csv")
        df_target = df.reindex(columns=["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom", "A", "B", "C"])
        target = torch.tensor([list(i[1:]) for i in df_target.itertuples()], dtype=torch.float)
        self.target = target

        with open("./uncharacterized.txt") as f:
            #計算できんかったやつ
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        
        smiles = df["smiles"].tolist()
        mols = [Chem.MolFromSmiles(m) for m in smiles]
        data_list = []
        for i, mol in enumerate(tqdm.tqdm(mols)):
            if i in skip: #計算できんかったやつを飛ばす
                continue

            mol = Chem.AddHs(mol)
        
            N = mol.GetNumAtoms() #分子の原子数
            
            conf = mol.GetConformers()

            type_idx = []
            atomic_number = []
            formal_charge = []
            valence = []
            degree = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []

            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                formal_charge.append(atom.GetFormalCharge())
                valence.append(atom.GetTotalValence())
                degree.append(atom.GetTotalDegree())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            
            #x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            #x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                            #dtype=torch.float).t().contiguous()
            desc_dict = {
            "atomic_number":atomic_number,
            "formal_charge":formal_charge,
            "valence":valence,
            "degree":degree,
            "aromatic":aromatic,
            "sp":sp,
            "sp2":sp2,
            "sp3":sp3,
            "num_hs":num_hs
            }
            descriptors_in_use = [atomic_number, formal_charge, valence, degree, aromatic, sp, sp2, sp3, num_hs]

            if pre_reduce:
                descriptors_in_use.remove(desc_dict[pre_reduce])
            # 標準化
            #descriptors_in_use = stdscaler.fit_transform(descriptors_in_use) 間違い
            x = torch.tensor(descriptors_in_use, dtype=torch.float).t().contiguous()
            #x = torch.cat([x1, x2], dim=-1)
            y = target[i].unsqueeze(0)
            smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)
            data = Data(x=x, edge_index=edge_index, smiles=smiles, edge_attr=edge_attr, y=y, idx=i)
            data_list.append(data)
            

        torch.save(self.collate(data_list), self.processed_paths[0])

# https://discuss.pytorch.org/t/rmse-loss-function/16540/3
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps
    
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_complicated(dataset, model, epoch_num, train_loader, valid_loader, optimizer, target_idx, initial_epoch_num=None, early_stopping=None):
    device = "cuda"
    #rmse = RMSELoss()
    mae = nn.L1Loss(reduction="sum")
    
    # torchmetrics
    rmse = MeanSquaredError(squared=False).to("cuda")
    calc_R2 = R2Score().to("cuda") # To avoid memory error
    #early_stopping = EarlyStopping(patience=1, verbose=True, path=f"{pre_reduce}_best.pt")
    results = []
    for epoch in range(epoch_num):
        if initial_epoch_num > 0:
            epoch += initial_epoch_num
        # train
        model.train()
        train_loss = 0
        total_graphs = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            label = batch.y[:, target_idx].unsqueeze(1)
            loss = rmse(prediction, label)
            #loss = mae(prediction, label)
            loss.backward()
            train_loss += loss.item()
            total_graphs += batch.num_graphs
            optimizer.step()

            with torch.no_grad(): # Required to avoid out of memory error
                try:
                    train_label = torch.cat((train_label, label), dim=0)
                except NameError:
                    train_label = label
                try:
                    train_prediction = torch.cat((train_prediction, prediction), dim=0)
                except NameError:
                    train_prediction = prediction
        train_loss = train_loss / total_graphs #損失の平均(batchあたり) ルートを取ってから平均
        train_R2 = calc_R2(train_label, train_prediction).item()
        del train_label, train_prediction, label, prediction

        # validation
        model.eval()
        valid_loss = 0
        total_graphs = 0
        for batch in valid_loader:
            batch.to(device)
            prediction = model(batch)
            label = batch.y[:, target_idx].unsqueeze(1)
            loss = rmse(prediction, label)
            #loss = mae(prediction, label)
            valid_loss += loss.item()
            total_graphs += batch.num_graphs
            with torch.no_grad(): # Required to avoid out of memory error
                try:
                    valid_label = torch.cat((valid_label, label), dim=0)
                except NameError:
                    valid_label = label
                try:
                    valid_prediction = torch.cat((valid_prediction, prediction), dim=0)
                except NameError:
                    valid_prediction = prediction
            
        valid_loss = valid_loss / total_graphs #損失の平均(batchあたり) ルートを取ってから平均
        valid_R2 = calc_R2(valid_label, valid_prediction).item()
        del valid_label, valid_prediction, label, prediction

        print(f"Epoch {epoch+1} | train_loss:{train_loss}, valid_loss:{valid_loss}, train_R2:{train_R2}, valid_R2:{valid_R2}")
        results.append({"Epoch":epoch+1, "train_loss":train_loss, "valid_loss":valid_loss, "train_R2":train_R2, "valid_R2":valid_R2})
    return results

def train(dataset, model, epoch_num, train_loader, valid_loader, optimizer, target_idx):
    device = "cuda"

    mae = nn.L1Loss(reduction="mean")
    rmse = MeanSquaredError(squared=False).to("cuda")
    calc_R2 = R2Score().to("cuda") # To avoid memory error

    results = []

    stdsc = StandardScaler()
    r2Score = R2Score().to("cuda")
    for epoch in range(epoch_num):
        # train
        model.train()
        train_loss = 0
        train_R2 = 0
        total_graphs = 0
        for batch in train_loader:
            batch.x = stdsc.fit_transform(batch.x)
            batch.x = torch.from_numpy(batch.x.astype(np.float32)).clone()
            batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            label = batch.y[:, target_idx].unsqueeze(1)
            loss = rmse(prediction, label)
            R2 = r2Score(prediction, label)
            train_R2 += R2.item()
            loss.backward()
            train_loss += loss.item()
            total_graphs += batch.num_graphs
            optimizer.step()
        train_loss = train_loss / len(train_loader) #損失の平均(batchあたり)
        train_R2 = train_R2 / len(train_loader)

        # validation
        model.eval()
        valid_loss = 0
        valid_R2 = 0
        total_graphs = 0
        for batch in valid_loader:
            batch.x = stdsc.fit_transform(batch.x)
            batch.x = torch.from_numpy(batch.x.astype(np.float32)).clone()
            batch.to(device)
            prediction = model(batch)
            label = batch.y[:, target_idx].unsqueeze(1)
            loss = rmse(prediction, label)
            R2 = r2Score(prediction, label)
            valid_loss += loss.item()
            valid_R2 += R2.item()
            total_graphs += batch.num_graphs
        valid_loss = valid_loss / len(valid_loader) #損失の平均(batchあたり)
        valid_R2 = valid_R2 / len(valid_loader)
        print(f"Epoch {epoch+1} | train_loss:{train_loss}, valid_loss:{valid_loss}, train_R2:{train_R2}, valid_R2:{valid_R2}")
        results.append({"Epoch":epoch+1, "train_loss":train_loss, "valid_loss":valid_loss, "train_R2":train_R2, "valid_R2":valid_R2})
    return results
