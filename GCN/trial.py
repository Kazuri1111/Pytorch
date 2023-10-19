# Create Dataset(For understanding)
# reference source code 
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html
# neural fingerprint on pytorch ref
# https://qiita.com/kimisyo/items/55a01e27aa03852d84e9


#RDLogger.DisableLog("rdApp.*")

# units conversion
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

class MyFirstDataset(pyg.data.Dataset):
    def __init__(self, root="./MyFirstDataset", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["qm9.pt"]

    def process(self):
        import rdkit
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType
        from rdkit.Chem.rdchem import HybridizationType
  
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
        
        # csv読み込み
        self.df = pd.read_csv("qm9_dataset.csv")
        # 列の並べ替え(reindex)
        #self.data = self.data.reindex(index=["mol_id", "smiles", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom", "A", "B", "C"])
        
        data_list = []
        idx = 0
        # data生成
        target = self.df["alpha"].values #分極率のみ
        target = torch.tensor(target, dtype=torch.long)
        for index, mol in self.df.iterrows():
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            mol_obj = Chem.AddHs(mol_obj)
            N = mol_obj.GetNumAtoms() # 原子数
            #AllChem.EmbedMultipleConfs(mol_obj)
            #conf = mol_obj.GetConformer() # コンフォーマー生成
            #AllChem.MMFFOptimizeMoleculeConfs(mol_obj) #MMFF最適化
            #pos = conf.GetPositions() # 各原子の位置
            #pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            
            for atom in mol_obj.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                if atom.GetIsAromatic():
                    aromatic.append(1)
                else:
                    aromatic.append(0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            z = torch.tensor(atomic_number, dtype=torch.long) #原子番号
            
            # edge index
            row, col, edge_type = [], [], []
            for bond in mol_obj.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]
            
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long) #結合次数
            edge_attr = one_hot(edge_type, num_classes=len(bonds)) #one hot vector: vector that has 0 or 1 for each value

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()
            # types: qm9に含まれる原子の一覧(H,C,N,O,F)。num_classesは原子の種類の数
            # x1:原子記号のリスト
            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            
            # node features: 原子番号、芳香性、混成の有無(sp,sp2,sp3)、水素の数
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            # x1,x2を結合
            x = torch.cat([x1, x2], dim=-1)
            #グラフ特徴量
            y = target.unsqueeze(0)

            # create dataset
            data = Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y,  idx=idx)
            idx = idx + 1
            data_list.append(data)
        torch.save(data_list, os.path.join())

    def len(self):
        return len(self.processed_file_names)
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data