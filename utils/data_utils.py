import constant
import torch

class InverseFoldDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_paths):
        self.data_file_paths = data_file_paths
        self.c1_cords = []
        self.fastas = []
        self.tokens = []
        self.pe = []
        self.transfer_all()
        
    def preprocess(self, data):
        c1_cords = []
        fastas = []
        chaintokens = []
        token = 1
        for item in data:
            c1_cords.append(item['C1_cords'])
            tmp = list(item['fasta'])
            for i in range(len(tmp)):
                if tmp[i] in ['DA', 'DC', 'DG', 'DT']:
                    tmp[i] = tmp[i][1]
                if tmp[i] not in ['A', 'C', 'G', 'T', 'U']:
                    tmp[i] = 'N'
                if tmp[i] == 'U':
                    tmp[i] = 'T'
            fastas.append(tmp)
            chaintokens += [token] * len(tmp)
            token += 1
        try:
            c1_cords = np.concatenate(c1_cords,axis=0)
            # fastas = np.array(fastas)
            # tokens = np.array(chaintokens)
            tokens = chaintokens
            tmp = []
            for fasta in fastas:
                tmp += fasta
            fastas = tmp
        except:
            return None, None, None
        if len(fastas) != len(tokens):
            print(fastas,tokens)
            raise ValueError('length not equal')
        return c1_cords, fastas, tokens
    
    def transfer_all(self):
        for data_file_path in self.data_file_paths:
            data = np.load(data_file_path, allow_pickle=True)
            c1_cords, fastas, tokens = self.preprocess(data)
            if c1_cords is None:
                continue
            if c1_cords.shape[0] == 0:
                continue
            if c1_cords.shape[0] > 512:
                continue
            self.c1_cords.append(c1_cords)
            self.fastas.append(fastas)
            self.tokens.append(tokens)
    
    def __len__(self):
        return len(self.fastas)

    def __getitem__(self, idx):
        return self.c1_cords[idx], self.fastas[idx], self.tokens[idx]

def collet_fn_inversefold(batch):
    c1_cords = [i[0] for i in batch]
    fastas = [i[1] for i in batch]
    tokens = [i[2] for i in batch]
    max_length = max([len(i) for i in fastas])
    fastas = [i + ['N'] * (max_length - len(i)) for i in fastas]
    fastas = torch.tensor([[constant.tokenonthot[j] for j in i] for i in fastas])
    tokens = [i + [0] * (max_length - len(i)) for i in tokens]
    tokens = torch.tensor([[int(j) for j in i] for i in tokens])
    mask = fastas != 0
    c1_cords = [torch.nn.functional.pad(torch.tensor(i,dtype=torch.float32),(0,0,0,max_length - len(i))) for i in c1_cords]
    c1_cords = torch.stack(c1_cords)
    return c1_cords, fastas, tokens, mask

class RDesignDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.coords = []
        self.fastas = []
        self.tokens = []
        self.transfer_all()
        
    def preprocess(self, item):
        keys = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
        coords = []
        for key in keys:
            coords.append(np.array(item['coords'][key]))
        tmp = list(item['seq'])
        for i in range(len(tmp)):
            if tmp[i] in ['DA', 'DC', 'DG', 'DT']:
                tmp[i] = tmp[i][1]
            if tmp[i] not in ['A', 'C', 'G', 'T', 'U']:
                tmp[i] = 'N'
            if tmp[i] == 'U':
                tmp[i] = 'T'
        fastas = tmp
        tokens = np.array(item["chain_idxs"]) + 1
        coords = np.concatenate(coords,axis=-1)
        return coords, fastas, tokens
    
    def transfer_all(self):
        for data in np.load(self.data_file_path, allow_pickle=True):
            coords, fastas, tokens = self.preprocess(data)
            if coords is None:
                continue
            if coords.shape[0] == 0:
                continue
            if coords.shape[0] > 512:
                continue
            self.coords.append(coords)
            self.fastas.append(fastas)
            self.tokens.append(tokens)

    def __len__(self):
        return len(self.fastas)

    def __getitem__(self, idx):
        return self.coords[idx], self.fastas[idx], self.tokens[idx]

def collet_fn_inversefold_rdesign(batch):
    cords = [i[0] for i in batch]
    fastas = [i[1] for i in batch]
    tokens = [i[2] for i in batch]
    max_length = max([len(i) for i in fastas])
    fastas = [i + ['N'] * (max_length - len(i)) for i in fastas]
    fastas = torch.tensor([[constant.tokenonthot[j] for j in i] for i in fastas])
    tokens = [list(i) + [0] * (max_length - len(list(i))) for i in tokens]
    tokens = torch.tensor([[int(j) for j in i] for i in tokens])
    mask = fastas != 0
    cords = [torch.nn.functional.pad(torch.tensor(i,dtype=torch.float32),(0,0,0,max_length - len(i))) for i in cords]
    cords = torch.stack(cords)
    b, s, d = cords.shape
    cords = cords.reshape(b, s, 6, 3)
    return cords, fastas, tokens, mask

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self,fasta,seq_len=100):
        self.fasta = fasta
        self.seq_len = seq_len
        self.size = 0
        for i in range(len(self.fasta)):
            self.size += len(self.fasta[i])
        
    def __len__(self):
        return len(self.fasta)
    
    def __getitem__(self, idx):
        seq = self.fasta[idx]
        seq = [constant.tokenonthot[i] for i in ['<CLS>'] + list(seq)]
        seq = torch.tensor(seq,dtype=torch.int8)
        mask = torch.ones(len(seq),dtype=torch.bool)
        return seq, mask

def collect_fn_pretrain(batch):
    seqs = [i[0] for i in batch]
    masks = [i[1] for i in batch]
    max_len = max([len(i) for i in seqs])
    seqs = [torch.nn.functional.pad(i,(0,max_len-len(i))) for i in seqs]
    masks = [torch.nn.functional.pad(i,(0,max_len-len(i))) for i in masks]
    seq = torch.stack(seqs)
    mask = torch.stack(masks).bool()
    return seq, mask

class UTRdataset(torch.utils.data.Dataset):
    
    def __init__(self,df,seq_len=100):
        self.df = df
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['Sequence (with flanking regions)']
        seq = [constant.tokenonthot[i] for i in ['<CLS>'] + list(seq)]
        seq = torch.tensor(seq,dtype=torch.int8)
        TE = torch.tensor(row['TE'],dtype=torch.float32)
        celltype = row['Cell tyes']
        TE = torch.log(TE+1e-6)
        mask = torch.ones(len(seq),dtype=torch.bool)
        return seq,TE,celltype,mask

def collect_fn_utr(batch):
    seqs = [i[0] for i in batch]
    tes = [i[1] for i in batch]
    celltypes = [i[2] for i in batch]
    masks = [i[3] for i in batch]
    max_len = max([len(i) for i in seqs])
    seqs = [torch.nn.functional.pad(i,(0,max_len-len(i))) for i in seqs]
    masks = [torch.nn.functional.pad(i,(0,max_len-len(i))) for i in masks]
    seq = torch.stack(seqs)
    tes = torch.stack(tes)
    mask = torch.stack(masks).bool()
    return seq, tes, celltypes, mask
