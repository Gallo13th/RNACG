import torch
import tqdm
import torch.optim as optim
import torch.utils
from modules import sequence_flow
from utils import flow_utils,constant
import numpy as np
from sklearn.metrics import f1_score
from models.trainer import Trainer
import pandas as pd
import RNA
def cal_ss(seq):
    return RNA.fold(seq)[0]
'''
    A    T/U  C    G
A   0    1    0    0
T/U 0    0    1    1
C   0    0    0    1
G   1    1    0    0

'''
NT_PAIR_MATRIX = torch.tensor(
    [
        [0,1,0,0],
        [0,0,1,1],
        [0,0,0,1],
        [1,1,0,0]
    ]
)

def post_process(predprob,pair_edges):
    pred1hot = torch.argmax(predprob,dim=-1)
    for start,end in pair_edges:
        prob = torch.einsum('i,j->ij',predprob[start],predprob[end]) * NT_PAIR_MATRIX.to(predprob.device)
        maxidx = prob.argmax()
        start_idx = maxidx // 4
        end_idx = maxidx % 4
        pred1hot[start] = start_idx
        pred1hot[end] = end_idx
    return pred1hot


def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

def top_k_accuracy(preds,labels,k=4):
    topk = torch.topk(preds,k=k,dim=-1).indices
    labels = labels.unsqueeze(-1).expand_as(topk)
    return (topk == labels).any(dim=-1).float().mean()
    
class InverseFoldDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_paths,target_rfam='RF00005'):
        self.data_file_paths = data_file_paths
        self.target_rfam = target_rfam
        self.edges = []
        self.seqs = []
        self.rfam_idx = {}
        self.transfer_all()
        
    def preprocess(self, data):
        seqs = []
        rfam_idx = []
        sses = []
        for idx, item in data[data['rfam_acc'] == self.target_rfam].iterrows():
            ss = item['ss']
            rfam_acc = item['rfam_acc']
            if rfam_acc not in self.rfam_idx:
                self.rfam_idx[rfam_acc] = len(self.rfam_idx)
            item['seq'] = item['seq'].replace('U','T')
            tmp = list(item['seq'])
            seqs.append(tmp)
            rfam_idx.append(self.rfam_idx[rfam_acc])
            sses.append(ss)
        return seqs, rfam_idx, sses
    
    def transfer_all(self):
        data = pd.read_csv(self.data_file_paths)
        seqs, rfam_idx,sses = self.preprocess(data)
        for seq in seqs:
            if len(seq) > 512:
                continue
            self.seqs.append(seq)
        self.sorted_idx = np.argsort([len(i) for i in self.seqs])
        self.seqs = [self.seqs[i] for i in self.sorted_idx]
        self.cls_idx = [rfam_idx[i] for i in self.sorted_idx]
        self.sses = [sses[i] for i in self.sorted_idx]
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.cls_idx[idx], self.sses[idx]

def collet_fn_rfamflow(batch):
    seqs = [i[0] for i in batch]
    cls_idx = [i[1] for i in batch]
    ss = [i[2] for i in batch]
    length = [len(i) for i in seqs]
    seq_t = []
    max_length = max(length)
    for idx, seq in enumerate(seqs):
        seq_t.append(torch.nn.functional.pad(torch.tensor([constant.tokenonthot[j] for j in seq]),(0,max_length - len(seq)),value=-1))
    seq = torch.stack(seq_t)
    mask = seq != -1
    seq[~mask] = 0
    return seq,mask,torch.tensor(cls_idx),ss

class InverseFold2DTrainer(Trainer):
    
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device)
        
    def generate_one_step(self, data, is_train=True):
        fasta,mask,idx = data
        x1 = torch.nn.functional.one_hot(fasta.to(torch.int64),num_classes=5).float().to(self.device)
        b, s ,d = x1.shape
        if is_train:
            xt, t = flow_utils.dirichlet_sample_cond_prob_path(x1,1)
        else:
            xt, t = flow_utils.dirichlet_sample_cond_prob_path(x1,0)
        mask_cls = torch.zeros(b,1,dtype=torch.bool).to(self.device)
        x1 = x1.to(self.device)
        xt = xt.to(self.device)
        t = t.to(self.device)
        mask = mask.to(self.device)
        idx = idx.to(self.device).to(torch.float32)
        mask = torch.cat([mask_cls,mask],dim=1)
        attn_mask = torch.einsum('bi,bj->bij',mask,mask)
        attn_mask = attn_mask.to(self.device)
        vt, x_hidden,c_hidden = self.model(xt,t,cls=idx,mask=~attn_mask,c=None)
        pred_clone = vt[...,1:,:].clone().argmax(-1)
        seq_auc = (pred_clone[mask[:,1:]] == x1[mask[:,1:]].argmax(dim=-1)).float().mean()
        return vt, x_hidden,c_hidden,x1,mask,seq_auc,xt
    
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        vt, x_hidden,c_hidden,x1,mask,seq_auc,xt = self.generate_one_step(data)
        seqerror = self.criterion(xt[mask[:,1:]][:,1:],vt[mask][:,1:].softmax(dim=-1),x1[mask[:,1:]][:,1:].argmax(dim=-1))
        loss = seqerror
        loss.backward()
        self.optimizer.step()
        return loss.item(),seq_auc.item()
    
    def evaluate(self, dataloader, time_steps=10):
        self.model.eval()
        total_loss = 0
        total_auc = 0
        with torch.no_grad():
            for data in dataloader:
                vt, x_hidden,c_hidden,x1,mask,seq_auc,_ = self.generate_one_step(data,is_train=False)
                loss = torch.nn.functional.cross_entropy(vt[mask][:,1:].softmax(dim=-1),x1[mask[:,1:]][:,1:])
                total_loss += loss
                total_auc += seq_auc
        return total_loss / len(dataloader), total_auc / len(dataloader)
    
    def train(self, train_dataloader, valid_dataloader, test_dataloader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_auc = 0
            count = 0
            train_bar = tqdm.tqdm(train_dataloader)
            for data in train_bar:
                loss,auc = self.train_one_step(data)
                total_loss += loss
                total_auc += auc
                count += 1
                train_bar.set_description(f'[TRAIN] loss: {total_loss/count:.3f} seq_auc: {total_auc/count:.3f}')
            train_loss = total_loss / len(train_dataloader)
            train_auc = total_auc / len(train_dataloader)
            valid_loss, valid_auc = self.evaluate(valid_dataloader)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}')
            self.test(test_dataloader,10)
        return self.model
    
    @torch.no_grad()
    def generate_on_timesteps(self, dataloader, timesteps=10):
        device = self.device
        model = self.model.to(device)
        pbar = tqdm.tqdm(dataloader)
        aucs = [[] for i in range(4)]
        target = []
        pred = []
        test_results = []
        for fasta,mask,cls_idx in pbar:
            x1 = torch.nn.functional.one_hot(fasta.to(torch.int64),num_classes=5).float().to(device)
            b, s ,d = x1.shape
            xt, _ = flow_utils.dirichlet_sample_cond_prob_path(x1,0)
            mask_cls = torch.zeros(b,1,dtype=torch.bool).to(device)
            x1 = x1.to(device)
            xt = xt.to(device)
            mask = mask.to(device)
            mask = torch.cat([mask_cls,mask],dim=1)
            cls_idx = cls_idx.to(device).to(torch.float32)
            attn_mask = torch.einsum('bi,bj->bij',mask,mask)
            attn_mask = attn_mask.to(device)
            eye = torch.eye(d).to(device)
            t_span = torch.linspace(1.001,8,timesteps).to(device)
            for i, (s,t) in enumerate(zip(t_span[:-1],t_span[1:])):
                x_,_,_ = model(xt, s[None], cls=cls_idx, mask=~attn_mask, c=None)
                c_factor = model.cond_flow.c_factor(xt.cpu().numpy(),s.item())
                c_factor = torch.from_numpy(c_factor).to(x_)
                c_factor = torch.nan_to_num(c_factor)
                flow_probs = torch.softmax(x_[:,1:]/1.0, dim = -1)
                cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
                flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
                xt = xt + flow * (t - s)
                xt = simplex_proj(xt)
            vt = xt
            target.append(x1[mask[:,1:]][:,1:].argmax(dim=-1).cpu().numpy())
            pred.append(vt[mask[:,1:]][:,1:].argmax(dim=-1).cpu().numpy())
            pred_clone = vt.clone().argmax(-1)
            predprob = vt[mask[:,1:],:][:,1:]
            for k in range(4):
                aucs[k].append(top_k_accuracy(predprob,x1[mask[:,1:]][:,1:].argmax(-1),k=k+1).item())
            mask = mask.cpu()
            x1 = x1.cpu()
            pred_clone = pred_clone.cpu()
            auc = np.mean(aucs[0])
            auc_1 = np.mean(aucs[1])
            auc_2 = np.mean(aucs[2])
            target_ = np.concatenate(target)
            pred_ = np.concatenate(pred)
            macro_f1 = f1_score(target_,pred_,average='macro')
            pbar.set_description(f'[TEST  on {device}] top 1 auc: {auc:.3f}, top 2 auc {auc_1:.3f}, top 3 auc {auc_2:.3f}, macro f1 {macro_f1  :.3f}')

    def test(self, dataloader, timesteps):
        self.model.eval()
        self.generate_on_timesteps(dataloader,timesteps)
    
    def generate(self, data, timesteps=10):
        self.model.eval()
        with torch.no_grad():
            fasta,mask,idx,ss = data
            seq_traj = []
            seq_rec_traj = []
            x1 = torch.nn.functional.one_hot(fasta.to(torch.int64),num_classes=5).float().to(self.device)
            b, s ,d = x1.shape
            xt, _ = flow_utils.dirichlet_sample_cond_prob_path(x1,0)
            mask_cls = torch.zeros(b,1,dtype=torch.bool).to(self.device)
            x1 = x1.to(self.device)
            xt = xt.to(self.device)
            mask = mask.to(self.device)
            mask = torch.cat([mask_cls,mask],dim=1)
            cls_idx = idx.to(self.device).to(torch.float32)
            attn_mask = torch.einsum('bi,bj->bij',mask,mask)
            attn_mask = attn_mask.to(self.device)
            eye = torch.eye(d).to(self.device)
            t_span = torch.linspace(1.001,8,timesteps).to(self.device)
            pair = ss2pair(ss[0])
            for i, (s,t) in enumerate(zip(t_span[:-1],t_span[1:])):
                x_,_,_ = self.model(xt, s[None], cls=cls_idx, mask=~attn_mask, c=None)
                c_factor = self.model.cond_flow.c_factor(xt.cpu().numpy(),s.item())
                c_factor = torch.from_numpy(c_factor).to(x_)
                c_factor = torch.nan_to_num(c_factor)
                flow_probs = torch.softmax(x_[:,1:]/1.0, dim = -1)
                cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
                flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
                xt = xt + flow * (t - s)
                xt = simplex_proj(xt)
                predprob = xt[mask[:,1:],:][:,1:]
                seq = ''.join('AUCG'[i] for i in predprob.argmax(-1).cpu().numpy())
                seq_traj.append(seq)
                predprob_proc = post_process(predprob,pair)
                seq_rec = ''.join('AUCG'[i] for i in predprob_proc.cpu().numpy())
                seq_rec_traj.append(seq_rec)
            predprob = xt[mask[:,1:],:][:,1:]
            seq = ''.join('AUCG'[i] for i in predprob.argmax(-1).cpu().numpy())
            seq_traj.append(seq)
            predprob_proc = post_process(predprob,pair)
            seq_rec = ''.join('AUCG'[i] for i in predprob_proc.cpu().numpy())
            seq_rec_traj.append(seq_rec)
            return seq_traj,seq_rec_traj
    
def ss2pair(target_ss):
    pair_edges = []
    for i in range(len(target_ss)):
        if target_ss[i] == '(':
            for j in range(i+1,len(target_ss)):
                if target_ss[j] == ')':
                    pair_edges.append([i,j])
                    pair_edges.append([j,i])
                    target_ss = target_ss[:i] + '.' + target_ss[i+1:j] + '.' + target_ss[j+1:]
                    break
        if target_ss[i] == '[':
            for j in range(i+1,len(target_ss)):
                if target_ss[j] == ']':
                    pair_edges.append([i,j])
                    pair_edges.append([j,i])
                    target_ss = target_ss[:i] + '.' + target_ss[i+1:j] + '.' + target_ss[j+1:]
                    break
        if target_ss[i] == '{':
            for j in range(i+1,len(target_ss)):
                if target_ss[j] == '}':
                    pair_edges.append([i,j])
                    pair_edges.append([j,i])
                    target_ss = target_ss[:i] + '.' + target_ss[i+1:j] + '.' + target_ss[j+1:]
                    break
        if target_ss[i] == '<':
            for j in range(i+1,len(target_ss)):
                if target_ss[j] == '>':
                    pair_edges.append([i,j])
                    pair_edges.append([j,i])
                    target_ss = target_ss[:i] + '.' + target_ss[i+1:j] + '.' + target_ss[j+1:]
                    break
    return pair_edges

def ss2edge(target_ss):
    edge = []
    edge_attr = []
    edge += [[i,i+1] for i in range(len(target_ss)-1)]
    edge_attr += [[0,0] for i in range(len(target_ss)-1)]
    edge += [[i+1,i] for i in range(len(target_ss)-1)]
    edge_attr += [[0,0] for i in range(len(target_ss)-1)]
    pair_edges = ss2pair(target_ss)
    for pair in pair_edges:
        edge.append(pair)
        edge_attr.append([1,1])
    return edge,edge_attr

class weighted_cross_entropy(torch.nn.Module):
    def __init__(self):
        super(weighted_cross_entropy,self).__init__()

    def _normal_weight(self,weight):
        # remain batch dim
        weight = weight / weight.sum(dim=-1,keepdim=True) * weight.shape[-1]
        return weight
    
    def forward(self, input, pred, target):
        # input/pred dim: [b, s, d]
        # target dim: [b, s]
        # for input[b, i, :], target[b, i] is the index of the target class, if input is closer to target, the weight is smaller
        # weight = torch.argmax(input, dim=-1) == target # [b, s]
        weight = torch.nn.functional.cross_entropy(input, target, reduction='none') # [b, s]
        weight = self._normal_weight(weight)
        weight_ = weight.clone().detach().requires_grad_(True)
        ce_pred_tar = torch.nn.functional.cross_entropy(pred, target, reduction='none') # [b, s]
        return (weight_ * ce_pred_tar).sum() / weight_.sum()


def main(device='cuda:0'):
    rfam_seq = pd.read_csv('rfam_seq.csv')
    for rfam_acc in rfam_seq['rfam_acc'].unique():
        dataset = InverseFoldDataset('rfam_seq.csv',target_rfam=rfam_acc)
        if len(dataset) == 0:
            continue
        train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),int(len(dataset)*0.1),len(dataset) - int(len(dataset)*0.8) - int(len(dataset)*0.1)])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=False,collate_fn=collet_fn_rfamflow)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=4,shuffle=False,collate_fn=collet_fn_rfamflow)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collet_fn_rfamflow)
        model = sequence_flow.SequenceFlow(5,128,6,16,16).to(device)
        model.load_state_dict(torch.load(f'./ckpts/bestmodel_inverse2d_{rfam_acc}.pth')['model'])
        trainer = InverseFold2DTrainer(model,None,weighted_cross_entropy(),device)
        output = f'{rfam_acc}_testgen.fasta'
        # clear the output file
        with open(output,'w') as f:
            f.write('')
        for gen_idx in tqdm.trange(1000):
            loader = iter(test_dataloader)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(test_dataloader)
                data = next(loader)
            seq_traj,seq_rec_traj = trainer.generate(data,timesteps=10)
            with open(output,'a') as f:
                for idx, seq in enumerate(seq_traj):
                    ss = data[3][0]
                    mfe_ss = cal_ss(seq)
                    f.write(f'>{rfam_acc}_{gen_idx}_time_{idx}_ss_{ss}_mfess_{mfe_ss}\n')
                    f.write(seq + '\n')
                    f.write(f'>{rfam_acc}_{gen_idx}_time_{idx}_ss_{ss}_mfess_{mfe_ss}_rec\n')
                    f.write(seq_rec_traj[idx] + '\n')
    return 

# def warmup(x):
#     if x < 50:
#         return 1 # 0.1 + 0.9 * x / 50
#     elif x < 100:
#         return 1
#     else:
#         # cosine decay
#         return 0.45 * (1 + np.cos((x-100) / 1900 * np.pi)) + 0.1




if __name__ == '__main__':
    main()
    pass