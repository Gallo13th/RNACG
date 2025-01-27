
import torch.utils
from modules import sequence_flow, track_2d
from utils import flow_utils,constant
import numpy as np
import torch
import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score 
from models.trainer import Trainer
from modules.inv3d.RNAMPNN import RNAMPNN2D
import torch
from ml_collections import ConfigDict
from modules.inv3d import rdesign_dataproc as dataproc

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
        # print(pred1hot[start],pred1hot[end])
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
    def __init__(self, data_file_paths):
        self.data_file_paths = data_file_paths
        self.edges = []
        self.seqs = []
        self.transfer_all()
        
    def preprocess(self, data):
        seqs = []
        edges = []
        for item in data[:300]:
            edge = item['edges']
            tmp = list(item['seq'])
            for i in range(len(tmp)):
                if tmp[i] in ['DA', 'DC', 'DG', 'DT']:
                    tmp[i] = tmp[i][1]
                if tmp[i] not in ['A', 'C', 'G', 'T', 'U']:
                    tmp[i] = 'N'
                if tmp[i] == 'U':
                    tmp[i] = 'T'
            seqs.append(tmp)
            edges.append(edge)
        return edges, seqs
    
    def transfer_all(self):
        data = np.load(self.data_file_paths, allow_pickle=True)
        edges, seqs = self.preprocess(data)
        for edge,seq in zip(edges,seqs):
            if len(seq) > 64:
                continue
            self.edges.append(edge)
            self.seqs.append(seq)
        self.sorted_idx = np.argsort([len(i) for i in self.seqs])
        self.edges = [self.edges[i] for i in self.sorted_idx]
        self.seqs = [self.seqs[i] for i in self.sorted_idx]
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.edges[idx], self.seqs[idx]

def collet_fn_inversefold(batch):
    edges = [torch.tensor(i[0]) for i in batch]
    seqs = [i[1] for i in batch]
    length = [len(i) for i in seqs]
    seq_t = []
    max_length = max(length)
    for idx, seq in enumerate(seqs):
        seq_t.append(torch.nn.functional.pad(torch.tensor(['NACGT'.index(j) for j in seq]),(0,max_length - len(seq)),value=-1))   
    seq = torch.stack(seq_t)
    mask = seq != -1
    seq[~mask] = 0
    return seq,edges,mask

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

class InverseFold3DTrainer(Trainer):
    
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device)
        
    def generate_one_step(self, data, is_train=True):
        S, edge, mask = data
        S, mask, edge= dataproc.cuda((S, mask, edge), device='cuda:0')
        B, L = S.shape
        
        x1 = torch.nn.functional.one_hot((S).to(torch.int64),num_classes=5).float().to('cuda:0')
        b, s ,d = x1.shape
        if is_train:
            xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1)
        else:
            xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1,0)
        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b,1,dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls,mask],dim=1)
        attn_mask = torch.einsum('bi,bj->bij',mask_,mask_)
        attn_mask = attn_mask.to('cuda:0')
        output,x_raw,c_raw = self.model(xt,t,mask=~attn_mask.bool(),c={'inputs':{'X':xt, 'S':S, 'mask':mask, 'edges':edge}})
        return output,mask_,S,x1,xt
    
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        vt,mask,S,x1,xt = self.generate_one_step(data)
        seqerror = self.criterion(xt[mask[:,1:]][:,1:],vt[mask][:,1:].softmax(dim=-1),x1[mask[:,1:]][:,1:].argmax(dim=-1))
        loss = seqerror
        seqauc = (vt[:,1:][mask[:,1:].bool()].argmax(dim=-1) == x1[mask[:,1:].bool()].argmax(dim=-1)).float().mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(),seqauc.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_auc = 0
        with torch.no_grad():
            for data in dataloader:
                output,mask_,S,x1,xt = self.generate_one_step(data,is_train=False)
                loss = torch.nn.functional.cross_entropy(output[:,1:][mask_[:,1:].bool()].softmax(dim=-1),S[mask_[:,1:].bool()])
                seqauc = (output[:,1:][mask_[:,1:].bool()].argmax(dim=-1) == x1[mask_[:,1:].bool()].argmax(dim=-1)).float().mean()
                total_loss += loss.item()
                total_auc += seqauc.item()
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
    def generate_on_timesteps(self, data, timesteps=10):
        S, edge, mask = data
        S, mask, edge= dataproc.cuda((S, mask, edge), device='cuda:0')
        x1 = torch.nn.functional.one_hot((S).to(torch.int64),num_classes=5).float().to('cuda:0')
        b, s ,d = x1.shape
        xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1,0)
        S_pred = torch.ones_like(S)
        S_pred = S_pred.long()
        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b,1,dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls,mask],dim=1)
        attn_mask = torch.einsum('bi,bj->bij',mask_,mask_)
        attn_mask = attn_mask.to('cuda:0')
        t_span = torch.linspace(1.001,8,timesteps).to('cuda:0')
        eye = torch.eye(d).to(xt)
        for i, (s,t) in enumerate(zip(t_span[:-1],t_span[1:])):
            x_,_,_ = self.model(xt, s[None], mask=~attn_mask.bool(), c={'inputs':{'X':xt, 'S':S, 'mask':mask, 'edges':data[1]}})
            c_factor = self.model.cond_flow.c_factor(xt.cpu().numpy(),s.item())
            c_factor = torch.from_numpy(c_factor).to(x_)
            c_factor = torch.nan_to_num(c_factor)
            flow_probs = torch.softmax(x_[:,1:]/1.0, dim = -1)
            cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
            xt = xt + flow * (t - s)
            xt = simplex_proj(xt)
        output = xt
        S_pred = output[...,1:].argmax(dim=-1)
        S = S - 1
        seqauc = (S_pred == S)[mask.bool()].float().mean()
        mask = mask.bool()
        return seqauc.item(),S,S_pred,mask

    def test(self, dataloader, timesteps=10):
        self.model.eval()
        total_auc = 0
        total_count = 0
        # short_results = []
        # medium_results = []
        # long_results = []
        target = []
        pred = []
        pbar = tqdm.tqdm(dataloader)
        for data in pbar:
            seqauc,S,S_pred,mask = self.generate_on_timesteps(data,timesteps)
            # b = S.shape[0]
            # for i in range(b):
            #     tmp_target = S[i,mask[i]].cpu().numpy()
            #     tmp_pred = S_pred[i,mask[i]].cpu().numpy()
                # if len(tmp_target) <= 50:
                #     short_results.append((tmp_target,tmp_pred))
                # elif len(tmp_target) <= 100:
                #     medium_results.append((tmp_target,tmp_pred))
                # else:
                #     long_results.append((tmp_target,tmp_pred))
            target.append(S[mask.bool()].cpu().numpy())
            pred.append(S_pred[mask.bool()].cpu().numpy())
            total_auc += seqauc
            total_count += 1
            pbar.set_description(f'[TEST] seq_auc: {total_auc/total_count:.3f}')
        auc = total_auc / total_count
        target = np.concatenate(target)
        pred = np.concatenate(pred)
        macro_f1 = f1_score(target,pred,average='macro')
        # short_target = np.concatenate([i[0] for i in short_results])
        # short_pred = np.concatenate([i[1] for i in short_results])
        # medium_target = np.concatenate([i[0] for i in medium_results])
        # medium_pred = np.concatenate([i[1] for i in medium_results])
        # long_target = np.concatenate([i[0] for i in long_results])
        # long_pred = np.concatenate([i[1] for i in long_results])
        # short_macro_f1 = f1_score(short_target,short_pred,average='macro')
        # medium_macro_f1 = f1_score(medium_target,medium_pred,average='macro')
        # long_macro_f1 = f1_score(long_target,long_pred,average='macro')
        # short_auc = np.mean(short_target == short_pred)
        # medium_auc = np.mean(medium_target == medium_pred)
        # long_auc = np.mean(long_target == long_pred)
        print(f'[TEST  on cuda:0] top 1 auc: {auc:.3f}, macro f1 {macro_f1:.3f}',flush=False)
        # print(f'[TEST  on cuda:0] short auc: {short_auc:.3f}, medium auc {medium_auc:.3f}, long auc {long_auc:.3f}',flush=False)
        # print(f'[TEST  on cuda:0] short macro f1: {short_macro_f1:.3f}, medium macro f1 {medium_macro_f1:.3f}, long macro f1 {long_macro_f1:.3f}',flush=False)

    def generate(self, dataloader, timesteps=10):
        self.model.eval()
        short_results = []
        medium_results = []
        long_results = []
        pbar = tqdm.tqdm(dataloader)
        for data in pbar:
            seqauc,S,S_pred,mask = self.generate_on_timesteps(data,timesteps)
            names = data[-1]
            b = S.shape[0]
            for i in range(b):
                tmp_pred = S_pred[i,mask[i]].cpu().numpy()
                tmp_pred = ''.join(['AUCG'[j] for j in tmp_pred])
                if len(tmp_pred) <= 50:
                    short_results.append((names[i],tmp_pred))
                elif len(tmp_pred) <= 100:
                    medium_results.append((names[i],tmp_pred))
                else:
                    long_results.append((names[i],tmp_pred))
        return short_results,medium_results,long_results

class wrap_model(torch.nn.Module):
    def __init__(self,model):
        super(wrap_model,self).__init__()
        self.model = model
    def forward(self,inputs):
        # show the inputs
        return self.model.forward_embedding(**inputs)

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
        # weight_ = torch.ones_like(weight).to(torch.float32).requires_grad_(True)
        ce_pred_tar = torch.nn.functional.cross_entropy(pred, target, reduction='none') # [b, s]
        # return ce_pred_tar.mean()
        return (weight_ * ce_pred_tar).sum() / weight_.sum()


def main():
    config = {"device": "cuda", 
              "node_feat_types": ["angle", "distance", "direction"],
              "edge_feat_types": ["orientation", "distance", "direction"], 
              "num_encoder_layers": 3,
              "num_decoder_layers": 3, 
              "hidden": 128, 
              "k_neighbors": 30, 
              "vocab_size": 4,
              "shuffle": 0.0, 
              "dropout": 0.1, 
              "smoothing": 0.1, 
              "weigth_clu_con": 0.5, 
              "weigth_sam_con": 0.5, 
              "ss_temp": 0.5}
    for idx in range(1):
        dataset = InverseFoldDataset(f'datas/TestSetA.npy')
        # name_seq_dict = {data['name']:data['raw_seq'] for data in test_set}
        # name_tvt_dict = {data['name']:data[f'train_valid_test_{idx}'] for data in test_set}
        # train_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='TRAIN']
        # valid_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='VAL']
        # test_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='TEST']
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=10, collate_fn=collet_fn_inversefold)
        # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=10, collate_fn=collet_fn_inversefold)
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.6),int(len(dataset)*0.2),len(dataset)-int(len(dataset)*0.6)-int(len(dataset)*0.2)])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collet_fn_inversefold)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collet_fn_inversefold)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collet_fn_inversefold)
        # S, edge, mask = next(iter(test_loader))
        # print(edge)
        # raise ValueError
        # raise ValueError
        condition = RNAMPNN2D(ConfigDict(config))
        condition = condition.to('cuda:0')
        condition = wrap_model(condition)
        
        flow_model = sequence_flow.SequenceFlow(5,128,6,16,16,condition=condition).to('cuda:0')
        
        # flow_model.load_state_dict(torch.load(f"ckpts/best_inv3dflow_ribodiffusion_{idx}.pth")['model'],strict=False)
    
        optim_param_group = []
        lr = 3e-4

        for name,param in flow_model.named_parameters():
            if 'condition' not in name:
                optim_param_group.append({'params':param,'lr':lr})
            else:
                optim_param_group.append({'params':param,'lr':lr})
                # param.detach_()
        optimizer = optim.Adam(optim_param_group,lr=lr)
        
        trainer = InverseFold3DTrainer(flow_model, optimizer, weighted_cross_entropy(), 'cuda:0')
        trainer.train(train_loader,valid_loader,test_loader,50)
        # trainer.test(test_loader,10)
        # short, medium, long = trainer.generate(test_loader,1)
        # with open(f'output_{idx}.txt','w') as f:
        #     for name, target in short:
        #         if '_missed' in name:
        #             continue
        #         f.write(f'{name}\t{target}\t{name_seq_dict[name]}\t{name_tvt_dict[name]}\n')
        #     for name, target in medium:
        #         if '_missed' in name:
        #             continue
        #         f.write(f'{name}\t{target}\t{name_seq_dict[name]}\t{name_tvt_dict[name]}\n')
        #     for name, target in long:
        #         if '_missed' in name:
        #             continue
        #         f.write(f'{name}\t{target}\t{name_seq_dict[name]}\t{name_tvt_dict[name]}\n')
            
        
        
    return False
if __name__ == '__main__':
    main()