
import torch
import warnings
warnings.filterwarnings('ignore')


import torch.utils
from modules import sequence_flow
from utils import flow_utils
import numpy as np
import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
from models.trainer import Trainer

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

def random_set_seed(seed=0):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        seed = np.random.randint(0,100000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

class wrap_model(torch.nn.Module):
    def __init__(self,model):
        super(wrap_model,self).__init__()
        self.model = model
    def forward(self,inputs):
        return self.model.forward_embedding(**inputs)

def loss_nll_flatten(S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av

from models import inversefold2d
from modules.inv3d.RNAMPNN import RNAMPNN
import torch
from ml_collections import ConfigDict
from modules.inv3d import rdesign_dataproc as dataproc
class test_dataset(torch.utils.data.Dataset):
    
    def __init__(self,datapath):
        self.data = self.prepare_data(datapath)

    def prepare_data(self,datapath):
        alphabet_set = set(['A', 'U', 'C', 'G'])
        datas = []
        data = np.load(datapath,allow_pickle=True)
        for entry in tqdm.tqdm(data):
            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)
            entry['cluster'] = 0
            bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
            if len(bad_chars) == 0  and len(entry['seq']) <= 50:
                datas.append(entry)
            # entry['raw_seq'] = entry['seq']
            # entry['seq'] = 'A' * len(entry['seq'])
            # datas.append(entry)
        return datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]

class InverseFold3DTrainer(Trainer):
    
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device)
        
    def generate_one_step(self, data, is_train=False):
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = data
        X, S, mask, lengths, clus, ss_pos = dataproc.cuda((X, S, mask, lengths, clus, ss_pos), device='cuda:0')
        x1 = torch.nn.functional.one_hot((S).to(torch.int64),num_classes=4).float().to('cuda:0')
        b, s ,d = x1.shape
        if is_train:
            xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1,1)
        else:
            xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1,0) # alpha = 0 means uniform distribution
        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b,1,dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls,mask],dim=1)
        attn_mask = torch.einsum('bi,bj->bij',mask_,mask_)
        triu = torch.triu(torch.ones(s+1,s+1),diagonal=1).bool()
        attn_mask[:,triu] = False
        attn_mask = attn_mask.to('cuda:0')
        output,x_raw,c_raw = self.model(xt,t,mask=~attn_mask.bool(),c={'inputs':{'X':X, 'S':S, 'mask':mask}})
        if is_train:
            return output,mask_,S,x1,xt
        else:
            return output,mask_,S,x1
    
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        output,mask_,S,x1,xt = self.generate_one_step(data,is_train=True)
        loss = self.criterion(xt[mask_[:,1:].bool()],output[:,1:][mask_[:,1:].bool()].softmax(dim=-1),S[mask_[:,1:].bool()])
        seqauc = (output[:,1:][mask_[:,1:].bool()].argmax(dim=-1) == x1[mask_[:,1:].bool()].argmax(dim=-1)).float().mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(),seqauc.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_auc = 0
        with torch.no_grad():
            for data in dataloader:
                output,mask_,S,x1 = self.generate_one_step(data)
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
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = data
        X, S, mask, lengths, clus, ss_pos = dataproc.cuda((X, S, mask, lengths, clus, ss_pos), device='cuda:0')
        x1 = torch.nn.functional.one_hot((S).to(torch.int64),num_classes=4).float().to('cuda:0')
        b, s ,d = x1.shape
        xt,t = flow_utils.dirichlet_sample_cond_prob_path(x1,0) # alpha = 0 means uniform distribution
        S_pred = torch.ones_like(S)
        S_pred = S_pred.long()
        x1 = torch.nn.functional.one_hot((S_pred).to(torch.int64),num_classes=4).float().to('cuda:0')
        xt = torch.ones_like(xt) / 4 # just for test, we don't want random initialization
        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b,1,dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls,mask],dim=1)
        attn_mask = torch.einsum('bi,bj->bij',mask_,mask_)
        triu = torch.triu(torch.ones(s+1,s+1),diagonal=1).bool()
        attn_mask[:,triu] = False
        attn_mask = attn_mask.to('cuda:0')
        t_span = torch.linspace(1.001,8,timesteps).to('cuda:0')
        eye = torch.eye(d).to(xt)
        for i, (s,t) in enumerate(zip(t_span[:-1],t_span[1:])):
            x_,_,_ = self.model(xt, s[None], mask=~attn_mask.bool(), c={'inputs':{'X':X, 'S':S, 'mask':mask}})
            c_factor = self.model.cond_flow.c_factor(xt.cpu().numpy(),s.item())
            c_factor = torch.from_numpy(c_factor).to(x_)
            c_factor = torch.nan_to_num(c_factor)
            flow_probs = torch.softmax(x_[:,1:]/1.0, dim = -1)
            cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
            xt = xt + flow * (t - s)
            xt = simplex_proj(xt)
        output = xt
        S_pred = output.argmax(dim=-1)
        
        seqauc = (S_pred == S)[mask.bool()].float().mean()
        mask = mask.bool()
        if not seqauc.isnan():
            return seqauc.item(),S,S_pred,mask
        else:
            return None,S,S_pred,mask

    def test(self, dataloader, timesteps=10):
        self.model.eval()
        total_auc = 0
        total_f1 = 0
        total_count = 0
        short_results = []
        medium_results = []
        long_results = []
        target = []
        pred = []
        pbar = tqdm.tqdm(dataloader)
        for data in pbar:
            seqauc,S,S_pred,mask = self.generate_on_timesteps(data,timesteps)
            b = S.shape[0]
            if seqauc is not None:
                for i in range(b):
                    tmp_target = S[i,mask[i]].cpu().numpy()
                    tmp_pred = S_pred[i,mask[i]].cpu().numpy()
                    if len(tmp_target) <= 50:
                        short_results.append((tmp_target,tmp_pred))
                    elif len(tmp_target) <= 100:
                        medium_results.append((tmp_target,tmp_pred))
                    else:
                        long_results.append((tmp_target,tmp_pred))
                target.append(S[mask.bool()].cpu().numpy())
                pred.append(S_pred[mask.bool()].cpu().numpy())
                total_f1 += f1_score(S[mask.bool()].cpu().numpy(),S_pred[mask.bool()].cpu().numpy(),average='macro')
                total_auc += seqauc
                total_count += 1
                pbar.set_description(f'[TEST] seq_auc: {total_auc/total_count:.3f}')
        auc = total_auc / total_count
        # target = np.concatenate(target)
        # pred = np.concatenate(pred)
        # macro_f1 = f1_score(target,pred,average='macro')
        macro_f1 = total_f1 / total_count
        # short_target = np.concatenate([i[0] for i in short_results])
        # short_pred = np.concatenate([i[1] for i in short_results])
        # medium_target = np.concatenate([i[0] for i in medium_results])
        # medium_pred = np.concatenate([i[1] for i in medium_results])
        # long_target = np.concatenate([i[0] for i in long_results])
        # long_pred = np.concatenate([i[1] for i in long_results])
        short_macro_f1 = np.mean([f1_score(short_target,short_pred,average='macro') for short_target,short_pred in short_results])
        medium_macro_f1 = np.mean([f1_score(medium_target,medium_pred,average='macro') for medium_target,medium_pred in medium_results])
        long_macro_f1 = np.mean([f1_score(long_target,long_pred,average='macro') for long_target,long_pred in long_results])
        short_auc = np.mean([np.mean(short_target == short_pred) for short_target,short_pred in short_results])
        medium_auc = np.mean([np.mean(medium_target == medium_pred) for medium_target,medium_pred in medium_results])
        long_auc = np.mean([np.mean(long_target == long_pred) for long_target,long_pred in long_results])
        print(f'[TEST  on cuda:0] top 1 auc: {auc:.3f}, macro f1 {macro_f1:.3f}',flush=False)
        print(f'[TEST  on cuda:0] short auc: {short_auc:.3f}, medium auc {medium_auc:.3f}, long auc {long_auc:.3f}',flush=False)
        print(f'[TEST  on cuda:0] short macro f1: {short_macro_f1:.3f}, medium macro f1 {medium_macro_f1:.3f}, long macro f1 {long_macro_f1:.3f}',flush=False)

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
        weight = torch.ones_like(weight).to(torch.float32)
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
    for idx in range(8):
        test_set = test_dataset('./datas/RDesign/pdb2rdesignfmt_new.pt')
        # name_seq_dict = {data['name']:data['raw_seq'] for data in test_set}
        # name_tvt_dict = {data['name']:data[f'train_valid_test_{idx}'] for data in test_set}
        train_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='TRAIN']
        valid_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='VAL']
        test_set = [data for i, data in enumerate(test_set) if data[f'train_valid_test_{idx}']=='TEST']
        # train_set = test_dataset('./datas/RDesign/train_data.pt')
        # valid_set = test_dataset('./datas/RDesign/val_data.pt')
        # test_set = test_dataset('./datas/RDesign/test_data.pt')
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, collate_fn=dataproc.featurize)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=2, collate_fn=dataproc.featurize)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=dataproc.featurize)
        # train_loader,valid_loader,test_loader = dataproc.load_data(16,'./datas/RDesign',num_workers=1)
        condition = RNAMPNN(ConfigDict(config))
        condition = condition.to('cuda:0')
        condition = wrap_model(condition)
        
        flow_model = sequence_flow.SequenceFlow(4,128,6,16,16,condition=condition).to('cuda:0')
        
        flow_model.load_state_dict(torch.load(f"ckpts/best_inv3dflow_ribodiffusion_{idx}.pth")['model'])
    
        optim_param_group = []
        lr = 3e-4

        for name,param in flow_model.named_parameters():
            if 'condition' not in name:
                optim_param_group.append({'params':param,'lr':lr})
            else:
                optim_param_group.append({'params':param,'lr':lr*0.1})
                # param.detach_()
        optimizer = optim.Adam(optim_param_group,lr=lr)
        
        trainer = InverseFold3DTrainer(flow_model, optimizer,  weighted_cross_entropy(), 'cuda:0')
        trainer.train(train_loader,valid_loader,test_loader,50)
        trainer.test(test_loader,10)
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