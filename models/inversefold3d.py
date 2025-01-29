import Bio.Data
import Bio.Data.PDBData
import torch
import warnings
import numpy as np
import tqdm
import torch.optim as optim
import Bio.PDB
import RNA
from sklearn.metrics import f1_score
from ml_collections import ConfigDict
from modules.inv3d.RNAMPNN import RNAMPNN
from modules.inv3d import rdesign_dataproc as dataproc
from models.trainer import Trainer
from modules import sequence_flow
from utils import flow_utils
from utils.flow_utils import simplex_proj

# Suppress warnings
warnings.filterwarnings('ignore')

# Default configuration for the model
DEFAULT_CONFIG = {
    "device": "cuda",
    "node_feat_types": ["angle", "distance", "direction"],
    "edge_feat_types": ["orientation", "distance", "direction"],
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "hidden": 128,
    "k_neighbors": 30,
    "vocab_size": 4,
    "dropout": 0.1,
}

def top_k_accuracy(preds, labels, k=4):
    """
    Calculate top-k accuracy.

    Args:
        preds: Predicted probabilities.
        labels: Ground truth labels.
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy.
    """
    topk = torch.topk(preds, k=k, dim=-1).indices
    labels = labels.unsqueeze(-1).expand_as(topk)
    return (topk == labels).any(dim=-1).float().mean()

def random_set_seed(seed=0):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value. If 0, a random seed is generated.

    Returns:
        The seed value used.
    """
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        seed = np.random.randint(0, 100000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

class wrap_model(torch.nn.Module):
    """
    Wrapper class for the model to simplify the forward pass.
    """
    def __init__(self, model):
        super(wrap_model, self).__init__()
        self.model = model

    def forward(self, inputs):
        return self.model.forward_embedding(**inputs)

class test_dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and preprocessing test data.
    """
    def __init__(self, datapath):
        self.data = self.prepare_data(datapath)

    def prepare_data(self, datapath):
        """
        Load and preprocess data from the given path.

        Args:
            datapath: Path to the data file.

        Returns:
            List of preprocessed data entries.
        """
        alphabet_set = set(['A', 'U', 'C', 'G'])
        datas = []
        data = np.load(datapath, allow_pickle=True)
        for entry in tqdm.tqdm(data):
            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)
            entry['cluster'] = 0
            bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
            if len(bad_chars) == 0 and len(entry['seq']) <= 50:
                datas.append(entry)
        return datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class InverseFold3DTrainer(Trainer):
    """
    Trainer class for 3D inverse folding tasks.
    """
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device)

    def generate_one_step(self, data, is_train=False):
        """
        Generate predictions for a single step.

        Args:
            data: Input data.
            is_train: Whether the model is in training mode.

        Returns:
            Model outputs and intermediate values.
        """
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = data
        X, S, mask, lengths, clus, ss_pos = dataproc.cuda((X, S, mask, lengths, clus, ss_pos), device='cuda:0')
        x1 = torch.nn.functional.one_hot((S).to(torch.int64), num_classes=4).float().to('cuda:0')
        b, s, d = x1.shape

        if is_train:
            xt, t = flow_utils.dirichlet_sample_cond_prob_path(x1, 1)
        else:
            xt, t = flow_utils.dirichlet_sample_cond_prob_path(x1, 0)  # alpha = 0 means uniform distribution

        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b, 1, dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls, mask], dim=1)
        attn_mask = torch.einsum('bi,bj->bij', mask_, mask_)
        triu = torch.triu(torch.ones(s + 1, s + 1), diagonal=1).bool()
        attn_mask[:, triu] = False
        attn_mask = attn_mask.to('cuda:0')

        output, x_raw, c_raw = self.model(xt, t, mask=~attn_mask.bool(), c={'inputs': {'X': X, 'S': S, 'mask': mask}})

        if is_train:
            return output, mask_, S, x1, xt
        else:
            return output, mask_, S, x1

    def train_one_step(self, data):
        """
        Perform a single training step.

        Args:
            data: Input data.

        Returns:
            Loss and accuracy for the step.
        """
        self.optimizer.zero_grad()
        output, mask_, S, x1, xt = self.generate_one_step(data, is_train=True)
        loss = self.criterion(xt[mask_[:, 1:].bool()], output[:, 1:][mask_[:, 1:].bool()].softmax(dim=-1), S[mask_[:, 1:].bool()])
        seqauc = (output[:, 1:][mask_[:, 1:].bool()].argmax(dim=-1) == x1[mask_[:, 1:].bool()].argmax(dim=-1)).float().mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(), seqauc.item()

    def evaluate(self, dataloader):
        """
        Evaluate the model on a given dataset.

        Args:
            dataloader: DataLoader for the evaluation dataset.

        Returns:
            Average loss and accuracy over the dataset.
        """
        self.model.eval()
        total_loss = 0
        total_auc = 0
        with torch.no_grad():
            for data in dataloader:
                output, mask_, S, x1 = self.generate_one_step(data)
                loss = torch.nn.functional.cross_entropy(output[:, 1:][mask_[:, 1:].bool()].softmax(dim=-1), S[mask_[:, 1:].bool()])
                seqauc = (output[:, 1:][mask_[:, 1:].bool()].argmax(dim=-1) == x1[mask_[:, 1:].bool()].argmax(dim=-1)).float().mean()
                total_loss += loss.item()
                total_auc += seqauc.item()
        return total_loss / len(dataloader), total_auc / len(dataloader)

    def train(self, train_dataloader, valid_dataloader, test_dataloader, num_epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            train_dataloader: DataLoader for the training dataset.
            valid_dataloader: DataLoader for the validation dataset.
            test_dataloader: DataLoader for the test dataset.
            num_epochs: Number of epochs to train.

        Returns:
            The trained model.
        """
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_auc = 0
            count = 0
            train_bar = tqdm.tqdm(train_dataloader)
            for data in train_bar:
                loss, auc = self.train_one_step(data)
                total_loss += loss
                total_auc += auc
                count += 1
                train_bar.set_description(f'[TRAIN] loss: {total_loss / count:.3f} seq_auc: {total_auc / count:.3f}')

            train_loss = total_loss / len(train_dataloader)
            train_auc = total_auc / len(train_dataloader)
            valid_loss, valid_auc = self.evaluate(valid_dataloader)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}')
            self.test(test_dataloader, 10)

        return self.model

    @torch.no_grad()
    def generate_on_timesteps(self, data, timesteps=10):
        """
        Generate predictions over multiple timesteps.

        Args:
            data: Input data.
            timesteps: Number of timesteps for generation.

        Returns:
            Sequence accuracy, true labels, predicted labels, and mask.
        """
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = data
        X, S, mask, lengths, clus, ss_pos = dataproc.cuda((X, S, mask, lengths, clus, ss_pos), device='cuda:0')
        x1 = torch.nn.functional.one_hot((S).to(torch.int64), num_classes=4).float().to('cuda:0')
        b, s, d = x1.shape
        xt, t = flow_utils.dirichlet_sample_cond_prob_path(x1, 0)  # alpha = 0 means uniform distribution
        S_pred = torch.ones_like(S)
        S_pred = S_pred.long()
        x1 = torch.nn.functional.one_hot((S_pred).to(torch.int64), num_classes=4).float().to('cuda:0')
        xt = torch.ones_like(xt) / 4  # Uniform initialization for testing
        x1 = x1.to('cuda:0')
        xt = xt.to('cuda:0')
        t = t.to('cuda:0')
        mask = mask.to('cuda:0')
        mask_cls = torch.zeros(b, 1, dtype=torch.bool).to('cuda:0')
        mask_ = torch.cat([mask_cls, mask], dim=1)
        attn_mask = torch.einsum('bi,bj->bij', mask_, mask_)
        triu = torch.triu(torch.ones(s + 1, s + 1), diagonal=1).bool()
        attn_mask[:, triu] = False
        attn_mask = attn_mask.to('cuda:0')
        t_span = torch.linspace(1.001, 8, timesteps).to('cuda:0')
        eye = torch.eye(d).to(xt)

        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            x_, _, _ = self.model(xt, s[None], mask=~attn_mask.bool(), c={'inputs': {'X': X, 'S': S, 'mask': mask}})
            c_factor = self.model.cond_flow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(x_)
            c_factor = torch.nan_to_num(c_factor)
            flow_probs = torch.softmax(x_[:, 1:] / 1.0, dim=-1)
            cond_flow = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
            xt = xt + flow * (t - s)
            xt = simplex_proj(xt)

        output = xt
        S_pred = output.argmax(dim=-1)
        seqauc = (S_pred == S)[mask.bool()].float().mean()
        mask = mask.bool()

        if not seqauc.isnan():
            return seqauc.item(), S, S_pred, mask
        else:
            return None, S, S_pred, mask

    def test(self, dataloader, timesteps=10):
        """
        Test the model on a given dataset.

        Args:
            dataloader: DataLoader for the test dataset.
            timesteps: Number of timesteps for generation.
        """
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
            seqauc, S, S_pred, mask = self.generate_on_timesteps(data, timesteps)
            b = S.shape[0]
            if seqauc is not None:
                for i in range(b):
                    tmp_target = S[i, mask[i]].cpu().numpy()
                    tmp_pred = S_pred[i, mask[i]].cpu().numpy()
                    if len(tmp_target) <= 50:
                        short_results.append((tmp_target, tmp_pred))
                    elif len(tmp_target) <= 100:
                        medium_results.append((tmp_target, tmp_pred))
                    else:
                        long_results.append((tmp_target, tmp_pred))
                target.append(S[mask.bool()].cpu().numpy())
                pred.append(S_pred[mask.bool()].cpu().numpy())
                total_f1 += f1_score(S[mask.bool()].cpu().numpy(), S_pred[mask.bool()].cpu().numpy(), average='macro')
                total_auc += seqauc
                total_count += 1
                pbar.set_description(f'[TEST] seq_auc: {total_auc / total_count:.3f}')

        auc = total_auc / total_count
        macro_f1 = total_f1 / total_count
        short_macro_f1 = np.mean([f1_score(short_target, short_pred, average='macro') for short_target, short_pred in short_results])
        medium_macro_f1 = np.mean([f1_score(medium_target, medium_pred, average='macro') for medium_target, medium_pred in medium_results])
        long_macro_f1 = np.mean([f1_score(long_target, long_pred, average='macro') for long_target, long_pred in long_results])
        short_auc = np.mean([np.mean(short_target == short_pred) for short_target, short_pred in short_results])
        medium_auc = np.mean([np.mean(medium_target == medium_pred) for medium_target, medium_pred in medium_results])
        long_auc = np.mean([np.mean(long_target == long_pred) for long_target, long_pred in long_results])

        print(f'[TEST on cuda:0] top 1 auc: {auc:.3f}, macro f1 {macro_f1:.3f}')
        print(f'[TEST on cuda:0] short auc: {short_auc:.3f}, medium auc {medium_auc:.3f}, long auc {long_auc:.3f}')
        print(f'[TEST on cuda:0] short macro f1: {short_macro_f1:.3f}, medium macro f1 {medium_macro_f1:.3f}, long macro f1 {long_macro_f1:.3f}')

    def generate(self, dataloader, timesteps=10):
        """
        Generate sequences from the model.

        Args:
            dataloader: DataLoader for the generation dataset.
            timesteps: Number of timesteps for generation.

        Returns:
            List of generated sequences.
        """
        self.model.eval()
        results = []
        pbar = tqdm.tqdm(dataloader)
        for data in pbar:
            _, S, S_pred, mask = self.generate_on_timesteps(data, timesteps)
            names = data[-1]
            b = S.shape[0]
            for i in range(b):
                tmp_pred = S_pred[i, mask[i]].cpu().numpy()
                tmp_pred = ''.join(['AUCG'[j] for j in tmp_pred])
                results.append((names[i], tmp_pred))
        return results

class weighted_cross_entropy(torch.nn.Module):
    """
    Weighted cross-entropy loss function.
    """
    def __init__(self):
        super(weighted_cross_entropy, self).__init__()

    def _normal_weight(self, weight):
        """
        Normalize the weight tensor.

        Args:
            weight: Input weight tensor.

        Returns:
            Normalized weight tensor.
        """
        weight = weight / weight.sum(dim=-1, keepdim=True) * weight.shape[-1]
        return weight

    def forward(self, input, pred, target):
        """
        Compute the weighted cross-entropy loss.

        Args:
            input: Input tensor.
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            Weighted cross-entropy loss.
        """
        weight = torch.nn.functional.cross_entropy(input, target, reduction='none')
        weight = self._normal_weight(weight)
        weight_ = weight.clone().detach().requires_grad_(True)
        ce_pred_tar = torch.nn.functional.cross_entropy(pred, target, reduction='none')
        return (weight_ * ce_pred_tar).sum() / weight_.sum()

def train():
    """
    Main training function for the 3D inverse folding task.
    """
    for idx in range(8):
        raw_set = test_dataset('./datas/RDesign/pdb2rdesignfmt_new.pt')
        train_set = [data for i, data in enumerate(raw_set) if data[f'train_valid_test_{idx}'] == 'TRAIN']
        valid_set = [data for i, data in enumerate(raw_set) if data[f'train_valid_test_{idx}'] == 'VAL']
        test_set = [data for i, data in enumerate(raw_set) if data[f'train_valid_test_{idx}'] == 'TEST']
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, collate_fn=dataproc.featurize)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=2, collate_fn=dataproc.featurize)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=dataproc.featurize)

        condition = RNAMPNN(ConfigDict(DEFAULT_CONFIG))
        condition = condition.to('cuda:0')
        condition = wrap_model(condition)

        flow_model = sequence_flow.SequenceFlow(4, 128, 6, 16, 16, condition=condition).to('cuda:0')
        flow_model.load_state_dict(torch.load(f"ckpts/best_inv3dflow_ribodiffusion_{idx}.pth")['model'])

        optim_param_group = []
        lr = 3e-4
        for name, param in flow_model.named_parameters():
            if 'condition' not in name:
                optim_param_group.append({'params': param, 'lr': lr})
            else:
                optim_param_group.append({'params': param, 'lr': lr * 0.1})

        optimizer = optim.Adam(optim_param_group, lr=lr)
        trainer = InverseFold3DTrainer(flow_model, optimizer, weighted_cross_entropy(), 'cuda:0')
        trainer.train(train_loader, valid_loader, test_loader, 50)
        trainer.test(test_loader, 10)

    return False

def cal_ss(seq):
    """Calculate secondary structure."""
    ss, _ = RNA.fold(seq)
    return ss

atom_cand = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]

def clean_pdb(pdb_file):
    # remove HETATM and protein residues
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    with open(pdb_file, 'w') as f:
        for line in lines:
            if line.startswith('ATOM'):
                resname = line[17:20].replace(' ', '')
                resname = resname + ' '*(3-len(resname))
                if resname in Bio.Data.PDBData.nucleic_letters_3to1_extended:
                    f.write(line)

def pdb2rdesignfmt(pdb_file):
    """Convert a PDB file to rdesignfmt."""
    # Clean PDB file
    clean_pdb(pdb_file)
    # Load PDB file
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    model = structure[0]
    seq = []
    # Extract first chain
    for chain in model:
        chain_A = chain

    # Extract coordinates
    coords = {'coords': {atom: [] for atom in atom_cand}}
    for residue in chain_A:
        resname = residue.get_resname()
        resname = resname + ' '*(3-len(resname))
        if resname not in Bio.Data.PDBData.nucleic_letters_3to1_extended:
            continue
        seq.append(Bio.Data.PDBData.nucleic_letters_3to1_extended[resname])
        for atom in atom_cand:
            try:
                coords['coords'][atom].append(residue[atom].get_coord())
            except KeyError:
                # print(f"Atom {atom} not found in residue {residue.get_resname()}")
                coords['coords'][atom].append([np.nan, np.nan, np.nan])
    coords['residues'] = seq
    coords['ss'] = '.' * len(seq)
    coords['name'] = pdb_file.split('/')[-1].split('.')[0]
    coords['cluster'] = 0
    coords['seq'] = ''.join(seq)
    return coords

def main(args):
    """
    Main function to run the training process.
    """
    ckpts_path = args.model
    device = args.device
    input_path = args.input
    output_path = args.output
    timesteps = args.n_steps
    condition = RNAMPNN(ConfigDict(DEFAULT_CONFIG))
    condition = condition.to(device)
    condition = wrap_model(condition)

    flow_model = sequence_flow.SequenceFlow(4, 128, 6, 16, 16, condition=condition).to(device)
    flow_model.load_state_dict(torch.load(ckpts_path)['model'])
    
    input_set = pdb2rdesignfmt(input_path)
    data = dataproc.featurize([input_set])
    
    trainer = InverseFold3DTrainer(flow_model, None, None, device)
    
    flow_model.eval()
    with torch.no_grad():
        output, mask_, S_pred, x1 = trainer.generate_on_timesteps(data, timesteps)
    with open(output_path, 'w') as f:
        for b in range(S_pred.shape[0]):
            S = S_pred[b]
            f.write(f'>generated_{b}\n')
            f.write(''.join(['AUCG'[i] for i in S.cpu().numpy()]) + '\n')


if __name__ == '__main__':
    main()