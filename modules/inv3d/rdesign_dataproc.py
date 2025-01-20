import torch
import numpy as np
import os
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import copy
import torch.utils.data as data

'''
This script is used to load the data from RDesign
'''
from collections.abc import Mapping, Sequence
def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    raise TypeError("Can't transfer object type `%s`" % type(obj))


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr

class AugDataset(data.Dataset):
    def __init__(self, path='./',  mode='train'):
        self.path = path
        self.mode = mode
        self.data = self.cache_data[mode]
    
    @cached_property
    def cache_data(self):
        alphabet_set = set(['A', 'U', 'C', 'G'])
        if os.path.exists(self.path):
            data_dict = {'train': [], 'val': [], 'test': []}
            # val and test data
            for split in ['train', 'val', 'test']:
                data = cPickle.load(open(os.path.join(self.path, split + '_data.pt'), 'rb'))
                for entry in tqdm(data):
                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)
                    bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                    if len(bad_chars) == 0:
                        data_dict[split].append(entry)
            data_dict['train'] = data_dict['train'] + data_dict['test']
            return data_dict
        else:
            raise "no such file:{} !!!".format(self.path)

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def find_bracket_pairs(ss, seq):
    pairs = []
    stack = []
    for i, c in enumerate(ss):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                pairs.append((stack.pop(), i))
            else:
                pairs.append((None, i)) 
    if stack:
        pairs.extend(zip(stack[::-1], range(i, i - len(stack), -1)))
        
    npairs = []
    for pair in pairs:
        if None in pair:
            continue
        p_a, p_b = pair
        if (seq[p_a], seq[p_b]) in (('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')):
            npairs.append(pair)
    return npairs

def featurize(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)
    
    ss_pair = []
    names = []
    
    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
        # Add by GLT 0111 
        # fix some missed atoms
        if np.isnan(x).any():
            b['name'] = b['name'] + '_missed'
            if not np.isnan(x).all():
                for res_j in range(x.shape[0]):
                    for atom_j in range(x.shape[1]):
                        x[res_j, atom_j] = np.nanmean(x[res_j, atom_j], axis=0)
                    if np.isnan(x[res_j]).all():
                        if res_j == 0:
                            x[res_j] = x[res_j+1] * 2 - x[res_j+2]
                        elif res_j == x.shape[0]-1:
                            x[res_j] = x[res_j-1] * 2 - x[res_j-2]
                        else:
                            x[res_j] = (x[res_j-1] + x[res_j+1]) / 2
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        ss_pos[i, :l] = np.asarray([1 if ss_val!='.' else 0 for ss_val in b['ss']], dtype=np.int32)
        ss_pair.append(find_bracket_pairs(b['ss'], b['seq']))
        names.append(b['name'])
        
        clus[i] = b['cluster']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    clus = torch.from_numpy(clus).to(dtype=torch.long)
    return X, S, mask, lengths, clus, ss_pos, ss_pair, names

def load_data(batch_size, data_root, num_workers=8, **kwargs):
    dataset = AugDataset(data_root, mode='train')
    train_set, valid_set, test_set = map(lambda x: copy.deepcopy(x), [dataset] * 3)
    valid_set.change_mode('val')
    test_set.change_mode('test')
    
    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=num_workers, collate_fn=featurize)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=featurize)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=featurize)
    return train_loader, valid_loader, test_loader


