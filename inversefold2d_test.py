import sys

from models import rfamflow,inversefold2d
from modules.inv3d.RNAMPNN import RNAMPNN
import torch
from ml_collections import ConfigDict
if __name__ == '__main__':
    #main_generate()
    # inversefold2d.main_single_gpu()
    rfamflow.main()
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
    print(RNAMPNN(ConfigDict(config)).load_state_dict(torch.load("./ckpts/inv3dcondition.pth")))
    pass