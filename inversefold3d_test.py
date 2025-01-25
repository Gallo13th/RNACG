from models import inversefold3d
import argparse
import hydra

def parser_args():
    parser = argparse.ArgumentParser(description='Inverse RNA 3D folding')
    parser.add_argument('--input', type=str, help='Input RNA PDB file')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of time steps for generation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--model_config', type=str, default='config.yaml', help='Configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    hydra.initialize_config_dir(config_dir='config')
    inversefold3d.main(args)
