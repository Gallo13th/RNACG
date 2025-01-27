from models import inversefold3d, rfamflow
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Inverse RNA 3D folding')
    parser.add_argument('--input', type=str, help='Input RNA PDB file')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--task', type=str, default='rfam', help='Task: rfam or inversefold')
    parser.add_argument('--mode', type=str, default='generate', help='Mode: generate or predict')
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of time steps for generation')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--model_config', type=str, default='config.yaml', help='Configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    if args.task == 'rfam':
        rfamflow.main(args)
    elif args.task == 'inversefold':
        inversefold3d.main(args)
    else:
        raise ValueError('Unknown task: {}'.format(args.task)+'\nChoose rfam or inversefold')

