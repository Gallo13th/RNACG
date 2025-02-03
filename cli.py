from models import inversefold3d, rfamflow
import argparse
import os

def validate_args(args):
    """参数校验逻辑"""
    if args.task == 'inversefold':
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file {args.input} not exists")
    if args.task not in ["rfam", "inversefold"]:
        raise ValueError(f"We don't support task: {args.task}")
    if args.n_samples < 1:
        raise ValueError("n_samples must be greater than 0")
    return args

def parser_args():
    """模块化参数解析"""
    parser = argparse.ArgumentParser(description='RNA Conditional Generation')
    
    # 必需参数
    io_group = parser.add_argument_group("IO Configuration")
    io_group.add_argument('--input', required=True, type=str, 
                        help='Input PDB file path (inversefold) or RFAM ID (rfam)')
    io_group.add_argument('--output', required=True, type=str,
                        help='Output file path')

    # 任务参数
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument('--task', choices=['rfam', 'inversefold'], 
                          default='rfam', help='task category')
    task_group.add_argument('--mode', choices=['generate'],
                          default='generate', help='mode')

    # 模型参数
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument('--model', required=True, type=str,
                           help='path to model checkpoint')
    model_group.add_argument('--model_config', type=str, 
                           default='config.yaml', help='path to model config file')

    # 设备参数
    device_group = parser.add_argument_group("Device Configuration")
    device_group.add_argument('--device', choices=['cpu', 'cuda'], 
                            default='cuda', help='device to run the model, default: cuda')
    device_group.add_argument('--seed', type=int, default=0,
                            help='random seed')

    # 生成参数
    gen_group = parser.add_argument_group("Generate Configuration")
    gen_group.add_argument('--n_samples', type=int, default=1,
                         help='number of samples to generate')
    gen_group.add_argument('--n_steps', type=int, default=10,
                         help='number of steps to generate')
    gen_group.add_argument('--seq_length', type=int, default=100,
                         help='length of the generated sequence')

    return validate_args(parser.parse_args())

if __name__ == '__main__':
    try:
        args = parser_args()
        if args.task == 'rfam':
            rfamflow.main(args)
        elif args.task == 'inversefold':
            inversefold3d.main(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)