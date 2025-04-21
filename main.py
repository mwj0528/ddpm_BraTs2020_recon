import argparse
import torch

from scripts import train_ddpm
from scripts import generate_samples

def parse_args():
    parser = argparse.ArgumentParser(description="BraTS T1 DDPM 프로젝트")
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], required=True,
                        help="실행 모드: 'train' 또는 'sample'")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="cuda 또는 cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'train':
        config_path = "config/train_config.yaml"
        train_ddpm(config_path=config_path)
    elif args.mode == 'sample':
        config_path = "config/sample_config.yaml"
        generate_samples(config_path=config_path)
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {args.mode}")

if __name__ == "__main__":
    main()

# python main.py --mode train
# python main.py --mode sample