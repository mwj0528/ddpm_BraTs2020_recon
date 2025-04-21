import os
import yaml
import torch
from tqdm import tqdm

from datasets import BraTST1Dataset
from models import ContextUnet
from models import DDPM
from utils import get_logger_and_writer
from utils import save_tensor_image


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_samples(config_path="config/sample_config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger, writer = get_logger_and_writer("sample", save_dir=config["log_dir"])

    model = ContextUnet(in_channels=1, base_channels=256).to(device)
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.eval()

    ddpm = DDPM(model, timesteps=config["n_steps"], beta_start=config["beta_start"], beta_end=config["beta_end"], device=device)

    os.makedirs(config["sample_dir"], exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(config["n_samples"]), desc="Sampling"):
            sample = ddpm.sample(img_size = (1,128,128))
            save_path = os.path.join(config["sample_dir"], f"generated_{i}.png")
            save_tensor_image(sample, save_path)
            writer.add_image(f"Generated/{i}", sample[0], i)
            logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    generate_samples()
