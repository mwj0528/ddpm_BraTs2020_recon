import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from datasets import BraTST1Dataset
from models import ContextUnet
from models import DDPM
from utils import get_logger_and_writer
from utils import get_scheduler
from utils import save_tensor_image


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_ddpm(config_path="config/train_config.yaml"):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger_and_writer("train", save_dir=config["log_dir"])

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1, 1] 범위로
    ])

    train_dataset = BraTST1Dataset(data_dir=config["data_dir"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    model = ContextUnet(in_channels=1, base_channels=256).to(device)
    ddpm = DDPM(model, timesteps=config["n_steps"], beta_start=config["beta_start"], beta_end=config["beta_end"], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = get_scheduler(optimizer, mode="cosine", T_max=config["epochs"])

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["sample_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config['epochs']}]")

        for images in pbar:
            images = images.to(device)
            loss = ddpm.train_step(images, optimizer)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            sampled = ddpm.sample(img_size=(1, 128,128))
            save_path = os.path.join(config["sample_dir"], f"sample_epoch{epoch+1}.png")
            save_tensor_image(sampled, save_path)
            writer.add_image("Sample", sampled[0], epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config["save_dir"], f"{epoch+1}_checkpoint.pth"))

    torch.save(model.state_dict(), os.path.join(config["save_dir"], "last.pth"))
    logger.info("Training complete and model saved.")


if __name__ == "__main__":
    train_ddpm()
