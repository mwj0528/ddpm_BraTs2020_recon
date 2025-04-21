import logging
import os
from torch.utils.tensorboard import SummaryWriter

def get_logger_and_writer(name: str, save_dir: str = "./logs", filename: str = "train.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 콘솔 로그
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # 파일 로그
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))

    return logger, writer
