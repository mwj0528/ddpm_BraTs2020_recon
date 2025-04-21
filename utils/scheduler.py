from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

def get_scheduler(optimizer, mode="cosine", **kwargs):
    if mode == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 50))
    elif mode == "step":
        return StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.5))
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")
