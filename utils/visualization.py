import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# def denormalize(tensor):
#     """
#     [-1, 1] 범위의 tensor를 [0, 1]로 변환
#     """
#     img = ((tensor + 1) / 2.0).clamp(0, 1)
#     return img

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def save_tensor_image(tensor, save_path):
    """
    Tensor (1, H, W) or (H, W) → .png 저장 (정규화 해제 포함)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = denormalize(tensor).squeeze().cpu()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_comparison(gt_tensor, gen_tensor, save_path=None):
    """
    GT vs 생성 이미지 비교 시각화 (정규화 해제 포함)
    """
    gt = denormalize(gt_tensor).squeeze().cpu().clamp(0, 1)
    gen = denormalize(gen_tensor).squeeze().cpu().clamp(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gt, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(gen, cmap='gray')
    axes[1].set_title("Generated")
    axes[1].axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
