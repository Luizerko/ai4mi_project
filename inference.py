import argparse
from pathlib import Path
from operator import itemgetter
from typing import Tuple, List

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from utils import (
    class2one_hot,
    probs2class,
    tqdm_,
    save_images,
)

# Same build and transform as the training process to get best results
def build_transforms(K: int):
    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,
        lambda nd: torch.tensor(nd, dtype=torch.float32),
    ])

    return img_transform


# Loading best model
def load_model(args, device: torch.device):
    weights_path = Path(args.bweights / "bestmodel.pkl")
    loaded = torch.load(weights_path, map_location=device, weights_only=False)
    net = loaded.to(device)
    net.eval()
    return net

# Creating dataloader
def make_loader(dataset: str, K: int, B: int, subset: str):
    img_transform = build_transforms(K)
    root_dir = Path(dataset)
    subset_ds = SliceDataset(subset, root_dir, img_transform=img_transform)
    subset_loader = DataLoader(subset_ds, batch_size=B, num_workers=5, shuffle=False)
    return subset_loader

# Running inference
def inference(net: nn.Module, device: torch.device, loader: DataLoader, K: int, save_dir: Path):
    stems: List[str] = []

    with torch.no_grad():
        tq = tqdm_(enumerate(loader), total=len(loader), desc="Inference")
        for _, batch in tq:
            img: Tensor = batch['images'].to(device)
            stems.extend(batch['stems'])

            logits: Tensor = net(img)
            probs: Tensor = F.softmax(logits, dim=1)

            pred_cls = probs2class(probs)
            mult: int = 63 if K == 5 else int(255 / (K - 1))
            save_images(pred_cls * mult, batch['stems'], save_dir)

    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TOY2')
    parser.add_argument('--split', default='test')
    parser.add_argument('--outdir', type=Path, required=True, help="Output directory for predictions")
    parser.add_argument('--bweights', type=Path, help="Path to full model's dir (where we can find bestmodel.pkl)")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    # Getting device
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    # Setting parameters
    K = 5
    B = args.batch

    # Preparing I/O
    pred_dir = args.outdir
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Getting best model
    net = load_model(args, device)

    # Getting dataloader
    loader = make_loader(args.dataset, K, B, args.split)

    # Running inference
    inference(net=net, device=device, loader=loader, K=K, save_dir=pred_dir)


if __name__ == "__main__":
    main()
