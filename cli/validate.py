import torch
from core.opt import Opts

from pathlib import Path
from core.augmentations import TRANSFORM_REGISTRY
from pkg.models import MODEL_REGISTRY
from pkg.datasets import DATASET_REGISTRY
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pkg.metrics.segmentation import SMAPIoUMetric
import lightning.pytorch as pl
import rich
from typing import Any
from pathlib import PosixPath
import ttach as tta

def move_to(obj: Any, device: torch.device):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to device
        device {torch.device} -- Device that object will be moved to
    Raises:
        TypeError: object is of type that is not implemented to process
    Returns:
        type(obj) -- same object but moved to specified device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        res = {k: move_to(v, device) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, PosixPath):
        return obj
    else:
        try:
            return obj.to(device)
        except:
            raise TypeError(f"Invalid type for move_to {type(obj)}")
        
def check(cfg, pretrained_ckpt=None):
    model = MODEL_REGISTRY.get(cfg.model["name"]).load_from_checkpoint(pretrained_ckpt,
                                       config=cfg,
                                       strict=True)
    images = []
    anns = []
    preds = []
    image_transform_test = TRANSFORM_REGISTRY.get('test_classify_tf')()
    ds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg["data"]["args"]["val"],
        data_cfg=cfg["data"]["args"],
        transform=image_transform_test,
    )

    dl = DataLoader(ds,batch_size=1, collate_fn=ds.collate_fn, num_workers=20)
    model.eval()
    device = model.device
    
    model = tta.SegmentationTTAWrapper(model, tta.aliases.nothing(), merge_mode='mean', output_mask_key='msk') # does not affect to model performance
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl), total=len(dl), desc="Validating"):
            img, ann = batch['images'][0], batch['labels'][0]
            pred = model(move_to(img.unsqueeze(0), device))['msk']
            pred = pred.detach().cpu().numpy().squeeze()
            images.append(img.cpu().numpy().transpose(1, 2, 0))
            anns.append(ann.cpu().numpy())
            preds.append(pred)

    metrics = SMAPIoUMetric()
    for ann, pred in zip(anns, preds):
        metrics.process(input={"gt": ann, "pred": pred})

    rich.print(metrics.compute_metrics(metrics.results))
    del model

def main():
    cfg = Opts().parse_args()
    check(cfg, pretrained_ckpt=cfg['global']['pretrained'])

if __name__ == "__main__":
    main()