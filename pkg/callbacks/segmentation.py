from enum import Enum
from typing import Any, Dict

# import lightning.pytorch as pl
import lightning.pytorch as pl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
import torch.nn.functional as F
from core.callbacks import CALLBACKS_REGISTRY
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms import functional as TFF

from .utils.colors import color_list
from .utils.visualizer import Visualizer


class VISUAL_TYPE(Enum):
    SCALAR = "scalar"
    FIGURE = "figure"
    TORCH_MODULE = "torch_module"
    TEXT = "text"


try:
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:
    pass


@CALLBACKS_REGISTRY.register()
class SemanticVisualizerCallbackWanDB(Callback):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.visualizer = Visualizer()

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = trainer.global_step
        self.exp = pl_module.logger
        valloader = pl_module.datamodule.valloader
        trainloader = pl_module.datamodule.trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        valset = valloader.dataset
        classnames = valset.classnames
        try:
            self.visualize_model(pl_module, train_batch)
        except:
            rich.print("Cannot log model architecture")
        self.visualize_gt(pl_module, train_batch, val_batch, iters, classnames)

    @torch.no_grad()
    def visualize_model(self, pl_module, batch):
        # Vizualize Model Graph
        rich.print("Visualizing architecture...")
        self.exp.watch(
            pl_module.get_model(),
            log="gradients",
            log_freq=100,
        )

    def visualize_gt(self, pl_module, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check
        """

        rich.print("Visualizing dataset...")
        images = train_batch["images"]
        masks = train_batch["labels"]

        batch = []

        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.cpu().numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)

            # # Pad the image and mask to the maximum height and width
            # img_show = F.pad(img_show, (0, max_width - img_show.shape[2], 0, max_height - img_show.shape[1]), value=0)
            # mask = F.pad(mask, (0, max_width - mask.shape[1], 0, max_height - mask.shape[0]), value=0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)

            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)

        # segmentation color legends
        patches = [
            mpatches.Patch(color=np.array(color_list[i][::-1]), label=classnames[i])
            for i in range(len(classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)
        tag = "Sanitycheck/batch/train"
        self.exp.log_image(key=tag, images=[fig], caption=["plants"], step=iters)

        # Validation
        images = val_batch["images"]
        masks = val_batch["labels"]

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.cpu().numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        tag = "Sanitycheck/batch/val"
        self.exp.log_image(key=tag, images=[fig], caption=["plants"], step=iters)

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.params = {}
        self.params["last_batch"] = batch

    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        iters = trainer.global_step
        last_batch = self.params["last_batch"]
        model = pl_module
        valloader = pl_module.datamodule.valloader

        # Vizualize model predictions
        rich.print("Visualizing model predictions...")

        model.eval()

        images = last_batch["images"]
        masks = last_batch["labels"]
        # w = last_batch["origin_w"]
        # h = last_batch["origin_h"]
        rich.print("Last batch size is...", len(last_batch))
        preds = pl_module.predict_step(last_batch)["msk"]

        batch = []
        for idx, (inputs, mask, pred) in enumerate(zip(images, masks, preds)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.cpu().numpy())
            decode_pred = self.visualizer.decode_segmap(pred.cpu().numpy())

            img_cam = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            decode_pred = TFF.to_tensor(decode_pred / 255.0)

            # img_cam = F.interpolate(img_cam, size=(w, h), mode='bilinear')
            # decode_mask = F.interpolate(decode_mask, size=(w, h), mode='bilinear')
            # decode_pred = F.interpolate(decode_pred, size=(w, h), mode='bilinear')

            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.title("Raw image - Prediction - Ground Truth")
        plt.imshow(grid_img)

        # segmentation color legends
        classnames = valloader.dataset.classnames
        patches = [
            mpatches.Patch(color=np.array(color_list[i][::-1]), label=classnames[i])
            for i in range(len(classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)
        tag = "Validation/prediction"
        self.exp.log_image(key=tag, images=[fig], caption=["plants"], step=iters)

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()
