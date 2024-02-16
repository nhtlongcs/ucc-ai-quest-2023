import torch
import torch.nn.functional as F
import torchvision

from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel
from core.utils.smooth_losses import MultiLoss


@MODEL_REGISTRY.register()
class DeeplabV3ResNet101(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            num_classes=self.cfg.model["args"]["NUM_CLASS"]
        )
        self.loss = MultiLoss(weights=dict(self.cfg.loss["weights"]),
                              smooth_factor=self.cfg.loss["smooth_factor"],
                              convert_to_onehot=True,
                              )

    def normalize_head(
            self,
            embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits = self.model(batch["images"])["out"]
        logits = self.normalize_head(logits)
        probs = torch.softmax(logits, dim=1)
        preds_msk = torch.argmax(probs, dim=1)
        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, return_dict=False, **kwargs):
        targets = batch["labels"].long()
        return self.loss(logits, targets, return_dict=return_dict)

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.cfg.model["args"]["freeze_epochs"]:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = True

        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss, loss_dict = self.compute_loss(**output, batch=batch, return_dict=True)
        loss = loss.mean()
        # 3. Update monitor
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=False, sync_dist=True, logger=True)

        return {"loss": loss, "log": {"train_loss": loss, **loss_dict}}

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        assert (
            batch["images"].shape[0] == 1
        ), "Batch size must be 1 for this model, sorry! ugly code"
        preds = self.forward(batch)
        # add meta data for inference stage
        sz = batch["sizes"][0]
        msk = preds["msk"][0].unsqueeze(0).unsqueeze(0).float()
        origin_msk = F.interpolate(msk, sz, mode="bilinear", align_corners=False).long()
        preds["msk"] = origin_msk.squeeze(0)
        preds.update({"paths": batch["paths"]})
        return preds

