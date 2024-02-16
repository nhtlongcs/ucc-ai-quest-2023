import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    norm_map = {
        "none": nn.Identity,
        "batch": nn.BatchNorm2d,
        "instance": nn.InstanceNorm2d,
    }

    activation_map = {
        "none": nn.Identity,
        "relu": nn.ReLU,
    }

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        cfg,
        norm="batch",
        activation="relu",
    ):
        super().__init__()

        conv_cfg = {} if cfg.get("conv", None) is None else cfg["conv"]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_cfg)

        assert (
            norm in Conv2dBlock.norm_map.keys()
        ), "Chosen normalization method is not implemented."
        norm_cfg = {} if cfg.get("norm", None) is None else cfg["norm"]
        self.norm = Conv2dBlock.norm_map[norm](out_channels, **norm_cfg)

        assert (
            activation in Conv2dBlock.activation_map.keys()
        ), "Chosen activation method is not implemented."
        activation_cfg = (
            {} if cfg.get("activation", None) is None else cfg["activation"]
        )
        self.activation = Conv2dBlock.activation_map[activation](**activation_cfg)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class NestedUNetEncoderBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.cfg = {
            "conv": {"padding": 1},
            "activation": {"inplace": True},
        }

        self.down_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # print('Input:', x.shape)
        x = self.down_conv(x)
        pool = self.pool(x)
        # print('Down:', x.shape, pool.shape)
        return x, pool


class NestedUNetDecoderBlock(nn.Module):
    def __init__(
        self,
        inputs,
        concats,
        outputs,
        upsample_method="deconv",
        sizematch_method="interpolate",
    ):
        super().__init__()

        assert upsample_method in ["deconv", "interpolate"]
        if upsample_method == "deconv":
            self.upsample = nn.ConvTranspose2d(inputs, inputs, kernel_size=2, stride=2)
        elif upsample_method == "interpolate":
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

        assert sizematch_method in ["interpolate", "pad"]
        if sizematch_method == "interpolate":
            self.sizematch = self.sizematch_interpolate
        elif sizematch_method == "pad":
            self.sizematch = self.sizematch_pad

        self.cfg = {
            "conv": {"padding": 1},
            "activation": {"inplace": True},
        }

        self.conv = nn.Sequential(
            Conv2dBlock(concats, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )

    def sizematch_interpolate(self, source, target):
        return F.interpolate(
            source,
            size=(target.size(2), target.size(3)),
            mode="bilinear",
            align_corners=True,
        )

    def sizematch_pad(self, source, target):
        diffX = target.size()[3] - source.size()[3]
        diffY = target.size()[2] - source.size()[2]
        return F.pad(
            source, (diffX // 2, diffX - diffX // 2, diffY // 2, diffX - diffY // 2)
        )

    def forward(self, x, x_copy):
        x = self.upsample(x)
        x = self.sizematch(x, x_copy[0])
        x = torch.cat([*x_copy, x], dim=1)
        x = self.conv(x)
        return x


class _NestedUNet(nn.Module):
    def __init__(self, in_channels, nclasses, first_channels, depth):
        super().__init__()
        self.depth = depth
        conv = [[NestedUNetEncoderBlock(in_channels, first_channels)]]
        conv[0].extend(
            [
                NestedUNetEncoderBlock(
                    first_channels * 2 ** (d - 1), first_channels * 2**d
                )
                for d in range(1, self.depth)
            ]
        )
        conv.extend(
            [
                [
                    NestedUNetDecoderBlock(
                        first_channels * 2 ** (i + 1),
                        first_channels * (j + 2) * 2**i,
                        first_channels * 2**i,
                    )
                    for i in range(self.depth - j)
                ]
                for j in range(1, self.depth)
            ]
        )
        self.conv = nn.ModuleList([nn.ModuleList(layer) for layer in conv])
        self.final = nn.Conv2d(first_channels, nclasses, kernel_size=1)

    def forward(self, x):
        X = []
        for conv in self.conv[0]:
            ft, x = conv(x)
            X.append([ft])
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                X[i].append(self.conv[j][i](X[i + 1][j - 1], X[i]))
        x = self.final(X[0][-1])
        return x


from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel


@MODEL_REGISTRY.register()
class NestedUnet(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.model = _NestedUNet(
            in_channels=3,
            nclasses=self.cfg.model["args"]["NUM_CLASS"],
            first_channels=64,
            depth=4,
        )
        # self.loss = DiceLoss(num_classes=self.cfg.model["args"]["NUM_CLASS"])
        self.loss = nn.CrossEntropyLoss()

    def normalize_head(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits = self.model(batch["images"])  # torch.Size([2, 2, 380, 380])
        # logits = self.normalize_head(preds)
        probs = torch.softmax(logits, dim=1)
        preds_msk = torch.argmax(probs, dim=1)

        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["labels"].long())

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({"paths": batch["paths"]})
        return preds
