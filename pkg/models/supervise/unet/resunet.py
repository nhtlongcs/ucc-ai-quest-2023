import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

# from losses import DiceLoss


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        with_nonlinearity=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
    ):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)

        dH = down_x.size()[2] - x.size()[2]
        dW = down_x.size()[3] - x.size()[3]
        x = F.pad(x, (dW // 2, dW - dW // 2, dH // 2, dH - dH // 2))

        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class _ResidUNet(nn.Module):
    DEPTH = 6

    def __init__(self, in_channels, nclasses):
        super().__init__()

        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=3, kernel_size=1, padding=0
            ),
            nn.Sequential(*list(resnet.children()))[:3],
        )
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bridge = Bridge(2048, 2048)

        up_blocks = []
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=128 + 64,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
        )
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=64 + in_channels,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, nclasses, kernel_size=1, stride=1)

        self.freeze_bn()

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (_ResidUNet.DEPTH - 1):
                continue
            pre_pools["layer_{}".format(i)] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = "layer_{}".format(_ResidUNet.DEPTH - 1 - i)
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel


@MODEL_REGISTRY.register()
class ResUnet(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.model = _ResidUNet(
            in_channels=3, nclasses=self.cfg.model["args"]["NUM_CLASS"]
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
