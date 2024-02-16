from core.models import MODEL_REGISTRY

# from .semi.cps import SemiSuperviseModel
# from .semi.hydra import DualClassifier
from .supervise.tf.dinov2 import DinoV2
from .supervise.tf.dpt import DPT

# transformer
from .supervise.tf.mask2former import Mask2former
from .supervise.tf.segformer import Segformer
from .supervise.unet.unet import Unet

# ConvNext
from .supervise.convnext_v2_sfnet.model import ConvNext_SFNet

# ConvNext_v1
from .supervise.convnext_v1_sfnet.model import ConvNextv1_SFNet

# RegNetY
from .supervise.regnety.regnety_model import RegNetY_UPerNet

# ConvNext_SSL

from .supervise.convnext_ssl_sfnet.model import ConvNextSSL_SFNet

# ResNet
from .supervise.resnet.deeplabv3_vicregl import DeeplabV3ResNet50VicregL
from .supervise.resnet.deeplabv3 import DeeplabV3ResNet101
from .supervise.resnet.resnet50 import SegResNet50
from .supervise.resnet.resnet101 import SegResNet101
from .supervise.resnet.resnet50_vicregl import ResNet50VicregL
from .supervise.resnet.resnet50_vicregl2 import ResNet50VicregL2
