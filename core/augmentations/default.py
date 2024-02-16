import cv2
from albumentations import (
    CLAHE,
    Blur,
    ChannelShuffle,
    ColorJitter,
    Compose,
    Downscale,
    ElasticTransform,
    Emboss,
    GaussianBlur,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    HueSaturationValue,
    IAASharpen,
    LongestMaxSize,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    OpticalDistortion,
    PadIfNeeded,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    RandomCrop,
    RandomGamma,
    RandomResizedCrop,
    RandomRotate90,
    RandomSizedCrop,
    Resize,
    RGBShift,
    Sharpen,
    ShiftScaleRotate,
    SmallestMaxSize,
    ToGray,
    VerticalFlip,
)

# https://github.com/albumentations-team/albumentations/issues/1246
cv2.setNumThreads(0)
from albumentations.pytorch.transforms import ToTensorV2

from . import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(RandomCrop, prefix="Alb")
TRANSFORM_REGISTRY.register(RGBShift, prefix="Alb")
TRANSFORM_REGISTRY.register(Normalize, prefix="Alb")
TRANSFORM_REGISTRY.register(Resize, prefix="Alb")
TRANSFORM_REGISTRY.register(Compose, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomBrightnessContrast, prefix="Alb")
TRANSFORM_REGISTRY.register(ShiftScaleRotate, prefix="Alb")
TRANSFORM_REGISTRY.register(SmallestMaxSize, prefix="Alb")
TRANSFORM_REGISTRY.register(MotionBlur, prefix="Alb")
TRANSFORM_REGISTRY.register(GaussianBlur, prefix="Alb")
TRANSFORM_REGISTRY.register(MedianBlur, prefix="Alb")
TRANSFORM_REGISTRY.register(Blur, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomRotate90, prefix="Alb")
TRANSFORM_REGISTRY.register(HorizontalFlip, prefix="Alb")
TRANSFORM_REGISTRY.register(VerticalFlip, prefix="Alb")
TRANSFORM_REGISTRY.register(HueSaturationValue, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomSizedCrop, prefix="Alb")
TRANSFORM_REGISTRY.register(IAASharpen, prefix="Alb")
TRANSFORM_REGISTRY.register(ToTensorV2, prefix="Alb")
TRANSFORM_REGISTRY.register(OneOf, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomContrast, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomGamma, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomBrightness, prefix="Alb")
TRANSFORM_REGISTRY.register(ElasticTransform, prefix="Alb")
TRANSFORM_REGISTRY.register(GridDistortion, prefix="Alb")
TRANSFORM_REGISTRY.register(OpticalDistortion, prefix="Alb")
TRANSFORM_REGISTRY.register(CLAHE, prefix="Alb")
TRANSFORM_REGISTRY.register(Downscale, prefix="Alb")
TRANSFORM_REGISTRY.register(LongestMaxSize, prefix="Alb")
TRANSFORM_REGISTRY.register(PadIfNeeded, prefix="Alb")
TRANSFORM_REGISTRY.register(ToGray, prefix="Alb")
TRANSFORM_REGISTRY.register(ChannelShuffle, prefix="Alb")
TRANSFORM_REGISTRY.register(ColorJitter, prefix="Alb")
TRANSFORM_REGISTRY.register(RandomResizedCrop, prefix="Alb")
TRANSFORM_REGISTRY.register(GaussNoise, prefix="Alb")
TRANSFORM_REGISTRY.register(Sharpen, prefix="Alb")
TRANSFORM_REGISTRY.register(Emboss, prefix="Alb")


@TRANSFORM_REGISTRY.register()
def train_classify_tf(img_size: int):
    return Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.3),
            OneOf(
                [
                    HueSaturationValue(p=0.3),
                    RandomGamma(p=0.3),
                    RGBShift(p=0.3),
                ],
                p=0.3,
            ),
            OneOf(
                [
                    OneOf(
                        [
                            GaussianBlur(blur_limit=3, p=0.5),
                            MotionBlur(blur_limit=3, p=0.5),
                            Downscale(scale_min=0.5, scale_max=0.9, p=0.5),
                        ],
                        p=0.3,
                    ),
                ],
                p=0.3,
            ),
            OneOf(
                [
                    ToGray(p=0.5),
                    ChannelShuffle(p=0.5),
                    ColorJitter(p=0.5),
                ],
                p=0.3,
            ),
            OneOf(
                [
                    Sharpen(p=0.3),
                    Emboss(p=0.3),
                    GaussNoise(p=0.3),
                ],
                p=0.3,
            ),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.3
            ),
            OneOf(
                [
                    RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0), p=1.0),
                    # Compose([
                    #         LongestMaxSize(img_size, p=1.0),
                    #         PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    #     ], p=1.0),
                    Resize(img_size, img_size, p=1.0),
                ],
                p=1.0,
            ),
            CLAHE(clip_limit=4.0, always_apply=True, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


@TRANSFORM_REGISTRY.register()
def valid_classify_tf(img_size: int = 256, aug: bool = False):
    if aug:
        return Compose(
            [
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return Compose(
            [
                Resize(img_size, img_size),
                CLAHE(clip_limit=4.0, always_apply=True, p=1.0),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


@TRANSFORM_REGISTRY.register()
def test_classify_tf(img_size=None, aug: bool = False):
    if aug:
        return Compose(
            [
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                CLAHE(clip_limit=4.0, always_apply=True, p=1.0),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        if img_size is None:
            return Compose(
                [
                    CLAHE(clip_limit=4.0, always_apply=True, p=1.0),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return Compose(
                [
                    Resize(img_size, img_size),
                    CLAHE(clip_limit=4.0, always_apply=True, p=1.0),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
