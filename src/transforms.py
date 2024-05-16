from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ToTensord,
)

# Transforms to be applied on training instances
train_transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ]
)

# Cuda version of "train_transform"
train_transforms_cuda = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=["image", "label"], device="cuda"),
    ]
)

# Transforms to be applied on validation instances
val_transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ]
)

# Cuda version of "val_transform"
val_transforms_cuda = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=["image", "label"], device="cuda"),
    ]
)
