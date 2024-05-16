"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
"""

import time

import torch
from torch import nn
from torchsummary import summary


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck: bool = False):
        """
        Initializes the Conv3DBlock module which are double 3x3x3 convolutions.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck (bool, optional): If True, uses bottleneck architecture.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels // 2, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.conv2 = nn.Conv3d(
            out_channels // 2, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck

        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(x)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res

        return out, res


class UpConv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        last_layer: bool = False,
        num_classes: int | None = None,
    ):
        """
        Initializes the UpConv3DBlock module with the specified input and residual channels,
        along with optional settings for the last layer and the number of classes.

        Parameters:
            in_channels (int): Number of input channels.
            res_channels (int): Number of residual channels.
            last_layer (bool, optional): If True, indicates the last layer. Defaults to False.
            num_classes (int | None, optional): Number of classes. Defaults to None.
        """
        super().__init__()
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None
        ), "If `last_layer` is False, `num_classes` must be None. If `last_layer` is True, `num_classes` must be specified."

        self.upconv1 = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=(2, 2, 2), stride=2, padding=0
        )

        self.conv1 = nn.Conv3d(
            in_channels + res_channels,
            in_channels // 2,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels // 2,
            in_channels // 2,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )

        self.bn = nn.BatchNorm3d(in_channels // 2)
        self.relu = nn.ReLU()

        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels // 2,
                num_classes,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            )

    def forward(
        self, input: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.upconv1(input)

        if residual != None:
            out = torch.cat((out, residual), 1)

        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))

        if self.last_layer:
            out = self.conv3(out)

        return out


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        level_channels: list[int] = [64, 128, 256],
        bottleneck_channels: bool = 512,
    ):
        """
        Initializes the UNet3D model with the specified input and output channels, along with the level channels
        for downsampling and the bottleneck channels for bottleneck block.

        Parameters:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            level_channels (list[int]): List of channel sizes for each level. Defaults to [64, 128, 256].
            bottleneck_channels (bool, optional): Number of channels in the bottleneck block. Defaults to 512.
        """
        super().__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = (
            level_channels[0],
            level_channels[1],
            level_channels[2],
        )

        # Downsampling blocks
        self.a_block1 = Conv3DBlock(in_channels, level_1_chnls)
        self.a_block2 = Conv3DBlock(level_1_chnls, level_2_chnls)
        self.a_block3 = Conv3DBlock(level_2_chnls, level_3_chnls)

        # Bottleneck block
        self.bottleneck = Conv3DBlock(
            level_3_chnls, bottleneck_channels, bottleneck=True
        )

        # Upsampling blocks
        self.s_block3 = UpConv3DBlock(bottleneck_channels, level_3_chnls)
        self.s_block2 = UpConv3DBlock(level_3_chnls, level_2_chnls)
        self.s_block1 = UpConv3DBlock(
            level_2_chnls, level_1_chnls, last_layer=True, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, residual_level1 = self.a_block1(x)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)

        out, _ = self.bottleneck(out)

        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)

        return out


if __name__ == "__main__":
    # Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=3, num_classes=1)
    start_time = time.time()
    summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))
