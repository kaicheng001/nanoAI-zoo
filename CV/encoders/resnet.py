import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Residual blocks
# ---------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        y = F.relu(y + x)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        mid = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.downsample is not None:
            identity = self.downsample(x)
        y = F.relu(y + identity)
        return y


# ---------------------------
# ResNet backbone (encoder-first design)
# ---------------------------


class ResNet(nn.Module):
    """
    Plug-and-play ResNet used as a vision encoder.

    Key properties:
      - Forward can return a token sequence [B, N, D] from the last feature map.
      - A 1x1 projection maps backbone channels -> embed_dim for downstream modules.
      - Optional 2D sinusoidal positional encoding is added on-the-fly (resolution-agnostic).
      - Also supports returning pooled features or the feature map.

    Token "patch size" (effective stride wrt input): 32 pixels
    (7x7 s=2 + maxpool s=2, then stage strides 1-2-2-2).
    """

    def __init__(
        self,
        block,
        layers,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        add_positional_encoding=True,
        add_cls_token=False,
    ):
        super().__init__()

        # stem
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # heads
        feat_channels = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feat_channels, num_classes)

        # projection to embedding dimension for token outputs
        self.proj = nn.Conv2d(feat_channels, embed_dim, kernel_size=1, bias=False)

        # cls token (optional, useful for ViT-style consumers)
        self.add_cls_token = add_cls_token
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if add_cls_token else None
        )

        # positional encoding generator flag (no parameters stored per resolution)
        self.add_positional_encoding = add_positional_encoding

        self._init_weights()

    # -------- public API ----------

    @torch.no_grad()
    def feature_map_stride(self) -> int:
        return 32  # effective patch stride wrt input

    def forward(
        self,
        images,
        output="tokens",
        return_dict=True,
    ):
        """
        Args:
          images: [B, 3, H, W]
          output: "tokens" | "map" | "pooled" | "logits"
          return_dict: if True and output=="tokens", returns a dict with keys:
              tokens: [B, N, D]
              mask:   [B, N] (all ones)
              hw:     (Hf, Wf) feature map size
              stride: int effective stride (32)

        """
        x = self._forward_backbone(images)  # [B, C, Hf, Wf]

        if output == "map":
            return x

        if output == "pooled":
            y = torch.flatten(self.avgpool(x), 1)  # [B, C]
            return y

        if output == "logits":
            y = torch.flatten(self.avgpool(x), 1)
            return self.classifier(y)

        # tokens path
        tokens, (hf, wf) = self._to_tokens(x)  # [B, N, D], (Hf, Wf)

        out = tokens
        if return_dict:
            mask = torch.ones(
                tokens.size(0), tokens.size(1), device=tokens.device, dtype=torch.bool
            )
            return {
                "tokens": out,
                "mask": mask,
                "hw": (hf, wf),
                "stride": self.feature_map_stride(),
            }
        return out

    # -------- internals ----------

    def _forward_backbone(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _to_tokens(self, feat_map):
        # project channels -> embed_dim, then flatten to tokens
        x = self.proj(feat_map)  # [B, D, Hf, Wf]
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D], N=Hf*Wf

        if self.add_positional_encoding:
            pe = build_2d_sincos_pos_embed(h, w, d, device=x.device)  # [1, N, D]
            x = x + pe

        if self.add_cls_token:
            cls = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
            x = torch.cat([cls, x], dim=1)  # N+1

        return x, (h, w)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.inplanes, out_channels, stride, downsample)]
        self.inplanes = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)


# ---------------------------
# Positional encoding (resolution-agnostic, no params)
# ---------------------------


def build_2d_sincos_pos_embed(h, w, dim, device=None):
    """
    Returns [1, H*W, D] sinusoidal embeddings constructed from 2D grids.
    Works for any spatial size at runtime (no learned table).
    """
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([yy, xx], dim=0)  # [2, H, W]

    d_half = dim // 2
    emb_y = _pos1d(yy.reshape(-1), d_half, device)  # [H*W, D/2]
    emb_x = _pos1d(xx.reshape(-1), dim - d_half, device)  # [H*W, D - D/2]
    emb = torch.cat([emb_y, emb_x], dim=1)  # [H*W, D]
    return emb.unsqueeze(0)  # [1, N, D]


def _pos1d(pos, dim, device):
    omega = torch.arange(dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if emb.shape[1] < dim:
        pad = torch.zeros(emb.shape[0], dim - emb.shape[1], device=device)
        emb = torch.cat([emb, pad], dim=1)
    return emb


# ---------------------------
# Factory helpers
# ---------------------------


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# ---------------------------
# Usage examples (encoder mode)
# ---------------------------


if __name__ == "__main__":
    import sys, pathlib, torch
    from PIL import Image
    from torchvision.datasets import CIFAR10

    # ensure project root is on sys.path
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from utils.load_data import load_single_image

    # -------------------------------
    # Single image test (assets/img.jpg)
    # -------------------------------
    print("\n--- Single Image Test ---")
    model = resnet50(embed_dim=768, add_positional_encoding=True, add_cls_token=False)
    single_image = load_single_image("assets/img.jpg", image_size=224)
    single_output = model(single_image, output="tokens")
    print("Input shape:", single_image.shape)
    print("Token shape:", single_output["tokens"].shape)
    print("Feature map (H, W):", single_output["hw"])
    print("Stride:", single_output["stride"])

    # -------------------------------
    # CIFAR10 batch test
    # -------------------------------
    print("\n--- CIFAR10 Batch Test ---")
    data_root = pathlib.Path(__file__).resolve().parents[2] / "data"
    dataset = CIFAR10(root=str(data_root), train=False, download=True)
    images = [Image.fromarray(dataset.data[i]) for i in range(8)]
    batch = torch.cat(
        [load_single_image(img, image_size=(32, 32)) for img in images], dim=0
    )
    batch_output = model(batch, output="tokens")
    print("Batch shape:", batch.shape)
    print("Token shape:", batch_output["tokens"].shape)
    print("Feature map (H, W):", batch_output["hw"])
    print("Stride:", batch_output["stride"])
