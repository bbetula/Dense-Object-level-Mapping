import torch
import torch.nn as nn
import timm

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class SimpleFeaturePyramid(nn.Module):
    """
    ViTDet-style Simple Feature Pyramid.
    Takes single-scale ViT features at stride 16 and produces multi-scale features.
    """

    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        # res2 (stride 4): 4x upsample via two ConvTranspose2d(stride=2)
        self.fpn_stride4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
        )
        # res3 (stride 8): 2x upsample
        self.fpn_stride8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
        )
        # res4 (stride 16): 1x1 projection
        self.fpn_stride16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )
        # res5 (stride 32): 2x downsample
        self.fpn_stride32 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return {
            "res2": self.fpn_stride4(x),
            "res3": self.fpn_stride8(x),
            "res4": self.fpn_stride16(x),
            "res5": self.fpn_stride32(x),
        }


@BACKBONE_REGISTRY.register()
class D2DINOv3ViTL(Backbone):
    """
    Detectron2 backbone wrapper for DINOv3 ViT-Large (timm Eva model)
    with Simple Feature Pyramid for multi-scale output.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        weights_path = cfg.MODEL.DINOV3.WEIGHTS
        out_dim = cfg.MODEL.DINOV3.OUT_DIM
        freeze_backbone = cfg.MODEL.DINOV3.FREEZE_BACKBONE

        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )

        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.embed_dim = 1024
        self.patch_size = 16
        # 1 CLS + 4 register tokens
        self.num_prefix_tokens = self.backbone.num_prefix_tokens

        self.sfp = SimpleFeaturePyramid(in_dim=self.embed_dim, out_dim=out_dim)

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {k: out_dim for k in self._out_features}
        self._out_feature_strides = {
            "res2": 4, "res3": 8, "res4": 16, "res5": 32,
        }

    def forward(self, x):
        B, _, H, W = x.shape

        with torch.no_grad():
            features = self.backbone.forward_features(x)

        patch_tokens = features[:, self.num_prefix_tokens:, :]

        h_tokens = H // self.patch_size
        w_tokens = W // self.patch_size
        feature_map = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, h_tokens, w_tokens
        )

        return self.sfp(feature_map)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        return self
