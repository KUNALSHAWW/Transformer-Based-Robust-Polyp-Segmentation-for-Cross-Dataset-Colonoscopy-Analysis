"""Models.
 decoder='unet' : smp.Unet with a MiT-B3 encoder (the original recipe; baseline).
 decoder='roy'  : MiT-B3 encoder + decoder built from the supervisor's published design (LwMLA-Net, IEEE TIM 2022):
                  multi-level attention (MLA) on every skip connection, depth-wise separable convolutions, ASPP in the bottleneck.
                  Flags: mla / aspp (switch the parts off for ablations).
 size head (Idea 2): predicts t = (log10(area)+3)/3 from the bottleneck; size_gate=True lets t gate the ASPP branches.
forward(x, t_gt=None) -> (logits, t_hat or None); t_gt is only used when oracle=True (upper-bound ablation)."""
import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder


class DSConv(nn.Module):
    """depth-wise 3x3 + point-wise 1x1 + BN + ReLU"""
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False); self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class MLA(nn.Module):
    """Multi-level attention as written in the supervisor's paper (Eq. 1-8):
    channel attention (avg+max pooled, shared MLP) -> spatial attention (7x7 on avg/max over channels) on F;
    pixel attention (1x1 conv, ReLU, 1x1 conv, sigmoid) on F; output = (SA(CA(F))) * (PA(F))."""
    def __init__(self, c, ratio=16):
        super().__init__()
        h = max(c // ratio, 8)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.sa = nn.Conv2d(2, 1, 7, padding=3)
        self.pa = nn.Sequential(nn.Conv2d(c, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 1))

    def forward(self, f):
        ca = torch.sigmoid(self.mlp(f.mean((2, 3)) + f.amax((2, 3))))[:, :, None, None]
        f1 = f * ca
        sa = torch.sigmoid(self.sa(torch.cat([f1.mean(1, keepdim=True), f1.amax(1, keepdim=True)], 1)))
        f2 = f1 * sa
        return f2 * (torch.sigmoid(self.pa(f)) * f)


class ASPP(nn.Module):
    """1x1 + three dilated depth-wise separable 3x3 (rates 3, 6, 12) + global pooling, concatenated, 1x1 projection.
    gate (B,5) optionally rescales the five branches (size-conditioned)."""
    def __init__(self, cin, cout, rates=(3, 6, 12)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.bs = nn.ModuleList([DSConv(cin, cout, r) for r in rates])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(5 * cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, gate=None):
        outs = [self.b0(x)] + [b(x) for b in self.bs] + [self.gp(x).expand(-1, -1, x.shape[2], x.shape[3])]
        if gate is not None:
            outs = [o * gate[:, i, None, None, None] for i, o in enumerate(outs)]
        return self.proj(torch.cat(outs, 1))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, mla):
        super().__init__()
        self.att = MLA(cskip) if (cskip and mla) else (nn.Identity() if cskip else None)
        self.blk = nn.Sequential(DSConv(cin + cskip, cout), DSConv(cout, cout))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, self.att(skip)], 1)
        return self.blk(x)


class PolypNet(nn.Module):
    def __init__(self, decoder="unet", size_head=False, size_gate=False, mla=True, aspp=True, encoder="mit_b3", pretrained=True, oracle=False):
        super().__init__()
        self.decoder_type, self.oracle = decoder, oracle
        if decoder == "unet":
            self.net = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1, activation=None)
            self.size = nn.Sequential(nn.Linear(self.net.encoder.out_channels[-1], 128), nn.ReLU(inplace=True), nn.Linear(128, 1)) if size_head else None
            assert not size_gate, "size gating needs the roy decoder (it gates the ASPP branches)"
            return
        self.encoder = get_encoder(encoder, in_channels=3, depth=5, weights="imagenet" if pretrained else None)
        c = self.encoder.out_channels                       # [3, 0, 64, 128, 320, 512]
        self.aspp = ASPP(c[5], 256) if aspp else DSConv(c[5], 256)
        self.has_aspp = aspp
        self.up1, self.up2, self.up3 = Up(256, c[4], 128, mla), Up(128, c[3], 64, mla), Up(64, c[2], 32, mla)
        self.up4, self.up5 = Up(32, 0, 16, mla), Up(16, 0, 16, mla)
        self.head = nn.Conv2d(16, 1, 1)
        self.size = nn.Sequential(nn.Linear(c[5], 128), nn.ReLU(inplace=True), nn.Linear(128, 1)) if size_head else None
        self.gate = None
        if size_gate:
            assert size_head and aspp, "size gating needs the size head and ASPP"
            self.gate = nn.Linear(1, 5); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)   # starts as identity

    def forward(self, x, t_gt=None):
        if self.decoder_type == "unet":
            feats = self.net.encoder(x)
            logits = self.net.segmentation_head(self.net.decoder(feats))
            t = torch.sigmoid(self.size(feats[-1].mean((2, 3)))) if self.size is not None else None
            return logits, t
        f = self.encoder(x)
        t = torch.sigmoid(self.size(f[5].mean((2, 3)))) if self.size is not None else None
        gate = None
        if self.gate is not None:
            t_use = t_gt if (self.oracle and t_gt is not None) else t
            gate = 1.0 + torch.tanh(self.gate(t_use))                     # in (0, 2); 1 at init
        b = self.aspp(f[5], gate) if self.has_aspp else self.aspp(f[5])
        d = self.up3(self.up2(self.up1(b, f[4]), f[3]), f[2])
        d = self.up5(self.up4(d))
        return self.head(d), t


def param_groups(model, lr_enc, lr_dec):
    enc = [p for n, p in model.named_parameters() if ".encoder." in "." + n or n.startswith("encoder.") or n.startswith("net.encoder.")]
    ids = {id(p) for p in enc}
    return [{"params": enc, "lr": lr_enc}, {"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr_dec}]


VARIANTS = {   # name -> (model kwargs, loss/training flags)
    "base":        (dict(decoder="unet"), dict()),
    "roy_plain":   (dict(decoder="roy", mla=False, aspp=False), dict()),
    "roy":         (dict(decoder="roy"), dict()),
    "roy_size":    (dict(decoder="roy", size_head=True, size_gate=True), dict(size_loss=True, size_tversky=True)),
    "roy_overlay": (dict(decoder="roy"), dict(overlay="pair")),
    "roy_full":    (dict(decoder="roy", size_head=True, size_gate=True), dict(size_loss=True, size_tversky=True, overlay="pair")),
    "roy_overlay_aug": (dict(decoder="roy"), dict(overlay="aug")),
    "roy_size_gate_only": (dict(decoder="roy", size_head=True, size_gate=True), dict(size_loss=True)),
    "roy_size_loss_only": (dict(decoder="roy", size_head=True), dict(size_loss=True, size_tversky=True)),
    # smaller encoders (efficiency study): same recipe, MiT-B0 / B1 / B2 instead of B3
    "base_b0": (dict(decoder="unet", encoder="mit_b0"), dict()),
    "base_b1": (dict(decoder="unet", encoder="mit_b1"), dict()),
    "base_b2": (dict(decoder="unet", encoder="mit_b2"), dict()),
    "roy_b0": (dict(decoder="roy", encoder="mit_b0"), dict()),
    "roy_b1": (dict(decoder="roy", encoder="mit_b1"), dict()),
    # tiny-polyp experiments: multi-scale training (0.75, 1, 1.25 x size) and a higher input size
    "base_ms": (dict(decoder="unet"), dict(ms=True)),
    "base_512": (dict(decoder="unet"), dict(size=512)),
    # augmentations chosen from the research (colour exchange: SANet; zoom-out: size mismatch between Kvasir and the small-polyp test sets)
    "base_lab": (dict(decoder="unet"), dict(lab=True)),
    "base_zoom": (dict(decoder="unet"), dict(zoom=True)),                          # zoom-out on a natural (blurred-picture) background
    "base_zoom_black": (dict(decoder="unet"), dict(zoom=True, zoom_bg="black")),    # control: the first version with a black canvas
    "base_zoom_match": (dict(decoder="unet"), dict(zoom="match")),                  # scale-matched: shrink factor chosen from the polyp size
    "base_lab_zoom": (dict(decoder="unet"), dict(lab=True, zoom=True)),
    "base_pid": (dict(decoder="unet"), dict(pid_dice=True)),                        # per-image Dice instead of the batch-pooled Dice
    # distillation: small student learns from the base (MiT-B3) model of the same fold
    "base_b0_kd": (dict(decoder="unet", encoder="mit_b0"), dict(kd="base")),
    "base_b1_kd": (dict(decoder="unet", encoder="mit_b1"), dict(kd="base")),
    "base_b2_kd": (dict(decoder="unet", encoder="mit_b2"), dict(kd="base")),
    "roy_b1_kd": (dict(decoder="roy", encoder="mit_b1"), dict(kd="base")),
}
