"""Losses. Base recipe = Dice + BCE + 0.5 * Lovasz-hinge (as in the original project).
Size-aware variant (Idea 2): the Dice term becomes a per-image Tversky loss whose beta grows for small predicted polyps
(beta = 0.5 + 0.3 * (1 - min(area_hat / 0.15, 1)), alpha = 1 - beta), i.e. misses on small polyps are punished more."""
import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp


def _lovasz_grad(g):
    gts = g.sum(); inter = gts - g.float().cumsum(0); union = gts + (1 - g).float().cumsum(0)
    j = 1.0 - inter / union
    if len(g) > 1:
        j[1:] = j[1:] - j[:-1]
    return j


def _lovasz_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    errors = 1.0 - logits * (2.0 * labels.float() - 1.0)
    es, perm = torch.sort(errors, descending=True)
    return torch.dot(F.relu(es), _lovasz_grad(labels[perm.data]))


def lovasz_hinge(logits, labels):
    lg, lb = logits.squeeze(1), labels.squeeze(1).long()
    return torch.stack([_lovasz_flat(a.contiguous().view(-1), b.contiguous().view(-1)) for a, b in zip(lg, lb)]).mean()


def tversky_per_image(logits, y, beta, eps=1.0):
    p = torch.sigmoid(logits).flatten(1); t = y.flatten(1); beta = beta.view(-1, 1); alpha = 1.0 - beta
    tp, fp, fn = (p * t).sum(1), (p * (1 - t)).sum(1), ((1 - p) * t).sum(1)
    return (1.0 - (tp + eps) / (tp + alpha.squeeze(1) * fp + beta.squeeze(1) * fn + eps)).mean()


def beta_from_t(t_hat):
    """t_hat in [0,1] (log-area code) -> Tversky beta in [0.5, 0.8]; area = 10**(3t-3)."""
    area = 10 ** (3.0 * t_hat - 3.0)
    return 0.5 + 0.3 * (1.0 - torch.clamp(area / 0.15, max=1.0))


class SegLoss(nn.Module):
    def __init__(self, size_tversky=False, w_bce=1.0, w_lov=0.5):
        super().__init__()
        self.size_tversky, self.w_bce, self.w_lov = size_tversky, w_bce, w_lov
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y, t_hat=None):
        region = tversky_per_image(logits, y, beta_from_t(t_hat.detach().squeeze(1))) if (self.size_tversky and t_hat is not None) else self.dice(logits, y)
        return region + self.w_bce * self.bce(logits, y) + self.w_lov * lovasz_hinge(logits, y)
