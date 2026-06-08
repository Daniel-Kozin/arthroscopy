"""Loss functions for stiffness regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Standard MSE on normalised predictions and labels."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class HuberLoss(nn.Module):
    """Huber loss — less sensitive to outliers than MSE."""
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class RankingLoss(nn.Module):
    """
    Encourages the model to get the *ordering* of zone stiffnesses right,
    in addition to the absolute values.

    Combined loss: alpha * MSE + (1 - alpha) * pairwise_ranking_loss
    """
    def __init__(self, alpha: float = 0.8, margin: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        # Pairwise ranking: for each pair (i, j) where target[i] > target[j],
        # penalise if pred[i] < pred[j] + margin
        B, N = pred.shape
        rank_loss = torch.tensor(0.0, device=pred.device)
        count = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    diff_target = target[:, i] - target[:, j]
                    diff_pred = pred[:, i] - pred[:, j]
                    # Only penalise when target says i > j but pred disagrees
                    mask = (diff_target > self.margin).float()
                    loss_ij = mask * F.relu(self.margin - diff_pred)
                    rank_loss = rank_loss + loss_ij.mean()
                    count += 1

        if count > 0:
            rank_loss = rank_loss / count

        return self.alpha * mse + (1 - self.alpha) * rank_loss
