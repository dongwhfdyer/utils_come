import os
from typing import Callable

import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        out_features: int | None = None,
        criterion: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "CrossEntropyLoss",
    ):
        super().__init__()
        out_features = out_features or in_features
        self.ln = nn.LayerNorm(in_features)
        # Cosine-style linear probe: no bias, use normalized inputs and weights
        self.fc = nn.Linear(in_features, out_features, bias=False)
        # Learnable logit scale to control margin (initialized to 1.0)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.criterion = getattr(nn, criterion)() if isinstance(criterion, str) else criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, return_loss: bool = False):
        # Normalize features to align with cosine geometry
        x = self.ln(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        # Normalize classifier weights and compute cosine logits
        weight = torch.nn.functional.normalize(self.fc.weight, p=2, dim=1)
        x = self.logit_scale * (x @ weight.t())
        if y is not None and return_loss:
            return self.criterion(x, y)
        return x, y
