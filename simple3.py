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
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_features = out_features or in_features

        # Use a reasonable hidden dimension if not specified
        # Generally 2-4x the input dimension works well for audio features
        if hidden_dim is None:
            hidden_dim = min(max(in_features * 2, 512), 2048)

        # More sophisticated architecture with multiple layers and proper regularization
        self.network = nn.Sequential(
            # Input normalization only (no aggressive feature normalization)
            nn.LayerNorm(in_features),

            # First hidden layer
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second hidden layer (creates more representational capacity)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer with bias (important for classification)
            nn.Linear(hidden_dim // 2, out_features, bias=True)
        )

        self.criterion = getattr(nn, criterion)() if isinstance(criterion, str) else criterion

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization for better training dynamics"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, return_loss: bool = False):
        # Simple forward pass without excessive normalization
        logits = self.network(x)

        if y is not None and return_loss:
            return self.criterion(logits, y)
        return logits, y