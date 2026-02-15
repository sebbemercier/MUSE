# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

class MuseSLM(nn.Module):
    """
    SLM dédié à la génération de texte SEO.
    Architecture: Tiny-GPT (Decoder-only).
    """
    def __init__(self, vocab_size=5000, embed_dim=256, n_head=8, n_layer=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(512, embed_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, batch_first=True)
        # On utilise TransformerEncoder avec un masque causal pour simuler un décodeur
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Masque causal pour la génération
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        
        logits = self.lm_head(x)
        return logits