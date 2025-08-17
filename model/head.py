
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# Prediction Head of PLM_RankReg 
class PLM_RankReg(nn.Module):
    def __init__(self, input_dim, output_dim=1, input_hidden_dim=256, hidden_dim=128,
                 model_type='mlp', num_heads=4, dropout=0):
        super().__init__()
        self.model_type = model_type
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, input_hidden_dim),
            nn.Dropout(dropout/2)
        )
        
        if model_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(input_hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), 
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        elif model_type == 'cnn':
            self.head = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding='same'),
                nn.LayerNorm([hidden_dim, input_hidden_dim]), 
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding='same'),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        elif model_type == 'light_attention':
            self.attn = nn.MultiheadAttention(
                embed_dim=input_hidden_dim,
                num_heads=num_heads,
                batch_first=True, 
                dropout=dropout
            )
            self.attn_norm = nn.LayerNorm(input_hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(input_hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                for p in m.parameters():
                    if p.dim() > 1:
                        init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_layer(x)
        
        if self.model_type == 'mlp':
            return self.head(x)
        
        elif self.model_type == 'cnn':
            x = x.unsqueeze(1)
            return self.head(x)
        
        elif self.model_type == 'light_attention':
            identity = x
            x = x.unsqueeze(1)
            attn_out, _ = self.attn(x, x, x)
            x = self.attn_norm(identity + attn_out.squeeze(1))
            return self.head(x)
