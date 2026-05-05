import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class EnhancedAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.multihead_attn = MultiheadAttention(hidden_size, num_heads, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # MultiheadAttention expects: (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x.transpose(0, 1) + attn_output)
        return output

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.attention = EnhancedAttentionModule(hidden_size * 2)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = self.layer_norm1(lstm_out)
        
        # Enhanced attention
        attended = self.attention(lstm_out)
        
        # Residual connection and feed forward
        x = self.fc1(attended)
        x = F.gelu(x)  # Using GELU activation
        x = self.dropout(x)
        x = self.layer_norm2(x)
        
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        return x

class EnhancedTemporalCompressor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, lstm_features_batch):
        compressed_vectors = []
        
        for features in lstm_features_batch:
            # Ensure features has correct shape
            if len(features.shape) == 1:
                features = features.unsqueeze(0)  # Add batch dimension if needed
                
            # Compute attention scores
            attention_scores = self.attention(features)  # Shape: (seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=0)  # Shape: (seq_len, 1)
            
            # Apply attention and transform features
            weighted_sum = torch.matmul(attention_weights.transpose(0, 1), features)  # Shape: (1, hidden_size)
            transformed = self.feature_transform(weighted_sum)  # Shape: (1, hidden_size)
            compressed_vectors.append(transformed)
            
        return torch.cat(compressed_vectors, dim=0)

class FeatureFusionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x1, x2):
        # Concatenate features
        combined = torch.cat([x1, x2], dim=-1)
        # Transform and fuse
        fused = self.fusion_layer(combined)
        return fused