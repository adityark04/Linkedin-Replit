"""
1D-CNN Model for LinkedIn Post Optimization
Focuses on local pattern recognition and content feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPostOptimizer(nn.Module):
    """
    1D-CNN model for LinkedIn post content generation.
    Uses convolutional layers to capture local patterns in text.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_filters: int = 256,
        kernel_sizes: list = [3, 4, 5],
        num_layers: int = 4,
        dropout: float = 0.3,
        max_seq_length: int = 512
    ):
        """
        Initialize 1D-CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes for conv layers
            num_layers: Number of CNN blocks
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple Conv1D layers with different kernel sizes
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels=embedding_dim if len(self.conv_layers) == 0 else num_filters,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k//2
                )
                for k in kernel_sizes
            ])
            self.conv_layers.append(convs)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters * len(kernel_sizes))
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers for generation
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), num_filters * 2)
        self.fc2 = nn.Linear(num_filters * 2, num_filters)
        self.fc_out = nn.Linear(num_filters, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(num_filters)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv_list in self.conv_layers:
            for conv in conv_list:
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass through the CNN model.
        
        Args:
            x: Input tokens (batch_size, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Output logits (batch_size, seq_len, vocab_size) or features
        """
        batch_size, seq_len = x.size()
        
        # Embed input tokens
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Transpose for Conv1d: (batch_size, embedding_dim, seq_len)
        x = embedded.transpose(1, 2)
        
        # Pass through CNN layers
        for i, conv_list in enumerate(self.conv_layers):
            # Apply multiple kernel sizes
            conv_outputs = []
            for conv in conv_list:
                conv_out = F.relu(conv(x))
                conv_outputs.append(conv_out)
            
            # Concatenate outputs from different kernel sizes
            x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(kernel_sizes), seq_len)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Dropout
            x = self.dropout(x)
        
        # Transpose back: (batch_size, seq_len, num_filters * len(kernel_sizes))
        x = x.transpose(1, 2)
        
        # Extract features if requested
        if return_features:
            return x
        
        # Fully connected layers for generation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Output projection
        logits = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(self, src, max_length=200, temperature=1.0, top_k=50):
        """
        Generate optimized post from draft using the CNN model.
        
        Args:
            src: Source tokens (batch_size, src_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated tokens
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Get features from source
        with torch.no_grad():
            # Start with source tokens
            current_seq = src.clone()
            generated = []
            
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(current_seq)  # (batch_size, seq_len, vocab_size)
                
                # Get logits for last position
                last_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = last_logits < torch.topk(last_logits, min(top_k, last_logits.size(-1)))[0][..., -1, None]
                    last_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                generated.append(next_token.squeeze().item())
                
                # Stop at EOS or max length
                if next_token.item() == self.vocab_size - 1:
                    break
                
                # Truncate if too long
                if current_seq.size(1) > self.max_seq_length:
                    current_seq = current_seq[:, -self.max_seq_length:]
        
        return generated
    
    def extract_features(self, x):
        """
        Extract CNN features for ensemble.
        
        Args:
            x: Input tokens (batch_size, seq_len)
            
        Returns:
            Extracted features
        """
        return self.forward(x, return_features=True)


def create_cnn_model(vocab_size: int, **kwargs):
    """
    Factory function to create CNN model.
    
    Args:
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        CNNPostOptimizer model
    """
    return CNNPostOptimizer(vocab_size, **kwargs)
