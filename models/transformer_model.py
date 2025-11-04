"""
Custom Transformer Seq2Seq Model
Encoder-decoder architecture for refining LinkedIn post drafts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerPostOptimizer(nn.Module):
    """
    Transformer encoder-decoder model for LinkedIn post optimization.
    Takes draft posts as input and generates optimized versions.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        """
        Initialize Transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src: Source tokens (seq_len, batch_size)
            src_mask: Source mask
            
        Returns:
            Encoder output (seq_len, batch_size, d_model)
        """
        # Embed and add positional encoding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        return src
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode target sequence.
        
        Args:
            tgt: Target tokens (seq_len, batch_size)
            memory: Encoder output
            tgt_mask: Target mask (causal mask)
            memory_mask: Memory mask
            
        Returns:
            Decoder output (seq_len, batch_size, d_model)
        """
        # Embed and add positional encoding
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        
        return tgt
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the model.
        
        Args:
            src: Source tokens (seq_len, batch_size)
            tgt: Target tokens (seq_len, batch_size)
            src_mask: Source mask
            tgt_mask: Target mask
            memory_mask: Memory mask
            
        Returns:
            Output logits (seq_len, batch_size, vocab_size)
        """
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def generate(self, src, max_length=200, temperature=1.0, top_k=50):
        """
        Generate optimized post from draft.
        
        Args:
            src: Source tokens (seq_len, 1)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated tokens
        """
        self.eval()
        device = src.device
        
        # Encode source
        with torch.no_grad():
            memory = self.encode(src)
            
            # Start with SOS token
            tgt = torch.zeros(1, 1, dtype=torch.long, device=device)
            
            generated = []
            for _ in range(max_length):
                # Create causal mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device)
                
                # Decode
                output = self.decode(tgt, memory, tgt_mask)
                
                # Get logits for last token
                logits = self.fc_out(output[-1, :, :]) / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=0)
                generated.append(next_token.item())
                
                # Stop at EOS token (assuming vocab_size - 1 is EOS)
                if next_token.item() == self.vocab_size - 1:
                    break
        
        return generated


def create_transformer_model(vocab_size: int, **kwargs):
    """
    Factory function to create transformer model.
    
    Args:
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        TransformerPostOptimizer model
    """
    return TransformerPostOptimizer(vocab_size, **kwargs)
