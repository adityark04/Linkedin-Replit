"""
GPT-Style Causal Language Model
Fine-tuned for LinkedIn post generation (replaces DistilGPT-2/T5-small due to library constraints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTAttention(nn.Module):
    """Multi-head self-attention for GPT."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        out = out.reshape(batch_size, seq_len, d_model)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class GPTBlock(nn.Module):
    """Transformer block for GPT model."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GPTAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        
        # Feed-forward with residual
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        
        return x


class GPTPostOptimizer(nn.Module):
    """
    GPT-style causal language model for LinkedIn post optimization.
    Trained from scratch as an alternative to DistilGPT-2/T5-small.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Tie weights between token embedding and output projection
        self.head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def get_causal_mask(self, seq_len: int, device):
        """Create causal mask for self-attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        return mask
    
    def forward(self, input_ids, return_features=False):
        """
        Forward pass through GPT model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            return_features: Whether to return features instead of logits
            
        Returns:
            Output logits (batch_size, seq_len, vocab_size) or features
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Get embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self.get_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Return features if requested
        if return_features:
            return x
        
        # Output projection
        logits = self.head(x)
        
        return logits
    
    def generate(
        self,
        prompt_ids,
        max_length=200,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        eos_token_id=None
    ):
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = prompt_ids.device
        generated = prompt_ids.clone()
        
        if eos_token_id is None:
            eos_token_id = self.vocab_size - 1
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for current sequence
                logits = self.forward(generated)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token generated
                if next_token.item() == eos_token_id:
                    break
                
                # Truncate if exceeds max length
                if generated.size(1) > self.max_seq_length:
                    generated = generated[:, -self.max_seq_length:]
        
        return generated
    
    def extract_features(self, input_ids):
        """Extract features for ensemble."""
        return self.forward(input_ids, return_features=True)


def create_gpt_model(vocab_size: int, **kwargs):
    """
    Factory function to create GPT model.
    
    Args:
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        GPTPostOptimizer model
    """
    return GPTPostOptimizer(vocab_size, **kwargs)
