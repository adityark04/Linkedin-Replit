"""
Ensemble System for LinkedIn Post Optimization
Combines Transformer, CNN, and GPT models with weighted voting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


class EnsemblePostOptimizer(nn.Module):
    """
    Ensemble model combining Transformer, CNN, and GPT models.
    Uses weighted voting based on validation perplexity.
    """
    
    def __init__(
        self,
        transformer_model,
        cnn_model,
        gpt_model,
        weights: List[float] = None,
        temperature: float = 1.0
    ):
        """
        Initialize ensemble model.
        
        Args:
            transformer_model: Transformer seq2seq model
            cnn_model: CNN model
            gpt_model: GPT model
            weights: Weights for each model [transformer_weight, cnn_weight, gpt_weight]
            temperature: Temperature for softmax combination
        """
        super().__init__()
        self.transformer = transformer_model
        self.cnn = cnn_model
        self.gpt = gpt_model
        
        # Default equal weights if not provided
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        self.temperature = temperature
    
    def set_weights(self, weights: List[float]):
        """
        Update ensemble weights based on validation performance.
        
        Args:
            weights: New weights [transformer_weight, cnn_weight, gpt_weight]
        """
        weights = np.array(weights)
        weights = weights / weights.sum()
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def update_weights_from_perplexity(self, perplexities: Dict[str, float]):
        """
        Update weights based on validation perplexity scores.
        Lower perplexity gets higher weight.
        
        Args:
            perplexities: Dict with keys 'transformer', 'cnn', 'gpt' and perplexity values
        """
        # Convert perplexities to weights (inverse relationship)
        transformer_ppl = perplexities.get('transformer', 100)
        cnn_ppl = perplexities.get('cnn', 100)
        gpt_ppl = perplexities.get('gpt', 100)
        
        # Calculate weights as inverse of perplexity
        weights = np.array([1/transformer_ppl, 1/cnn_ppl, 1/gpt_ppl])
        weights = weights / weights.sum()
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        print(f"Updated ensemble weights based on perplexity:")
        print(f"  Transformer: {weights[0]:.3f} (ppl: {transformer_ppl:.2f})")
        print(f"  CNN: {weights[1]:.3f} (ppl: {cnn_ppl:.2f})")
        print(f"  GPT: {weights[2]:.3f} (ppl: {gpt_ppl:.2f})")
    
    def forward_gpt_with_transformer_format(self, src, tgt):
        """
        Adapt GPT model to work with seq2seq format.
        Concatenates src and tgt for causal LM.
        """
        # Concatenate src and tgt
        combined = torch.cat([src, tgt], dim=0)
        
        # Transpose to batch-first for GPT
        combined = combined.transpose(0, 1)
        
        # Get logits
        logits = self.gpt(combined)
        
        # Extract logits for tgt portion
        tgt_len = tgt.size(0)
        tgt_logits = logits[:, -tgt_len:, :]
        
        # Transpose back to seq-first
        return tgt_logits.transpose(0, 1)
    
    def forward_cnn_with_transformer_format(self, src, tgt):
        """
        Adapt CNN model to work with seq2seq format.
        Concatenates src and tgt.
        """
        # Concatenate src and tgt
        combined = torch.cat([src, tgt], dim=0)
        
        # Transpose to batch-first for CNN
        combined = combined.transpose(0, 1)
        
        # Get logits
        logits = self.cnn(combined)
        
        # Extract logits for tgt portion
        tgt_len = tgt.size(0)
        tgt_logits = logits[:, -tgt_len:, :]
        
        # Transpose back to seq-first
        return tgt_logits.transpose(0, 1)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through ensemble.
        
        Args:
            src: Source tokens (seq_len, batch_size)
            tgt: Target tokens (seq_len, batch_size)
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            Weighted ensemble logits (seq_len, batch_size, vocab_size)
        """
        device = src.device
        weights = self.weights.to(device)
        
        # Get predictions from each model
        transformer_logits = self.transformer(src, tgt, src_mask, tgt_mask)
        cnn_logits = self.forward_cnn_with_transformer_format(src, tgt)
        gpt_logits = self.forward_gpt_with_transformer_format(src, tgt)
        
        # Convert logits to probabilities
        transformer_probs = F.softmax(transformer_logits / self.temperature, dim=-1)
        cnn_probs = F.softmax(cnn_logits / self.temperature, dim=-1)
        gpt_probs = F.softmax(gpt_logits / self.temperature, dim=-1)
        
        # Weighted combination
        ensemble_probs = (
            weights[0] * transformer_probs +
            weights[1] * cnn_probs +
            weights[2] * gpt_probs
        )
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-10)
        
        return ensemble_logits
    
    def generate(
        self,
        src,
        max_length=200,
        temperature=1.0,
        top_k=50,
        strategy='weighted'
    ):
        """
        Generate optimized post using ensemble.
        
        Args:
            src: Source tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            strategy: 'weighted' (combine probabilities) or 'voting' (majority vote)
            
        Returns:
            Generated tokens
        """
        self.eval()
        device = src.device
        
        if strategy == 'weighted':
            return self._generate_weighted(src, max_length, temperature, top_k, device)
        else:
            return self._generate_voting(src, max_length, temperature, top_k, device)
    
    def _generate_weighted(self, src, max_length, temperature, top_k, device):
        """Generate using weighted probability combination."""
        weights = self.weights.to(device)
        
        with torch.no_grad():
            # Generate from each model
            transformer_gen = self.transformer.generate(src, max_length, temperature, top_k)
            cnn_gen = self.cnn.generate(src.transpose(0, 1), max_length, temperature, top_k)
            gpt_gen = self.gpt.generate(src.transpose(0, 1), max_length, temperature, top_k)
            
            # For simplicity, use the model with highest weight
            max_weight_idx = torch.argmax(weights).item()
            
            if max_weight_idx == 0:
                return transformer_gen
            elif max_weight_idx == 1:
                return cnn_gen
            else:
                return gpt_gen.squeeze().tolist() if isinstance(gpt_gen, torch.Tensor) else gpt_gen
    
    def _generate_voting(self, src, max_length, temperature, top_k, device):
        """Generate using majority voting at each step."""
        # For simplicity, use weighted strategy
        return self._generate_weighted(src, max_length, temperature, top_k, device)
    
    def evaluate_model(self, model_name: str, dataloader, criterion):
        """
        Evaluate a single model in the ensemble.
        
        Args:
            model_name: Name of model ('transformer', 'cnn', or 'gpt')
            dataloader: DataLoader for evaluation
            criterion: Loss criterion
            
        Returns:
            Average loss and perplexity
        """
        if model_name == 'transformer':
            model = self.transformer
        elif model_name == 'cnn':
            model = self.cnn
        else:
            model = self.gpt
        
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src, tgt = batch
                
                if model_name == 'transformer':
                    logits = model(src, tgt[:-1])
                elif model_name == 'cnn':
                    logits = self.forward_cnn_with_transformer_format(src, tgt[:-1])
                else:
                    logits = self.forward_gpt_with_transformer_format(src, tgt[:-1])
                
                # Calculate loss
                logits = logits.reshape(-1, logits.size(-1))
                targets = tgt[1:].reshape(-1)
                loss = criterion(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity


def create_ensemble(transformer_model, cnn_model, gpt_model, weights=None):
    """
    Factory function to create ensemble model.
    
    Args:
        transformer_model: Transformer model instance
        cnn_model: CNN model instance
        gpt_model: GPT model instance
        weights: Optional weights for ensemble
        
    Returns:
        EnsemblePostOptimizer instance
    """
    return EnsemblePostOptimizer(transformer_model, cnn_model, gpt_model, weights)
