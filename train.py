"""
Training Pipeline for LinkedIn Post Optimizer Models
Trains Transformer, CNN, and GPT models on scraped and synthetic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

from models.transformer_model import create_transformer_model
from models.cnn_model import create_cnn_model
from models.gpt_model import create_gpt_model
from models.ensemble import create_ensemble
from utils.preprocessing import Tokenizer, create_dataloaders
from utils.metrics import calculate_perplexity


class Trainer:
    """Trainer for deep learning models."""
    
    def __init__(
        self,
        model,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            model_name: Name of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training {self.model_name}")
        
        for batch in progress_bar:
            sources, targets = batch
            sources = sources.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (adapt based on model type)
            if 'gpt' in self.model_name.lower() or 'cnn' in self.model_name.lower():
                # Batch-first models
                logits = self.model(sources)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            else:
                # Seq-first transformer
                sources_t = sources.transpose(0, 1)
                targets_t = targets.transpose(0, 1)
                tgt_mask = self.model.generate_square_subsequent_mask(targets_t.size(0) - 1).to(self.device)
                logits = self.model(sources_t, targets_t[:-1], tgt_mask=tgt_mask)
                logits = logits.reshape(-1, logits.size(-1))
                targets_flat = targets_t[1:].reshape(-1)
                loss = self.criterion(logits, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sources, targets = batch
                sources = sources.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if 'gpt' in self.model_name.lower() or 'cnn' in self.model_name.lower():
                    logits = self.model(sources)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = targets[:, 1:].contiguous()
                    loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
                else:
                    sources_t = sources.transpose(0, 1)
                    targets_t = targets.transpose(0, 1)
                    tgt_mask = self.model.generate_square_subsequent_mask(targets_t.size(0) - 1).to(self.device)
                    logits = self.model(sources_t, targets_t[:-1], tgt_mask=tgt_mask)
                    logits = logits.reshape(-1, logits.size(-1))
                    targets_flat = targets_t[1:].reshape(-1)
                    loss = self.criterion(logits, targets_flat)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(self, num_epochs: int = 5, save_dir: str = 'checkpoints'):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"\nTraining {self.model_name} on {self.device}")
        print(f"Epochs: {num_epochs}, LR: {self.learning_rate}")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            train_ppl = calculate_perplexity(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_ppl = calculate_perplexity(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(save_dir, f'{self.model_name}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                }, checkpoint_path)
                print(f"✓ Saved best model to {checkpoint_path}")
        
        print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
        return self.history


def prepare_data():
    """Prepare training data from scraped and synthetic posts."""
    print("Preparing training data...")
    
    # Load or generate scraped data
    if os.path.exists('data/scraped_posts.csv'):
        scraped_df = pd.read_csv('data/scraped_posts.csv')
    else:
        from scraper import LinkedInScraper
        scraper = LinkedInScraper()
        scraped_df = scraper.scrape_demo_posts(num_posts=700)
        scraper.save_data(scraped_df)
    
    # Load or generate synthetic data
    if os.path.exists('data/synthetic_posts.csv'):
        synthetic_df = pd.read_csv('data/synthetic_posts.csv')
    else:
        from synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        synthetic_df = generator.generate_dataset(num_posts=1000)
        generator.save_data(synthetic_df)
    
    # Combine datasets
    combined_df = pd.concat([
        scraped_df[['content']],
        synthetic_df[['content']]
    ], ignore_index=True)
    
    print(f"✓ Combined dataset: {len(combined_df)} posts")
    
    return combined_df


def main():
    """Main training function."""
    print("LinkedIn Post Optimizer - Training Pipeline")
    print("="*60)
    
    # Prepare data
    df = prepare_data()
    
    # Build tokenizer
    tokenizer = Tokenizer(vocab_size=10000, max_length=256)
    tokenizer.build_vocab(df['content'].tolist())
    
    # Save tokenizer
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'word2idx': tokenizer.word2idx,
        'idx2word': tokenizer.idx2word,
        'vocab_size': len(tokenizer.word2idx),
        'max_length': tokenizer.max_length
    }, 'checkpoints/tokenizer.pt')
    print("✓ Saved tokenizer")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df, tokenizer, batch_size=8, train_split=0.8, val_split=0.1
    )
    
    vocab_size = len(tokenizer.word2idx)
    
    # Train Transformer model
    print("\n" + "="*60)
    print("1/3: Training Transformer Model")
    print("="*60)
    transformer = create_transformer_model(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=256
    )
    transformer_trainer = Trainer(
        transformer, 'transformer', train_loader, val_loader, learning_rate=0.0001
    )
    transformer_history = transformer_trainer.train(num_epochs=3)
    
    # Train CNN model
    print("\n" + "="*60)
    print("2/3: Training CNN Model")
    print("="*60)
    cnn = create_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        num_layers=3,
        dropout=0.3,
        max_seq_length=256
    )
    cnn_trainer = Trainer(
        cnn, 'cnn', train_loader, val_loader, learning_rate=0.0001
    )
    cnn_history = cnn_trainer.train(num_epochs=3)
    
    # Train GPT model
    print("\n" + "="*60)
    print("3/3: Training GPT Model")
    print("="*60)
    gpt = create_gpt_model(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        max_seq_length=256,
        dropout=0.1
    )
    gpt_trainer = Trainer(
        gpt, 'gpt', train_loader, val_loader, learning_rate=0.0001
    )
    gpt_history = gpt_trainer.train(num_epochs=3)
    
    # Create ensemble and save weights
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    
    perplexities = {
        'transformer': transformer_history['val_ppl'][-1],
        'cnn': cnn_history['val_ppl'][-1],
        'gpt': gpt_history['val_ppl'][-1]
    }
    
    ensemble = create_ensemble(transformer, cnn, gpt)
    ensemble.update_weights_from_perplexity(perplexities)
    
    # Save ensemble weights
    torch.save({
        'weights': ensemble.weights.tolist(),
        'perplexities': perplexities
    }, 'checkpoints/ensemble_weights.pt')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nAll models saved to checkpoints/")
    print("Ready for inference with Streamlit app!")


if __name__ == "__main__":
    main()
