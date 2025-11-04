"""
LinkedIn Post Optimizer - Streamlit Web Application
Interactive interface for optimizing LinkedIn posts using ensemble deep learning models.
"""

import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
from typing import Dict

from models.transformer_model import create_transformer_model
from models.cnn_model import create_cnn_model
from models.gpt_model import create_gpt_model
from models.ensemble import create_ensemble
from utils.preprocessing import Tokenizer
from utils.metrics import MetricsEvaluator


# Page configuration
st.set_page_config(
    page_title="LinkedIn Post Optimizer",
    page_icon="📱",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load trained models and tokenizer."""
    device = torch.device('cpu')
    
    # Check if models exist
    if not os.path.exists('checkpoints/tokenizer.pt'):
        return None, None, None
    
    # Load tokenizer
    tokenizer_data = torch.load('checkpoints/tokenizer.pt', map_location=device, weights_only=True)
    tokenizer = Tokenizer()
    tokenizer.word2idx = tokenizer_data['word2idx']
    tokenizer.idx2word = tokenizer_data['idx2word']
    tokenizer.vocab_built = True
    tokenizer.max_length = tokenizer_data.get('max_length', 256)
    
    vocab_size = len(tokenizer.word2idx)
    
    # Initialize models
    transformer = create_transformer_model(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        max_seq_length=256
    )
    
    cnn = create_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        num_layers=3,
        max_seq_length=256
    )
    
    gpt = create_gpt_model(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        max_seq_length=256
    )
    
    # Load model weights if available
    if os.path.exists('checkpoints/transformer_best.pt'):
        transformer.load_state_dict(torch.load('checkpoints/transformer_best.pt', map_location=device, weights_only=True)['model_state_dict'])
    if os.path.exists('checkpoints/cnn_best.pt'):
        cnn.load_state_dict(torch.load('checkpoints/cnn_best.pt', map_location=device, weights_only=True)['model_state_dict'])
    if os.path.exists('checkpoints/gpt_best.pt'):
        gpt.load_state_dict(torch.load('checkpoints/gpt_best.pt', map_location=device, weights_only=True)['model_state_dict'])
    
    # Create ensemble
    ensemble = create_ensemble(transformer, cnn, gpt)
    
    # Load ensemble weights if available
    if os.path.exists('checkpoints/ensemble_weights.pt'):
        ensemble_data = torch.load('checkpoints/ensemble_weights.pt', map_location=device, weights_only=True)
        ensemble.weights = torch.tensor(ensemble_data['weights'])
    
    ensemble.eval()
    
    return ensemble, tokenizer, device


@st.cache_resource
def load_evaluator():
    """Load metrics evaluator."""
    return MetricsEvaluator()


def optimize_post(draft: str, model_choice: str, tokenizer: Tokenizer, ensemble, device) -> str:
    """
    Optimize a LinkedIn post draft.
    
    Args:
        draft: Draft post text
        model_choice: Which model to use
        tokenizer: Tokenizer instance
        ensemble: Ensemble model
        device: Torch device
        
    Returns:
        Optimized post text
    """
    # Encode draft
    input_ids = tokenizer.encode(draft, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate based on model choice
    with torch.no_grad():
        if model_choice == "Ensemble (All Models)":
            # Use ensemble generation
            generated_ids = ensemble.generate(
                input_tensor.transpose(0, 1),
                max_length=200,
                temperature=0.8,
                top_k=50
            )
        elif model_choice == "Transformer":
            generated_ids = ensemble.transformer.generate(
                input_tensor.transpose(0, 1),
                max_length=200,
                temperature=0.8,
                top_k=50
            )
        elif model_choice == "CNN":
            generated_ids = ensemble.cnn.generate(
                input_tensor,
                max_length=200,
                temperature=0.8,
                top_k=50
            )
        else:  # GPT
            generated_output = ensemble.gpt.generate(
                input_tensor,
                max_length=200,
                temperature=0.8,
                top_k=50
            )
            if isinstance(generated_output, torch.Tensor):
                generated_ids = generated_output[0].tolist()
            else:
                generated_ids = generated_output
    
    # Decode
    if isinstance(generated_ids, list):
        optimized_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        optimized_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    
    # If generation is empty or too short, return enhanced version of original
    if len(optimized_text.strip()) < 10:
        optimized_text = enhance_draft_rule_based(draft)
    
    return optimized_text


def enhance_draft_rule_based(draft: str) -> str:
    """
    Fallback rule-based enhancement when model generation fails.
    
    Args:
        draft: Original draft
        
    Returns:
        Enhanced draft
    """
    import re
    
    # Clean up the text
    enhanced = draft.strip()
    
    # Add emojis if missing and appropriate
    if not any(char in enhanced for char in ['🚀', '💡', '✨', '📈', '🎯']):
        enhanced = '✨ ' + enhanced
    
    # Ensure proper hashtags
    if '#' not in enhanced:
        enhanced += '\n\n#Leadership #Innovation #Tech'
    
    # Add a call to action if missing
    cta_keywords = ['share', 'comment', 'thoughts', 'dm', 'link']
    if not any(keyword in enhanced.lower() for keyword in cta_keywords):
        enhanced += '\n\nWhat are your thoughts? Share in the comments!'
    
    return enhanced


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📱 LinkedIn Post Optimizer")
    st.markdown("""
    Transform your LinkedIn drafts into high-engagement posts using ensemble deep learning models!
    
    **Features:**
    - 🤖 Three AI models: Transformer, CNN, and GPT
    - 🎯 Ensemble predictions for optimal results
    - 📊 Comprehensive quality metrics
    - 💯 Engagement score prediction
    """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        ensemble, tokenizer, device = load_models()
        evaluator = load_evaluator()
    
    # Check if models are trained
    models_trained = ensemble is not None
    
    if not models_trained:
        st.warning("⚠️ Models not yet trained. Please run the training pipeline first:")
        st.code("python train.py", language="bash")
        st.info("For demo purposes, rule-based optimization is available below.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_choice = st.selectbox(
            "Select Model",
            ["Ensemble (All Models)", "Transformer", "CNN", "GPT"],
            help="Choose which model to use for optimization"
        )
        
        st.markdown("---")
        st.header("📖 About")
        st.markdown("""
        This tool uses three deep learning models:
        
        **🔄 Transformer**: Seq2seq encoder-decoder architecture
        
        **🎯 CNN**: 1D convolutional neural network for pattern recognition
        
        **🧠 GPT**: Causal language model for text generation
        
        The ensemble combines all three for optimal results!
        """)
        
        if models_trained and ensemble is not None:
            st.markdown("---")
            st.header("📊 Model Weights")
            weights = ensemble.weights.numpy()
            st.metric("Transformer", f"{weights[0]:.2%}")
            st.metric("CNN", f"{weights[1]:.2%}")
            st.metric("GPT", f"{weights[2]:.2%}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Your Draft")
        draft = st.text_area(
            "Enter your LinkedIn post draft:",
            height=300,
            placeholder="Excited to share that our team just launched a new product...",
            help="Paste or type your draft LinkedIn post here"
        )
        
        optimize_button = st.button("✨ Optimize Post", type="primary", use_container_width=True)
    
    with col2:
        st.header("🚀 Optimized Post")
        optimized_placeholder = st.empty()
    
    # Process optimization
    if optimize_button and draft.strip():
        with st.spinner("Optimizing your post..."):
            if models_trained and ensemble is not None:
                optimized = optimize_post(draft, model_choice, tokenizer, ensemble, device)
            else:
                optimized = enhance_draft_rule_based(draft)
            
            optimized_placeholder.text_area(
                "Optimized version:",
                value=optimized,
                height=300,
                help="Copy this optimized version to LinkedIn"
            )
        
        # Evaluation metrics
        st.markdown("---")
        st.header("📊 Post Quality Analysis")
        
        metrics = evaluator.evaluate_post(draft, optimized)
        
        # Display metrics in columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Engagement Score",
                f"{metrics['engagement_score']:.0f}/100",
                help="Predicted engagement based on content features"
            )
        
        with metric_col2:
            sentiment_label = "Positive" if metrics['sentiment']['compound'] > 0.1 else "Negative" if metrics['sentiment']['compound'] < -0.1 else "Neutral"
            st.metric(
                "Sentiment",
                sentiment_label,
                f"{metrics['sentiment']['compound']:.2f}",
                help="Overall sentiment of the post"
            )
        
        with metric_col3:
            readability = metrics['readability']['flesch_reading_ease']
            readability_label = "Easy" if readability > 60 else "Medium" if readability > 30 else "Hard"
            st.metric(
                "Readability",
                readability_label,
                f"{readability:.0f}",
                help="Flesch Reading Ease score"
            )
        
        with metric_col4:
            st.metric(
                "Word Count",
                metrics['word_count'],
                help="Total words in optimized post"
            )
        
        # Detailed metrics
        with st.expander("📈 Detailed Metrics"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.subheader("Similarity Metrics")
                st.write(f"**BLEU Score:** {metrics['bleu']:.3f}")
                st.write(f"**ROUGE-1:** {metrics['rouge']['rouge1']:.3f}")
                st.write(f"**ROUGE-2:** {metrics['rouge']['rouge2']:.3f}")
                st.write(f"**ROUGE-L:** {metrics['rouge']['rougeL']:.3f}")
                
                st.subheader("Content Stats")
                st.write(f"**Characters:** {metrics['char_count']}")
                st.write(f"**Hashtags:** {metrics['hashtag_count']}")
            
            with detail_col2:
                st.subheader("Readability Scores")
                st.write(f"**Flesch Reading Ease:** {metrics['readability']['flesch_reading_ease']:.1f}")
                st.write(f"**Grade Level:** {metrics['readability']['flesch_kincaid_grade']:.1f}")
                st.write(f"**Gunning Fog:** {metrics['readability']['gunning_fog']:.1f}")
                
                st.subheader("Sentiment Breakdown")
                st.write(f"**Positive:** {metrics['sentiment']['pos']:.2%}")
                st.write(f"**Neutral:** {metrics['sentiment']['neu']:.2%}")
                st.write(f"**Negative:** {metrics['sentiment']['neg']:.2%}")
        
        # Copy button
        st.markdown("---")
        st.success("✅ Your post has been optimized! Copy it and share on LinkedIn.")
    
    elif optimize_button and not draft.strip():
        st.error("❌ Please enter a draft post to optimize.")
    
    # Demo examples
    with st.expander("💡 Example Posts"):
        st.markdown("""
        **Example 1: Product Launch**
        ```
        We just launched our new product today. It helps companies automate their workflows.
        ```
        
        **Example 2: Career Update**
        ```
        Started a new position as Product Manager. Looking forward to working with the team.
        ```
        
        **Example 3: Insights**
        ```
        Working remotely has changed how we collaborate. Video calls are now essential.
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with ❤️ using Streamlit, PyTorch, and Deep Learning</p>
        <p>Models: Transformer Seq2Seq • 1D-CNN • GPT • Ensemble</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
