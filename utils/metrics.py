"""
Evaluation Metrics
Implements BLEU, ROUGE, readability, and sentiment analysis.
"""

import numpy as np
from typing import List, Dict
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rouge_score import rouge_scorer
import re


class MetricsEvaluator:
    """Evaluator for LinkedIn post quality metrics."""
    
    def __init__(self):
        """Initialize metrics evaluator."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_bleu(self, reference: str, candidate: str, n: int = 4) -> float:
        """
        Calculate BLEU score (simplified implementation).
        
        Args:
            reference: Reference text
            candidate: Candidate text
            n: Maximum n-gram size
            
        Returns:
            BLEU score
        """
        # Tokenize
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # Calculate precision for each n-gram size
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, i)
            cand_ngrams = self._get_ngrams(cand_tokens, i)
            
            if len(cand_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = sum(min(ref_ngrams.get(ng, 0), cand_ngrams[ng]) for ng in cand_ngrams)
            precision = matches / len(cand_ngrams)
            precisions.append(precision)
        
        # Brevity penalty
        bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(cand_tokens))
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = bp * np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        
        return bleu
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """Get n-grams from tokens."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dict with various readability scores
        """
        if len(text.strip()) == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0
            }
        
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text)
            }
        except:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0
            }
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores
    
    def calculate_engagement_score(self, text: str) -> float:
        """
        Predict engagement score based on text features.
        
        Args:
            text: Post text
            
        Returns:
            Predicted engagement score (0-100)
        """
        # Simple heuristic-based engagement prediction
        score = 50  # Base score
        
        # Word count (optimal around 50-100 words)
        word_count = len(text.split())
        if 50 <= word_count <= 150:
            score += 15
        elif word_count < 50:
            score += 5
        
        # Has hashtags
        hashtags = re.findall(r'#\w+', text)
        if len(hashtags) > 0:
            score += 10
        if len(hashtags) > 3:
            score -= 5  # Too many hashtags
        
        # Has numbers (data/metrics)
        if re.search(r'\d+', text):
            score += 8
        
        # Has questions
        if '?' in text:
            score += 7
        
        # Call to action words
        cta_words = ['share', 'comment', 'dm', 'link', 'check out', 'join', 'follow']
        if any(word in text.lower() for word in cta_words):
            score += 10
        
        # Sentiment (positive posts tend to perform better)
        sentiment = self.calculate_sentiment(text)
        if sentiment['compound'] > 0.5:
            score += 10
        elif sentiment['compound'] < -0.5:
            score -= 5
        
        # Readability (easier to read = more engagement)
        readability = self.calculate_readability(text)
        flesch = readability.get('flesch_reading_ease', 0)
        if 60 <= flesch <= 80:  # Easy to read
            score += 8
        
        return max(0, min(100, score))
    
    def evaluate_post(self, reference: str, candidate: str) -> Dict[str, any]:
        """
        Comprehensive evaluation of generated post.
        
        Args:
            reference: Reference/original post
            candidate: Generated/optimized post
            
        Returns:
            Dict with all evaluation metrics
        """
        results = {
            'bleu': self.calculate_bleu(reference, candidate),
            'rouge': self.calculate_rouge(reference, candidate),
            'readability': self.calculate_readability(candidate),
            'sentiment': self.calculate_sentiment(candidate),
            'engagement_score': self.calculate_engagement_score(candidate),
            'word_count': len(candidate.split()),
            'char_count': len(candidate),
            'hashtag_count': len(re.findall(r'#\w+', candidate))
        }
        
        return results
    
    def print_evaluation(self, results: Dict[str, any]):
        """Pretty print evaluation results."""
        print("\n" + "="*50)
        print("POST EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nSimilarity Metrics:")
        print(f"  BLEU Score: {results['bleu']:.3f}")
        print(f"  ROUGE-1: {results['rouge']['rouge1']:.3f}")
        print(f"  ROUGE-2: {results['rouge']['rouge2']:.3f}")
        print(f"  ROUGE-L: {results['rouge']['rougeL']:.3f}")
        
        print(f"\nReadability:")
        print(f"  Flesch Reading Ease: {results['readability']['flesch_reading_ease']:.1f}")
        print(f"  Grade Level: {results['readability']['flesch_kincaid_grade']:.1f}")
        
        print(f"\nSentiment Analysis:")
        print(f"  Positive: {results['sentiment']['pos']:.3f}")
        print(f"  Neutral: {results['sentiment']['neu']:.3f}")
        print(f"  Negative: {results['sentiment']['neg']:.3f}")
        print(f"  Compound: {results['sentiment']['compound']:.3f}")
        
        print(f"\nEngagement Prediction:")
        print(f"  Predicted Score: {results['engagement_score']:.1f}/100")
        
        print(f"\nPost Statistics:")
        print(f"  Word Count: {results['word_count']}")
        print(f"  Character Count: {results['char_count']}")
        print(f"  Hashtags: {results['hashtag_count']}")
        print("="*50 + "\n")


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return np.exp(loss)
