"""
Synthetic Data Generation Module
Generates 1000 diverse draft LinkedIn posts with varying tones, lengths, and contexts.
"""

import pandas as pd
import random
import numpy as np
from typing import List, Dict


class SyntheticDataGenerator:
    """
    Generate synthetic LinkedIn post drafts for training data augmentation.
    Creates diverse posts mimicking different tones, lengths, and demographics.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define various dimensions for diversity
        self.tones = [
            'professional', 'enthusiastic', 'thoughtful', 'casual', 
            'inspirational', 'educational', 'conversational', 'urgent'
        ]
        
        self.post_types = [
            'announcement', 'insight', 'question', 'story', 
            'list', 'quote', 'achievement', 'hiring', 'event'
        ]
        
        self.industries = [
            'software engineering', 'marketing', 'sales', 'product management',
            'data science', 'design', 'hr', 'finance', 'consulting', 'startups'
        ]
        
        self.demographics = [
            'early career', 'mid career', 'senior leader', 'founder',
            'freelancer', 'student', 'career switcher'
        ]
        
    def generate_announcement(self, tone: str, length: str) -> str:
        """Generate announcement-type post."""
        templates = {
            'short': [
                "Big news: {announcement}!",
                "Excited to share: {announcement}",
                "Quick update: {announcement}"
            ],
            'medium': [
                "Thrilled to announce {announcement}! This is a huge milestone for our team and I couldn't be more proud.",
                "After months of hard work, {announcement}. Grateful for everyone who made this possible.",
                "Today marks a special day: {announcement}. Here's to new beginnings!"
            ],
            'long': [
                "I'm incredibly excited to share some news that's been months in the making: {announcement}.\n\nThis journey started when {backstory}. Along the way, we learned {lesson}.\n\nHuge thanks to {acknowledgment} for making this happen. The best is yet to come!",
            ]
        }
        
        announcements = [
            "we've raised $5M in Series A funding",
            "our product just hit 10,000 users",
            "I'm joining an amazing team at TechCorp",
            "we're launching our new platform today",
            "our team has grown to 50 people",
            "we've been acquired by BigTech"
        ]
        
        template = random.choice(templates[length])
        return template.format(
            announcement=random.choice(announcements),
            backstory="we saw a gap in the market",
            lesson="the importance of listening to customers",
            acknowledgment="our incredible team"
        )
    
    def generate_insight(self, tone: str, length: str) -> str:
        """Generate insight/advice-type post."""
        templates = {
            'short': [
                "{hot_take}",
                "Key learning: {insight}",
                "Reminder: {advice}"
            ],
            'medium': [
                "After {years} years in {industry}, here's what I've learned: {insight}\n\n{elaboration}",
                "{hot_take}\n\nHere's why this matters: {reasoning}",
                "The best advice I ever received: {quote}\n\n{application}"
            ],
            'long': [
                "{number} lessons from {experience}:\n\n1. {lesson1}\n2. {lesson2}\n3. {lesson3}\n4. {lesson4}\n5. {lesson5}\n\nWhat would you add to this list?",
            ]
        }
        
        template = random.choice(templates[length])
        return template.format(
            years=random.randint(3, 15),
            industry=random.choice(self.industries),
            insight="customer empathy trumps everything else",
            elaboration="The products that win are built with deep user understanding.",
            hot_take="Your competitors aren't your biggest threat—your own assumptions are.",
            reasoning="We often build what we think users want, not what they actually need.",
            quote="'Don't find customers for your product, find products for your customers'",
            application="This shifted my entire approach to product development.",
            number=random.randint(3, 7),
            experience="building in public",
            lesson1="Ship fast and iterate",
            lesson2="Listen more than you talk",
            lesson3="Focus on solving real problems",
            lesson4="Build community, not just product",
            lesson5="Celebrate small wins"
        )
    
    def generate_question(self, tone: str, length: str) -> str:
        """Generate question/discussion-type post."""
        questions = [
            "What's the best career advice you've ever received?",
            "How do you stay productive while working remotely?",
            "What's one skill every {role} should master?",
            "How has {technology} changed your workflow?",
            "What's your go-to productivity hack?",
            "How do you handle {challenge}?",
        ]
        
        question = random.choice(questions).format(
            role=random.choice(['marketer', 'developer', 'designer', 'leader']),
            technology=random.choice(['AI', 'automation', 'cloud computing']),
            challenge=random.choice(['burnout', 'imposter syndrome', 'work-life balance'])
        )
        
        if length == 'short':
            return question
        elif length == 'medium':
            return f"{question}\n\nI'll start: {random.choice(['Consistency beats intensity', 'Focus on one thing at a time', 'Always be learning'])}"
        else:
            return f"Let's discuss: {question}\n\nI've been thinking about this a lot lately because {random.choice(['it directly impacts team performance', 'I see this challenge everywhere', 'it's more important than ever'])}\n\nIn my experience, {random.choice(['the best approach is to', 'what works is', 'the key is'])} {random.choice(['start small and build habits', 'focus on fundamentals', 'embrace continuous learning'])}\n\nWhat's your take?"
    
    def generate_story(self, tone: str, length: str) -> str:
        """Generate storytelling-type post."""
        stories = [
            "failed my first startup",
            "got rejected from 50 companies",
            "switched careers at 35",
            "built my first product",
            "led my first team",
            "overcame imposter syndrome"
        ]
        
        story = random.choice(stories)
        
        if length == 'short':
            return f"When I {story}, I learned {random.choice(['persistence pays off', 'failure is feedback', 'timing matters'])}."
        elif length == 'medium':
            return f"A few years ago, I {story}.\n\nIt taught me that {random.choice(['success is rarely linear', 'the journey matters more than the destination', 'growth comes from discomfort'])}\n\nThat lesson still guides me today."
        else:
            return f"Story time: How I {story}\n\nFive years ago, I was {random.choice(['stuck in a job I hated', 'afraid to take risks', 'playing it safe'])}\n\nThen {random.choice(['everything changed', 'I made a decision', 'I took a leap'])}\n\nI {story}. It was {random.choice(['terrifying', 'exhilarating', 'the best decision I ever made'])}\n\nLooking back, the biggest lesson was: {random.choice(['bet on yourself', 'embrace uncertainty', 'take calculated risks'])}\n\nWhat's a risk you took that paid off?"
    
    def generate_list_post(self, tone: str, length: str) -> str:
        """Generate list-type post."""
        topics = [
            "tools every {role} should know",
            "mistakes to avoid in {industry}",
            "habits of successful {role}s",
            "trends shaping {industry}",
            "books that changed my {perspective}"
        ]
        
        topic = random.choice(topics).format(
            role=random.choice(['marketer', 'founder', 'leader', 'developer']),
            industry=random.choice(self.industries),
            perspective=random.choice(['career', 'life', 'leadership'])
        )
        
        num_items = 3 if length == 'short' else 5 if length == 'medium' else 7
        
        items = [
            "Focus on value, not features",
            "Listen to your customers",
            "Ship fast and iterate",
            "Build in public",
            "Embrace failure as learning",
            "Network authentically",
            "Invest in personal growth",
            "Stay curious",
            "Take calculated risks",
            "Celebrate small wins"
        ]
        
        selected_items = random.sample(items, num_items)
        list_content = '\n'.join([f"{i+1}. {item}" for i, item in enumerate(selected_items)])
        
        return f"{num_items} {topic}:\n\n{list_content}\n\nWhat would you add?"
    
    def determine_length_category(self, word_count: int) -> str:
        """Categorize post by length."""
        if word_count < 30:
            return 'short'
        elif word_count < 100:
            return 'medium'
        else:
            return 'long'
    
    def generate_draft_post(self) -> Dict:
        """Generate a single diverse draft post."""
        # Random parameters
        tone = random.choice(self.tones)
        post_type = random.choice(self.post_types)
        industry = random.choice(self.industries)
        demographic = random.choice(self.demographics)
        target_length = random.choice(['short', 'medium', 'long'])
        
        # Generate content based on type
        generators = {
            'announcement': self.generate_announcement,
            'insight': self.generate_insight,
            'question': self.generate_question,
            'story': self.generate_story,
            'list': self.generate_list_post,
        }
        
        # Default to insight for types not explicitly defined
        generator = generators.get(post_type, self.generate_insight)
        content = generator(tone, target_length)
        
        # Add hashtags
        hashtag_options = [
            f"#{industry.replace(' ', '')}",
            "#Leadership",
            "#CareerGrowth",
            "#Entrepreneurship",
            "#Marketing",
            "#Tech",
            "#StartupLife",
            "#Innovation"
        ]
        num_hashtags = random.randint(2, 4)
        hashtags = ' '.join(random.sample(hashtag_options, num_hashtags))
        
        full_content = f"{content}\n\n{hashtags}"
        
        return {
            'content': full_content,
            'draft_quality': random.choice(['rough', 'good', 'polished']),
            'tone': tone,
            'post_type': post_type,
            'industry': industry,
            'demographic': demographic,
            'word_count': len(full_content.split()),
            'length_category': self.determine_length_category(len(full_content.split())),
            'has_emoji': False,
            'has_hashtags': True,
            'engagement_prediction': random.randint(50, 2000)
        }
    
    def generate_dataset(self, num_posts: int = 1000) -> pd.DataFrame:
        """
        Generate complete dataset of synthetic posts.
        
        Args:
            num_posts: Number of posts to generate
            
        Returns:
            DataFrame with synthetic posts
        """
        print(f"Generating {num_posts} synthetic LinkedIn draft posts...")
        
        posts = []
        for i in range(num_posts):
            if i % 100 == 0:
                print(f"  Generated {i}/{num_posts} posts...")
            posts.append(self.generate_draft_post())
        
        df = pd.DataFrame(posts)
        print(f"✓ Generated {len(df)} synthetic posts")
        print(f"  Tones: {df['tone'].nunique()}")
        print(f"  Types: {df['post_type'].nunique()}")
        print(f"  Length distribution: {dict(df['length_category'].value_counts())}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str = 'data/synthetic_posts.csv'):
        """Save synthetic data to CSV."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved synthetic data to {filepath}")
        return filepath


def main():
    """Main function to generate synthetic data."""
    generator = SyntheticDataGenerator()
    df = generator.generate_dataset(num_posts=1000)
    filepath = generator.save_data(df)
    
    print(f"\nSample synthetic post:\n{df.iloc[0]['content']}\n")
    print(f"Metadata: tone={df.iloc[0]['tone']}, type={df.iloc[0]['post_type']}, words={df.iloc[0]['word_count']}")
    
    return df


if __name__ == "__main__":
    main()
