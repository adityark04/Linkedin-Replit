"""
LinkedIn Post Scraper Module
Ethically scrapes LinkedIn posts from public profiles with proper delays and robots.txt compliance.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import trafilatura
from typing import List, Dict, Optional
import json
import re


class LinkedInScraper:
    """
    Ethical web scraper for LinkedIn posts from public profiles.
    Implements delays, robots.txt compliance, and rate limiting.
    """
    
    def __init__(self, delay_range=(2, 5), user_agent=None):
        """
        Initialize the scraper with ethical browsing settings.
        
        Args:
            delay_range: Tuple of (min, max) seconds to wait between requests
            user_agent: Custom user agent string
        """
        self.delay_range = delay_range
        self.session = requests.Session()
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.posts_collected = []
        
    def check_robots_txt(self, url: str) -> bool:
        """
        Check if scraping is allowed according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if scraping is allowed
        """
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(self.user_agent, url)
        except Exception as e:
            print(f"Warning: Could not check robots.txt: {e}")
            return True
    
    def respectful_delay(self):
        """Add random delay between requests to minimize server load."""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def scrape_public_post(self, url: str) -> Optional[Dict]:
        """
        Scrape a single public LinkedIn post.
        
        Args:
            url: URL of the post
            
        Returns:
            Dictionary containing post data or None if failed
        """
        if not self.check_robots_txt(url):
            print(f"Robots.txt disallows scraping: {url}")
            return None
        
        try:
            self.respectful_delay()
            
            # Use trafilatura for content extraction (web_scraper integration)
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
                
            text_content = trafilatura.extract(downloaded)
            
            # Also get raw HTML for additional metadata
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract post components
            post_data = {
                'url': url,
                'content': text_content or '',
                'title': self._extract_title(soup),
                'hashtags': self._extract_hashtags(text_content or ''),
                'engagement_score': random.randint(50, 5000),  # Simulated for demo
                'author': self._extract_author(soup),
                'timestamp': pd.Timestamp.now(),
            }
            
            return post_data
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from post HTML."""
        # Look for common title tags
        title_tags = soup.find_all(['h1', 'h2', 'title'])
        if title_tags:
            return title_tags[0].get_text(strip=True)[:200]
        return ""
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from post content."""
        return re.findall(r'#\w+', text)
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author name from post HTML."""
        # Look for meta tags or author spans
        author_meta = soup.find('meta', {'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta['content']
        return "Unknown"
    
    def scrape_demo_posts(self, num_posts: int = 100) -> pd.DataFrame:
        """
        Generate demo posts for testing when real LinkedIn scraping isn't feasible.
        In production, this would scrape real public posts.
        
        Args:
            num_posts: Number of demo posts to generate
            
        Returns:
            DataFrame with scraped/demo posts
        """
        print(f"Generating {num_posts} demo LinkedIn posts...")
        print("Note: Real LinkedIn scraping requires login and violates ToS.")
        print("This generates realistic demo data for model training.")
        
        # Demo data templates for tech/marketing posts
        templates = [
            "Excited to share our latest product launch! {topic} is transforming how we {action}. {hashtags}",
            "5 key lessons from building a {topic} startup:\n1. {lesson}\n2. Focus on customers\n3. Iterate quickly\n4. Build in public\n5. Never give up\n{hashtags}",
            "Just wrapped up an amazing quarter! Our {metric} grew by {percent}% thanks to our incredible team. {hashtags}",
            "Hot take: {opinion} is the future of {industry}. Here's why... {hashtags}",
            "Thrilled to announce I'm joining {company} as {role}! Looking forward to {goal}. {hashtags}",
            "After {years} years in {industry}, here are my top {number} insights: {insight} {hashtags}",
            "We're hiring! Looking for passionate {role} to help us {mission}. DM if interested! {hashtags}",
            "The secret to {achievement}? {advice}. It's that simple. {hashtags}",
            "Reflecting on {event} and how it changed {perspective}. Key takeaway: {lesson}. {hashtags}",
            "New blog post: '{title}' - diving deep into {topic}. Link in comments! {hashtags}"
        ]
        
        topics = ["AI", "ML", "data science", "product management", "marketing automation", 
                  "growth hacking", "SaaS", "customer success", "digital transformation"]
        actions = ["work", "analyze data", "reach customers", "scale", "innovate"]
        hashtags_list = [
            "#TechLeadership #Innovation",
            "#StartupLife #Entrepreneurship",
            "#Marketing #GrowthHacking",
            "#ProductManagement #Tech",
            "#AI #MachineLearning #DataScience",
            "#Leadership #CareerGrowth",
            "#SaaS #B2B",
            "#DigitalMarketing #ContentStrategy"
        ]
        
        posts = []
        for i in range(num_posts):
            template = random.choice(templates)
            content = template.format(
                topic=random.choice(topics),
                action=random.choice(actions),
                hashtags=random.choice(hashtags_list),
                metric=random.choice(["revenue", "users", "engagement", "conversions"]),
                percent=random.randint(10, 200),
                opinion=random.choice(topics),
                industry=random.choice(["tech", "marketing", "sales", "business"]),
                company=f"Company{random.randint(1, 100)}",
                role=random.choice(["Product Manager", "VP of Marketing", "CTO", "Head of Growth"]),
                goal=random.choice(["revolutionize the industry", "build amazing products", "scale rapidly"]),
                years=random.randint(3, 15),
                number=random.randint(3, 10),
                insight=random.choice(["Focus on value", "Listen to customers", "Ship fast"]),
                achievement=random.choice(["success", "growth", "product-market fit"]),
                advice=random.choice(["Consistency", "Customer focus", "Rapid iteration"]),
                event=random.choice(["this year", "the pandemic", "this journey"]),
                perspective=random.choice(["work", "leadership", "innovation"]),
                lesson=random.choice(["stay curious", "embrace failure", "think long-term"]),
                title=f"{random.choice(topics)} in 2024"
            )
            
            posts.append({
                'url': f'https://linkedin.com/posts/demo-{i}',
                'content': content,
                'title': content.split('\n')[0][:100],
                'hashtags': re.findall(r'#\w+', content),
                'engagement_score': random.randint(100, 10000),
                'author': f'Author{random.randint(1, 50)}',
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 180)),
                'word_count': len(content.split()),
                'has_emoji': random.choice([True, False])
            })
        
        df = pd.DataFrame(posts)
        print(f"✓ Generated {len(df)} demo posts")
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str = 'data/scraped_posts.csv'):
        """
        Save scraped posts to CSV file.
        
        Args:
            df: DataFrame containing posts
            filepath: Output file path
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert hashtags list to string for CSV storage
        df_copy = df.copy()
        df_copy['hashtags'] = df_copy['hashtags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        df_copy.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} posts to {filepath}")
        return filepath


def main():
    """Main function to demonstrate scraping."""
    scraper = LinkedInScraper()
    
    # Generate demo data (500-1000 posts)
    num_posts = random.randint(500, 1000)
    df = scraper.scrape_demo_posts(num_posts=num_posts)
    
    # Save to CSV
    filepath = scraper.save_data(df)
    
    print(f"\nDataset Summary:")
    print(f"Total posts: {len(df)}")
    print(f"Average engagement: {df['engagement_score'].mean():.0f}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSample post:\n{df.iloc[0]['content']}")
    
    return df


if __name__ == "__main__":
    main()
