"""News detection and sentiment analysis for market signals."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class NewsSentiment(Enum):
    """Sentiment classification for news items."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class NewsSource(Enum):
    """Supported news sources."""
    POLYMARKET_COMMENTS = "polymarket_comments"
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS_API = "news_api"
    GOOGLE_NEWS = "google_news"
    RSS = "rss"
    MANUAL = "manual"


@dataclass
class NewsItem:
    """A single news item with sentiment analysis."""
    title: str
    content: str
    source: NewsSource
    url: Optional[str]
    published_at: datetime

    # Sentiment analysis
    sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    sentiment_score: Decimal = Decimal("0.0")  # -1.0 to 1.0
    confidence: Decimal = Decimal("0.5")

    # Relevance
    related_keywords: list[str] = field(default_factory=list)
    related_markets: list[str] = field(default_factory=list)

    # Impact assessment
    impact_score: Decimal = Decimal("0.5")  # 0.0 to 1.0
    is_breaking: bool = False

    @property
    def age_hours(self) -> float:
        """Hours since publication."""
        delta = datetime.utcnow() - self.published_at
        return delta.total_seconds() / 3600

    @property
    def is_fresh(self) -> bool:
        """Check if news is recent (< 6 hours)."""
        return self.age_hours < 6

    @property
    def weighted_sentiment(self) -> Decimal:
        """Sentiment weighted by confidence and freshness."""
        freshness_weight = Decimal(str(max(0.1, 1.0 - (self.age_hours / 24))))
        return self.sentiment_score * self.confidence * freshness_weight

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content[:500],  # Truncate for API
            "source": self.source.value,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "sentiment": self.sentiment.value,
            "sentiment_score": float(self.sentiment_score),
            "confidence": float(self.confidence),
            "impact_score": float(self.impact_score),
            "is_breaking": self.is_breaking,
            "age_hours": round(self.age_hours, 1),
        }


class NewsProvider:
    """
    Aggregates news from multiple sources for market analysis.

    Supports:
    - Polymarket community comments
    - Twitter/X via Grok API
    - News APIs
    - RSS feeds
    """

    def __init__(
        self,
        twitter_bearer_token: Optional[str] = None,
        news_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
    ):
        self._twitter_token = twitter_bearer_token
        self._news_api_key = news_api_key
        self._xai_api_key = xai_api_key
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cache: dict[str, list[NewsItem]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update: dict[str, datetime] = {}

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

    async def disconnect(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache or key not in self._last_cache_update:
            return False
        return datetime.utcnow() - self._last_cache_update[key] < self._cache_ttl

    async def get_news_for_market(
        self,
        market_question: str,
        keywords: Optional[list[str]] = None,
        hours_back: int = 24,
        max_items: int = 20,
    ) -> list[NewsItem]:
        """
        Get relevant news for a specific market.

        Args:
            market_question: The market question to find news for
            keywords: Additional search keywords
            hours_back: How far back to search
            max_items: Maximum items to return

        Returns:
            List of relevant news items sorted by relevance
        """
        cache_key = f"{market_question[:50]}_{hours_back}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][:max_items]

        # Extract keywords from market question
        search_keywords = self._extract_keywords(market_question)
        if keywords:
            search_keywords.extend(keywords)

        all_news: list[NewsItem] = []

        # Fetch from multiple sources in parallel
        tasks = []

        if self._news_api_key:
            tasks.append(self._fetch_from_news_api(search_keywords, hours_back))

        if self._twitter_token or self._xai_api_key:
            tasks.append(self._fetch_from_twitter(search_keywords, hours_back))

        # Always try RSS feeds (free)
        tasks.append(self._fetch_from_rss(search_keywords, hours_back))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"News fetch failed: {result}")

        # Deduplicate by title similarity
        unique_news = self._deduplicate(all_news)

        # Score relevance to market
        for item in unique_news:
            item.impact_score = self._score_relevance(item, market_question, search_keywords)

        # Sort by impact and freshness
        unique_news.sort(
            key=lambda x: (x.is_breaking, float(x.impact_score), -x.age_hours),
            reverse=True
        )

        # Update cache
        self._cache[cache_key] = unique_news
        self._last_cache_update[cache_key] = datetime.utcnow()

        return unique_news[:max_items]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract searchable keywords from text."""
        # Remove common words and punctuation
        stop_words = {
            "will", "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "by", "be", "is", "are", "was", "were", "been", "being", "have",
            "has", "had", "do", "does", "did", "this", "that", "these", "those",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "before", "after", "during", "between", "above", "below", "from",
        }

        # Extract words
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]+\b', text)

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            lower = word.lower()
            if lower not in stop_words and lower not in seen and len(lower) > 2:
                keywords.append(word)
                seen.add(lower)

        return keywords[:10]  # Limit to top 10

    async def _fetch_from_news_api(
        self,
        keywords: list[str],
        hours_back: int,
    ) -> list[NewsItem]:
        """Fetch news from News API."""
        if not self._news_api_key or not self._http_client:
            return []

        try:
            query = " OR ".join(keywords[:5])
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()

            response = await self._http_client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 50,
                    "apiKey": self._news_api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            items = []
            for article in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                    item = NewsItem(
                        title=article.get("title", ""),
                        content=article.get("description", "") or article.get("content", ""),
                        source=NewsSource.NEWS_API,
                        url=article.get("url"),
                        published_at=published,
                        related_keywords=keywords,
                    )
                    items.append(item)
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")

            return items

        except Exception as e:
            logger.error(f"News API fetch failed: {e}")
            return []

    async def _fetch_from_twitter(
        self,
        keywords: list[str],
        hours_back: int,
    ) -> list[NewsItem]:
        """
        Fetch tweets using Twitter API or Grok for analysis.

        If Grok API is available, use it to search and analyze tweets.
        Otherwise fall back to Twitter API v2.
        """
        if not self._http_client:
            return []

        # Prefer Grok for Twitter analysis (better context)
        if self._xai_api_key:
            return await self._analyze_twitter_with_grok(keywords, hours_back)

        if not self._twitter_token:
            return []

        try:
            query = " OR ".join(keywords[:5]) + " -is:retweet lang:en"

            response = await self._http_client.get(
                "https://api.twitter.com/2/tweets/search/recent",
                params={
                    "query": query,
                    "max_results": 50,
                    "tweet.fields": "created_at,public_metrics,context_annotations",
                },
                headers={"Authorization": f"Bearer {self._twitter_token}"},
            )
            response.raise_for_status()
            data = response.json()

            items = []
            for tweet in data.get("data", []):
                try:
                    published = datetime.fromisoformat(
                        tweet["created_at"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                    item = NewsItem(
                        title=tweet["text"][:100],
                        content=tweet["text"],
                        source=NewsSource.TWITTER,
                        url=f"https://twitter.com/i/status/{tweet['id']}",
                        published_at=published,
                        related_keywords=keywords,
                    )

                    # Use engagement as impact proxy
                    metrics = tweet.get("public_metrics", {})
                    engagement = (
                        metrics.get("retweet_count", 0) * 2 +
                        metrics.get("like_count", 0) +
                        metrics.get("reply_count", 0) * 3
                    )
                    item.impact_score = Decimal(str(min(1.0, engagement / 1000)))

                    items.append(item)
                except Exception as e:
                    logger.debug(f"Failed to parse tweet: {e}")

            return items

        except Exception as e:
            logger.error(f"Twitter fetch failed: {e}")
            return []

    async def _analyze_twitter_with_grok(
        self,
        keywords: list[str],
        hours_back: int,
    ) -> list[NewsItem]:
        """Use Grok API to analyze Twitter sentiment for keywords."""
        if not self._xai_api_key or not self._http_client:
            return []

        try:
            prompt = f"""Analyze recent Twitter/X discussion about: {', '.join(keywords)}

Search for the most relevant tweets from the past {hours_back} hours.
For each significant tweet or thread, provide:
1. Summary of the content
2. Sentiment (very_bearish, bearish, neutral, bullish, very_bullish)
3. Estimated impact (low, medium, high)
4. Whether it's breaking news

Format as JSON array with objects containing: summary, sentiment, impact, is_breaking, approximate_time"""

            response = await self._http_client.post(
                "https://api.x.ai/v1/chat/completions",
                json={
                    "model": "grok-3",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                headers={
                    "Authorization": f"Bearer {self._xai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            # Parse Grok's response
            items = self._parse_grok_twitter_analysis(content, keywords)
            return items

        except Exception as e:
            logger.error(f"Grok Twitter analysis failed: {e}")
            return []

    def _parse_grok_twitter_analysis(
        self,
        content: str,
        keywords: list[str],
    ) -> list[NewsItem]:
        """Parse Grok's Twitter analysis response."""
        import json

        items = []
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                tweets_data = json.loads(json_match.group())

                for tweet in tweets_data:
                    sentiment_map = {
                        "very_bearish": (NewsSentiment.VERY_BEARISH, Decimal("-0.8")),
                        "bearish": (NewsSentiment.BEARISH, Decimal("-0.4")),
                        "neutral": (NewsSentiment.NEUTRAL, Decimal("0.0")),
                        "bullish": (NewsSentiment.BULLISH, Decimal("0.4")),
                        "very_bullish": (NewsSentiment.VERY_BULLISH, Decimal("0.8")),
                    }

                    sentiment_str = tweet.get("sentiment", "neutral").lower()
                    sentiment, score = sentiment_map.get(
                        sentiment_str,
                        (NewsSentiment.NEUTRAL, Decimal("0.0"))
                    )

                    impact_map = {"low": Decimal("0.3"), "medium": Decimal("0.6"), "high": Decimal("0.9")}
                    impact = impact_map.get(tweet.get("impact", "medium").lower(), Decimal("0.5"))

                    item = NewsItem(
                        title=tweet.get("summary", "")[:100],
                        content=tweet.get("summary", ""),
                        source=NewsSource.TWITTER,
                        url=None,
                        published_at=datetime.utcnow(),  # Approximate
                        sentiment=sentiment,
                        sentiment_score=score,
                        confidence=Decimal("0.7"),  # Grok analysis confidence
                        impact_score=impact,
                        is_breaking=tweet.get("is_breaking", False),
                        related_keywords=keywords,
                    )
                    items.append(item)

        except json.JSONDecodeError:
            logger.warning("Failed to parse Grok response as JSON")
        except Exception as e:
            logger.error(f"Error parsing Grok analysis: {e}")

        return items

    async def _fetch_from_rss(
        self,
        keywords: list[str],
        hours_back: int,
    ) -> list[NewsItem]:
        """Fetch from financial RSS feeds."""
        if not self._http_client:
            return []

        # Financial news RSS feeds
        feeds = [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        ]

        items = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)

        for feed_url in feeds:
            try:
                response = await self._http_client.get(feed_url, timeout=10.0)
                if response.status_code != 200:
                    continue

                # Simple RSS parsing
                content = response.text

                # Extract items using regex (avoiding xml dependency)
                item_matches = re.findall(
                    r'<item>(.*?)</item>',
                    content,
                    re.DOTALL
                )

                for item_xml in item_matches[:20]:
                    title_match = re.search(r'<title>(.*?)</title>', item_xml)
                    desc_match = re.search(r'<description>(.*?)</description>', item_xml, re.DOTALL)
                    link_match = re.search(r'<link>(.*?)</link>', item_xml)
                    date_match = re.search(r'<pubDate>(.*?)</pubDate>', item_xml)

                    if not title_match:
                        continue

                    title = self._clean_html(title_match.group(1))
                    description = self._clean_html(desc_match.group(1)) if desc_match else ""

                    # Check keyword relevance
                    text = f"{title} {description}".lower()
                    if not any(kw.lower() in text for kw in keywords):
                        continue

                    # Parse date
                    published = datetime.utcnow()
                    if date_match:
                        try:
                            from email.utils import parsedate_to_datetime
                            published = parsedate_to_datetime(date_match.group(1)).replace(tzinfo=None)
                        except Exception:
                            pass

                    if published < cutoff:
                        continue

                    item = NewsItem(
                        title=title,
                        content=description,
                        source=NewsSource.RSS,
                        url=link_match.group(1) if link_match else None,
                        published_at=published,
                        related_keywords=keywords,
                    )
                    items.append(item)

            except Exception as e:
                logger.debug(f"RSS feed {feed_url} failed: {e}")

        return items

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and entities."""
        # Remove CDATA
        text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', text, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode common entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text.strip()

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicate news items based on title similarity."""
        unique = []
        seen_titles = set()

        for item in items:
            # Normalize title for comparison
            normalized = re.sub(r'[^a-z0-9]', '', item.title.lower())[:50]

            if normalized not in seen_titles:
                unique.append(item)
                seen_titles.add(normalized)

        return unique

    def _score_relevance(
        self,
        item: NewsItem,
        market_question: str,
        keywords: list[str],
    ) -> Decimal:
        """Score how relevant a news item is to a market."""
        score = Decimal("0.0")
        text = f"{item.title} {item.content}".lower()
        question_lower = market_question.lower()

        # Keyword matches
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
        score += Decimal(str(min(0.5, keyword_matches * 0.1)))

        # Question word overlap
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        text_words = set(re.findall(r'\b\w+\b', text))
        overlap = len(question_words & text_words) / max(len(question_words), 1)
        score += Decimal(str(overlap * 0.3))

        # Freshness boost
        if item.is_fresh:
            score += Decimal("0.2")

        # Breaking news boost
        if item.is_breaking:
            score += Decimal("0.3")

        return min(Decimal("1.0"), score)

    async def analyze_sentiment(
        self,
        text: str,
        llm_provider=None,
    ) -> tuple[NewsSentiment, Decimal, Decimal]:
        """
        Analyze sentiment of text, optionally using LLM.

        Returns (sentiment, score, confidence).
        """
        # Simple keyword-based sentiment if no LLM
        if llm_provider is None:
            return self._simple_sentiment(text)

        # Use LLM for better analysis
        try:
            from src.llm.base import Message, Role

            prompt = f"""Analyze the sentiment of this text regarding market/prediction outcomes.

Text: {text[:1000]}

Respond with JSON only:
{{"sentiment": "very_bearish|bearish|neutral|bullish|very_bullish", "score": -1.0 to 1.0, "confidence": 0.0 to 1.0}}"""

            response = await llm_provider.complete([
                Message(role=Role.USER, content=prompt)
            ], max_tokens=100)

            import json
            match = re.search(r'\{.*\}', response.content)
            if match:
                data = json.loads(match.group())
                sentiment_map = {
                    "very_bearish": NewsSentiment.VERY_BEARISH,
                    "bearish": NewsSentiment.BEARISH,
                    "neutral": NewsSentiment.NEUTRAL,
                    "bullish": NewsSentiment.BULLISH,
                    "very_bullish": NewsSentiment.VERY_BULLISH,
                }
                sentiment = sentiment_map.get(data.get("sentiment", "neutral"), NewsSentiment.NEUTRAL)
                score = Decimal(str(data.get("score", 0.0)))
                confidence = Decimal(str(data.get("confidence", 0.5)))
                return sentiment, score, confidence

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")

        return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> tuple[NewsSentiment, Decimal, Decimal]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        bullish_words = [
            "surge", "soar", "jump", "rally", "gain", "rise", "bullish",
            "optimistic", "growth", "profit", "success", "win", "beat",
            "exceed", "strong", "positive", "up", "higher", "increase",
        ]

        bearish_words = [
            "crash", "plunge", "drop", "fall", "decline", "bearish",
            "pessimistic", "loss", "fail", "miss", "weak", "negative",
            "down", "lower", "decrease", "concern", "fear", "risk",
        ]

        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return NewsSentiment.NEUTRAL, Decimal("0.0"), Decimal("0.3")

        score = Decimal(str((bullish_count - bearish_count) / max(total, 1)))

        if score > Decimal("0.5"):
            sentiment = NewsSentiment.VERY_BULLISH
        elif score > Decimal("0.2"):
            sentiment = NewsSentiment.BULLISH
        elif score < Decimal("-0.5"):
            sentiment = NewsSentiment.VERY_BEARISH
        elif score < Decimal("-0.2"):
            sentiment = NewsSentiment.BEARISH
        else:
            sentiment = NewsSentiment.NEUTRAL

        confidence = Decimal(str(min(0.8, total * 0.1 + 0.2)))

        return sentiment, score, confidence


# Global instance
_news_provider: Optional[NewsProvider] = None


def get_news_provider() -> NewsProvider:
    """Get or create the global news provider."""
    global _news_provider
    if _news_provider is None:
        _news_provider = NewsProvider()
    return _news_provider
