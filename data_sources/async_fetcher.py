"""
Async Concurrent Fetcher Module - Supports web content retrieval and search result caching.

Features:
1. Asynchronous concurrent fetching of multiple URLs
2. Google Search API async calls
3. URL content caching and search result caching
4. Smart encoding detection and HTML parsing
"""

import asyncio
import aiohttp
import requests
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import chardet
import re
from concurrent.futures import ThreadPoolExecutor
import time


class RateLimiter:
    """Token bucket rate limiter (for Jina API)"""
    
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        Args:
            rate_limit: Maximum number of requests allowed within the time window
            time_window: Time window size in seconds (default: 60)
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = None

    async def acquire(self):
        """Acquire a token; waits if no tokens are available"""
        if self.lock is None:
            self.lock = asyncio.Lock()

        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(5)
            
            self.tokens -= 1
            return True


# Global Jina rate limiter
_jina_rate_limiter = RateLimiter(rate_limit=100)  # 100 per minute


class AsyncFetcher:
    """
    Async Concurrent Fetcher
    
    Features:
    - Concurrency control: limits max concurrent connections
    - Caching: URL content cache + search result cache
    - Smart parsing: automatic encoding detection + HTML text extraction
    - Jina AI support: better handling of JS-rendered pages
    - Error handling: timeout retry + exception catching
    """
    
    # Error indicators (used to detect invalid content)
    ERROR_INDICATORS = [
        'limit exceeded',
        'Error fetching',
        'Invalid bearer token',
        'HTTP error occurred',
        'Connection error',
        'Request timed out',
        'Please turn on Javascript',
        'Enable JavaScript',
        'Please enable cookies',
        'Access Denied',
        '403 Forbidden',
        '404 Not Found',
    ]
    
    # Request headers
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(
        self, 
        max_concurrent: int = 20, 
        timeout: int = 30,
        max_content_length: int = 15000,
        use_jina: bool = False,
        jina_api_key: Optional[str] = None
    ):
        """
        Initialize the async fetcher.
        
        Args:
            max_concurrent: Maximum number of concurrent connections
            timeout: Request timeout in seconds
            max_content_length: Maximum content length per URL
            use_jina: Whether to use Jina AI for parsing (better for JS pages)
            jina_api_key: Jina API Key (get from https://jina.ai/)
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        
        # Caches
        self.url_cache: Dict[str, str] = {}
        self.search_cache: Dict[str, List[Dict]] = {}
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    async def fetch_urls_async(self, urls: List[str], use_jina: Optional[bool] = None) -> Dict[str, str]:
        """
        Asynchronously fetch content from multiple URLs.
        
        Args:
            urls: List of URLs
            use_jina: Whether to use Jina (None uses instance default)
            
        Returns:
            {url: content} dictionary
        """
        if not urls:
            return {}
        
        # Temporarily switch Jina setting
        original_use_jina = self.use_jina
        if use_jina is not None:
            self.use_jina = use_jina
        
        # Filter already cached URLs
        urls_to_fetch = [u for u in urls if u not in self.url_cache]
        
        if urls_to_fetch:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout, 
                headers=self.HEADERS
            ) as session:
                tasks = [self._fetch_single_url(url, session) for url in urls_to_fetch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, content in zip(urls_to_fetch, results):
                    if isinstance(content, str) and not self._has_error(content):
                        self.url_cache[url] = content
                    elif isinstance(content, Exception):
                        self.url_cache[url] = f"Error: {str(content)}"
        
        # Restore original setting
        self.use_jina = original_use_jina
        
        return {u: self.url_cache.get(u, "") for u in urls}
    
    def fetch_urls(self, urls: List[str], use_jina: Optional[bool] = None) -> Dict[str, str]:
        """
        Synchronous wrapper: fetch content from multiple URLs.
        
        Args:
            urls: List of URLs
            use_jina: Whether to use Jina (None uses instance default)
            
        Returns:
            {url: content} dictionary
        """
        return asyncio.run(self.fetch_urls_async(urls, use_jina))
    
    async def google_search_async(
        self, 
        query: str, 
        api_key: str, 
        cse_id: str, 
        num: int = 10,
        fetch_content: bool = True
    ) -> List[Dict]:
        """
        Async Google Search + optional content fetching.
        
        Args:
            query: Search keywords
            api_key: Google API Key
            cse_id: Google Custom Search Engine ID
            num: Number of results to return
            fetch_content: Whether to fetch full page content
            
        Returns:
            List of search results, each containing title, snippet, url, content (optional)
        """
        cache_key = f"google:{query}:{num}"
        
        # Check cache
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Execute search
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(num, 10),  # Google API returns at most 10 results
        }
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        
                        for item in items:
                            results.append({
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                                "url": item.get("link", ""),
                                "content": "",  # To be filled
                            })
        except Exception as e:
            print(f"Google search failed: {e}")
            return []
        
        # Fetch full page content
        if fetch_content and results:
            urls = [r["url"] for r in results if r["url"]]
            contents = await self.fetch_urls_async(urls)
            
            for r in results:
                r["content"] = contents.get(r["url"], r["snippet"])
        
        # Cache results
        self.search_cache[cache_key] = results
        return results
    
    def google_search(
        self, 
        query: str, 
        api_key: str, 
        cse_id: str, 
        num: int = 10,
        fetch_content: bool = True
    ) -> List[Dict]:
        """Synchronous wrapper: Google Search"""
        return asyncio.run(self.google_search_async(query, api_key, cse_id, num, fetch_content))
    
    def clear_cache(self):
        """Clear all caches"""
        self.url_cache.clear()
        self.search_cache.clear()
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    async def _fetch_single_url(self, url: str, session: aiohttp.ClientSession) -> str:
        """
        Fetch content from a single URL.
        
        Args:
            url: Target URL
            session: aiohttp session
            
        Returns:
            Extracted text content
        """
        try:
            # Use Jina AI for parsing
            if self.use_jina and self.jina_api_key:
                return await self._fetch_with_jina(url, session)
            
            # Standard HTTP fetch
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Error: HTTP {response.status}"
                
                # Read content
                content = await response.read()
                
                # Detect encoding
                detected = chardet.detect(content)
                encoding = detected.get('encoding', 'utf-8') or 'utf-8'
                
                try:
                    html = content.decode(encoding, errors='replace')
                except:
                    html = content.decode('utf-8', errors='replace')
                
                # Check if content is valid
                if self._has_error(html) or len(html) < 100:
                    return f"Error: Invalid content from {url}"
                
                # Parse HTML to extract text
                text = self._extract_text(html)
                
                return text[:self.max_content_length]
                
        except asyncio.TimeoutError:
            return f"Error: Timeout fetching {url}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _fetch_with_jina(self, url: str, session: aiohttp.ClientSession) -> str:
        """
        Fetch web page using Jina AI.
        
        Advantages:
        - Better handling of JavaScript-rendered pages
        - Returns clean Markdown format
        - Bypasses some anti-scraping measures
        """
        try:
            # Rate limiting
            await _jina_rate_limiter.acquire()
            
            jina_headers = {
                'Authorization': f'Bearer {self.jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            
            jina_url = f'https://r.jina.ai/{url}'
            
            async with session.get(jina_url, headers=jina_headers) as response:
                if response.status != 200:
                    # Jina failed, fallback to standard fetch
                    return await self._fetch_without_jina(url, session)
                
                text = await response.text()
                
                # Remove URLs from Markdown (reduce noise)
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", text)
                text = text.replace('---', '-').replace('===', '=').replace('   ', ' ')
                
                if self._has_error(text) or len(text) < 100:
                    return f"Error: Jina returned invalid content"
                
                return text[:self.max_content_length]
                
        except Exception as e:
            # Jina error, try standard fetch
            return await self._fetch_without_jina(url, session)
    
    async def _fetch_without_jina(self, url: str, session: aiohttp.ClientSession) -> str:
        """Standard HTTP fetch (Jina fallback)"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Error: HTTP {response.status}"
                
                content = await response.read()
                detected = chardet.detect(content)
                encoding = detected.get('encoding', 'utf-8') or 'utf-8'
                
                try:
                    html = content.decode(encoding, errors='replace')
                except:
                    html = content.decode('utf-8', errors='replace')
                
                if self._has_error(html) or len(html) < 100:
                    return f"Error: Invalid content from {url}"
                
                text = self._extract_text(html)
                return text[:self.max_content_length]
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_text(self, html: str) -> str:
        """
        Extract plain text from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted plain text
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
        except:
            try:
                soup = BeautifulSoup(html, 'html.parser')
            except:
                return html[:self.max_content_length]
        
        # Remove useless tags
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 
                                   'aside', 'meta', 'link', 'noscript', 'iframe']):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _has_error(self, content: str) -> bool:
        """Check if content contains error indicators"""
        if not content or len(content) < 50:
            return True
        
        content_lower = content.lower()
        word_count = len(content.split())
        
        # If content is short and contains error indicators
        if word_count < 100:
            for indicator in self.ERROR_INDICATORS:
                if indicator.lower() in content_lower:
                    return True
        
        return False


# ============================================================================
# Convenience Functions
# ============================================================================

# Global fetcher instance (shared cache)
_global_fetcher: Optional[AsyncFetcher] = None

def get_fetcher(use_jina: bool = False, jina_api_key: Optional[str] = None) -> AsyncFetcher:
    """
    Get global fetcher instance.
    
    Args:
        use_jina: Whether to use Jina AI
        jina_api_key: Jina API Key
    """
    global _global_fetcher
    if _global_fetcher is None:
        import os
        jina_key = jina_api_key or os.environ.get("JINA_API_KEY")
        _global_fetcher = AsyncFetcher(
            use_jina=use_jina,
            jina_api_key=jina_key
        )
    elif jina_api_key:
        # Update Jina settings
        _global_fetcher.use_jina = use_jina
        _global_fetcher.jina_api_key = jina_api_key
    return _global_fetcher

def fetch_urls(urls: List[str], use_jina: bool = False) -> Dict[str, str]:
    """
    Convenience function: fetch multiple URLs.
    
    Args:
        urls: List of URLs
        use_jina: Whether to use Jina AI (requires JINA_API_KEY env var)
    """
    return get_fetcher().fetch_urls(urls, use_jina=use_jina)

def google_search_with_content(
    query: str, 
    api_key: str, 
    cse_id: str, 
    num: int = 10,
    use_jina: bool = False
) -> List[Dict]:
    """
    Convenience function: Google search and fetch content.
    
    Args:
        use_jina: Whether to use Jina AI for web page fetching
    """
    return get_fetcher().google_search(query, api_key, cse_id, num, fetch_content=True)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import time
    
    # Test URL fetching
    test_urls = [
        "https://huggingface.co/",
        "https://www.reddit.com/r/LocalLLaMA/",
        "https://arxiv.org/",
    ]
    
    fetcher = AsyncFetcher(max_concurrent=10, timeout=20)
    
    print("Testing async concurrent fetching...")
    start = time.time()
    results = fetcher.fetch_urls(test_urls)
    elapsed = time.time() - start
    
    print(f"Fetched {len(test_urls)} URLs in {elapsed:.2f} seconds")
    for url, content in results.items():
        print(f"  {url}: {len(content)} chars")
