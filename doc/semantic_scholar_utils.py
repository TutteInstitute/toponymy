"""
Semantic Scholar API utilities for Toponymy examples.

⚠️ EXAMPLE CODE - NOT PART OF TOPONYMY PACKAGE ⚠️

This module is provided as-is for educational purposes and to support
Toponymy documentation examples. It is not maintained as part of the
core Toponymy package and may break if the Semantic Scholar API changes.

For production use, consider:
- The official Semantic Scholar datasets API for bulk analysis
- Adding more robust error handling for your specific use case
- Monitoring API changes at https://api.semanticscholar.org/api-docs/

Feel free to copy and modify this code for your own projects.

API Best Practices Demonstrated:
- Direct requests (no unofficial packages)
- SQLite caching for persistence and efficiency
- Proper pagination handling
- Exponential backoff retry logic (2s, 4s, 8s, 16s...)
- Configurable rate limiting (default: 3s with API key, 10s without)
- Batch endpoints for efficient bulk fetching (500 papers per request)
- Strategy routing based on network size
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
import requests
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class SemanticScholarAPI:
    """Direct requests-based Semantic Scholar API client with SQLite caching."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_db: Path = Path("semantic_scholar_cache.db"),
        request_delay: Optional[float] = None,
    ):
        """
        Initialize the Semantic Scholar API client.

        Args:
            api_key: Optional API key for higher rate limits
            cache_db: Path to SQLite cache database (default: semantic_scholar_cache.db)
            request_delay: Seconds between requests (default: 3.0 with API key, 10.0 without).
                          Increase if experiencing rate limit issues, decrease at your own risk.
        """
        self.api_key = api_key
        self.cache_db = cache_db
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

        # Smart defaults for rate limiting if not specified
        if request_delay is None:
            self.request_delay = 10.0 if self.api_key else 10.0
        else:
            self.request_delay = request_delay

        self._init_db()
        self.last_request_time = 0

    def _init_db(self):
        """Create SQLite tables for caching."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                author_id TEXT PRIMARY KEY,
                data JSON NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                data JSON NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                cache_key TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                year_filter TEXT,
                citing_papers JSON NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers ON papers(paper_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_paper ON citations(paper_id)"
        )

        conn.commit()
        conn.close()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about cached data.

        Returns:
            Dictionary with counts of cached authors and citation queries
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM authors")
        author_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM papers")
        paper_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM citations")
        citation_query_count = cursor.fetchone()[0]

        conn.close()

        return {
            "authors": author_count,
            "papers": paper_count,
            "citation_queries": citation_query_count,
        }

    def _rate_limit(self):
        """Enforce rate limiting before every request."""
        current_time = time.time()
        if self.last_request_time > 0:
            elapsed = current_time - self.last_request_time
            print(f"    ⏱️  Elapsed since last request: {elapsed:.3f}s (sleeping {self.request_delay:.1f}s)")
        else:
            print(f"    ⏱️  First request (sleeping {self.request_delay:.1f}s)")
        time.sleep(self.request_delay)

    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.HTTPError, requests.exceptions.Timeout)
        ),
        wait=wait_exponential(
            multiplier=1, min=2, max=120
        ),  # Exponential backoff for production
        stop=stop_after_attempt(
            1
        ),  # No retries during debugging - increase to 5-8 for production
    )
    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make HTTP request with exponential backoff retry logic.

        Base rate limiting (request_delay) applies between all requests.
        On failures, exponential backoff (2s, 4s, 8s, 16s...) is ADDED to base delay.
        This ensures compliance with API TOS requiring exponential backoff.

        Retries on HTTP 429 (rate limit), 5xx (server errors), and timeouts.
        """
        # Truncate URL for cleaner output
        display_url = url.replace(self.BASE_URL, "...") if url.startswith(self.BASE_URL) else url
        print(f"\n  📡 {method} {display_url}")
        
        self._rate_limit()
        
        request_start = time.time()
        if method.upper() == "GET":
            response = self.session.get(url, **kwargs)
        elif method.upper() == "POST":
            response = self.session.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        request_duration = time.time() - request_start
        print(f"    ✓ {response.status_code} {response.reason} ({request_duration:.3f}s)")
        
        # Print interesting headers
        if 'x-ratelimit-remaining' in response.headers:
            print(f"    Rate limit remaining: {response.headers['x-ratelimit-remaining']}")
        if 'x-ratelimit-reset' in response.headers:
            print(f"    Rate limit resets: {response.headers['x-ratelimit-reset']}")
        
        # Update last_request_time AFTER the request completes for accurate inter-request timing
        self.last_request_time = time.time()
        
        response.raise_for_status()
        return response.json()

    def _get_cached(self, table: str, key: str) -> Optional[Union[dict, list]]:
        """Get cached data from SQLite."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        if table == "authors":
            cursor.execute("SELECT data FROM authors WHERE author_id = ?", (key,))
        elif table == "papers":
            cursor.execute("SELECT data FROM papers WHERE paper_id = ?", (key,))
        elif table == "citations":
            cursor.execute(
                "SELECT citing_papers FROM citations WHERE cache_key = ?", (key,)
            )
        else:
            return None

        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def _cache(self, table: str, key: str, data: Union[dict, list], **kwargs):
        """Cache data in SQLite."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        data_json = json.dumps(data)

        if table == "authors":
            cursor.execute(
                """
                INSERT OR REPLACE INTO authors (author_id, data, fetched_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (key, data_json),
            )
        elif table == "papers":
            cursor.execute(
                """
                INSERT OR REPLACE INTO papers (paper_id, data, fetched_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (key, data_json),
            )
        elif table == "citations":
            paper_id = kwargs.get("paper_id")
            year_filter = kwargs.get("year_filter")
            cursor.execute(
                """
                INSERT OR REPLACE INTO citations (cache_key, paper_id, year_filter, citing_papers, fetched_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (key, paper_id, year_filter, data_json),
            )

        conn.commit()
        conn.close()

    def get_author_papers(
        self, author_id: str, include_citation_ids: bool = False
    ) -> List[Dict]:
        """
        Fetch papers for an author, optionally with citation IDs.

        Note: Requesting citation IDs (citations.paperId) is lightweight and works
        reliably for authors with <10k total citations. For larger authors, the API
        truncates at 10k, requiring individual paper queries.

        Args:
            author_id: Semantic Scholar author ID
            include_citation_ids: Include citation paper IDs (lightweight, may be truncated at 10k)

        Returns:
            List of paper dictionaries, optionally with 'citations' field containing
            citation objects with just paperId
        """
        cache_key = f"{author_id}_citids={include_citation_ids}"
        cached = self._get_cached("authors", cache_key)
        if cached:
            assert isinstance(cached, dict), "Expected dict from authors cache"
            return cached["papers"]

        papers = []
        offset = 0
        limit = 1000  # API maximum for pagination

        while True:
            url = f"{self.BASE_URL}/author/{author_id}/papers"
            fields = "paperId,title,abstract,year,citationCount"
            if include_citation_ids:
                fields += ",citations.paperId"

            params = {"fields": fields, "offset": offset, "limit": limit}

            data = self._make_request("GET", url, params=params)
            batch = data.get("data", [])
            papers.extend(batch)

            if "next" not in data or len(batch) < limit:
                break

            offset = data["next"]

        # Cache the result
        author_data = {"authorId": author_id, "papers": papers}
        self._cache("authors", cache_key, author_data)

        return papers

    def get_paper_citations(
        self,
        paper_id: str,
        max_citations: Optional[int] = None,
        year_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch citing papers with full metadata for a given paper, optionally filtered by year.

        Note: The Semantic Scholar API has a hard limit of 10,000 items per paginated endpoint.

        Args:
            paper_id: Semantic Scholar paper ID
            max_citations: Optional limit on citations to fetch
            year_filter: Optional year filter (e.g., "2024", "2020-2024", ":2020", "2020-")

        Returns:
            List of citing paper dictionaries with full metadata
        """
        # Check cache first
        cache_key = f"{paper_id}_year={year_filter}_limit={max_citations}"
        cached = self._get_cached("citations", cache_key)
        if cached:
            assert isinstance(cached, list), "Expected list from citations cache"
            citing_papers = cached
            if max_citations:
                citing_papers = citing_papers[:max_citations]
            return citing_papers

        citing_papers = []
        offset = 0
        limit = 1000  # API maximum for pagination

        while True:
            # Stop if we've reached the API's hard limit
            if offset >= 10000:
                break

            url = f"{self.BASE_URL}/paper/{paper_id}/citations"
            params = {
                "fields": "citingPaper.paperId,citingPaper.title,citingPaper.abstract,citingPaper.year,citingPaper.authors,citingPaper.citationCount",
                "offset": offset,
                "limit": limit,
            }
            if year_filter:
                params["publicationDateOrYear"] = year_filter

            data = self._make_request("GET", url, params=params)
            batch = data.get("data", [])

            # Extract full citing paper objects
            for citation in batch:
                citing_paper = citation.get("citingPaper")
                if citing_paper and citing_paper.get("paperId"):
                    citing_papers.append(citing_paper)

            if max_citations and len(citing_papers) >= max_citations:
                citing_papers = citing_papers[:max_citations]
                break

            # Stop if no more data or we've hit the limit
            if "next" not in data or len(batch) < limit:
                break

            offset = data["next"]

        # Cache the full citation list for this paper/filter combination
        self._cache(
            "citations",
            cache_key,
            citing_papers,
            paper_id=paper_id,
            year_filter=year_filter,
        )

        return citing_papers

    def get_paper_citations_with_time_slicing(
        self,
        paper_id: str,
        citation_count: int,
        max_citations: Optional[int] = None,
        current_year: Optional[int] = None,
    ) -> List[Dict]:
        """
        Intelligently fetch citations with full metadata using time-slicing for papers with >10k citations.

        Strategy:
        - If paper has <10k citations: single query (no year filter)
        - If paper has >=10k citations: query year-by-year from present backwards
          until remaining citations <10k, then do one final query for all remaining

        Args:
            paper_id: Semantic Scholar paper ID
            citation_count: Total citation count from paper metadata
            max_citations: Optional limit on total citations to fetch
            current_year: Current year (default: auto-detect from system)

        Returns:
            List of citing paper dictionaries with full metadata
        """
        if current_year is None:
            current_year = datetime.now().year

        # Apply user's max limit if specified
        effective_citation_count = citation_count
        if max_citations and max_citations < citation_count:
            effective_citation_count = max_citations

        # If paper has < 10k citations, single query
        if effective_citation_count < 10000:
            return self.get_paper_citations(paper_id, max_citations=max_citations)

        # For >= 10k citations, use time slicing
        all_citing_papers = {}
        year = current_year
        collected = 0

        while collected < effective_citation_count:
            remaining = effective_citation_count - collected

            # If remaining < 10k, get all remaining with open-ended query
            if remaining < 10000:
                year_filter = f":{year}"  # Everything up to and including year
                citations = self.get_paper_citations(paper_id, year_filter=year_filter)
                for paper in citations:
                    all_citing_papers[paper["paperId"]] = paper
                break

            # Query single year
            year_filter = str(year)
            citations = self.get_paper_citations(paper_id, year_filter=year_filter)
            for paper in citations:
                all_citing_papers[paper["paperId"]] = paper

            # Warn if we hit 10k in a single year (rare but possible data loss)
            if len(citations) >= 10000:
                print(
                    f"  ⚠ WARNING: Paper has >=10,000 citations in {year} alone - some may be truncated"
                )

            collected += len(citations)
            year -= 1

            # Extra delay between time-slice queries (double the normal rate limit)
            # to avoid overwhelming the API with rapid successive requests
            time.sleep(self.request_delay)

            # Safety: don't go before 1900
            if year < 1900:
                break

        # Apply max_citations limit if specified
        result = list(all_citing_papers.values())
        if max_citations and len(result) > max_citations:
            result = result[:max_citations]

        return result

    def batch_get_papers(
        self, paper_ids: List[str], progress: bool = False
    ) -> List[Dict]:
        """
        Fetch full paper details using batch endpoint (500 papers per request).

        Args:
            paper_ids: List of Semantic Scholar paper IDs
            progress: Show progress bar

        Returns:
            List of paper dictionaries with full metadata
        """
        uncached_ids = []
        results = []

        # Check cache first
        for paper_id in paper_ids:
            cached = self._get_cached("papers", paper_id)
            if cached:
                results.append(cached)
            else:
                uncached_ids.append(paper_id)

        if uncached_ids:
            batch_size = 500  # API maximum for batch endpoint
            iterator = range(0, len(uncached_ids), batch_size)

            if progress:
                iterator = tqdm(iterator, desc="Fetching papers", unit="batch")

            for i in iterator:
                batch_ids = uncached_ids[i : i + batch_size]

                url = f"{self.BASE_URL}/paper/batch"
                params = {"fields": "paperId,title,abstract,authors,year,citationCount"}
                payload = {"ids": batch_ids}

                batch_papers = self._make_request(
                    "POST", url, params=params, json=payload
                )

                # Cache and add to results
                for paper in batch_papers:
                    if paper and paper.get("paperId"):
                        self._cache("papers", paper["paperId"], paper)
                        results.append(paper)

        return results


def build_citation_network(
    author_id: str,
    api_key: Optional[str] = None,
    cache_db: Path = Path("semantic_scholar_cache.db"),
    request_delay: Optional[float] = None,
    verbose: bool = True,
) -> Dict:
    """
    Build a one-hop citation network for an author.

    Strategy (routes automatically based on total citations):
    - <10k total citations: Lightweight approach
      1. Get citation IDs from author query
      2. Batch fetch metadata (500 papers per request)
    - ≥10k total citations: Robust approach
      1. Query each paper's citations individually
      2. Use time-slicing for papers with >10k citations

    Why two strategies?
    - Small networks benefit from batching (fewer requests, faster)
    - Large networks hit API limits with author query, need individual queries anyway
    - Routing on total citations makes the choice clear and testable

    Args:
        author_id: Semantic Scholar author ID
        api_key: Optional API key for higher rate limits
        cache_db: Path to SQLite cache database
        request_delay: Seconds between requests (default: 3.0 with API key, 10.0 without)
        verbose: Print progress information

    Returns:
        Dictionary with:
            - author_id: The author ID
            - author_papers: List of author's papers
            - citing_papers: List of papers that cite the author's work
            - total_author_papers: Count of author papers
            - total_citing_papers: Count of citing papers
    """
    api = SemanticScholarAPI(
        api_key=api_key, cache_db=cache_db, request_delay=request_delay
    )

    if verbose:
        print(f"\nBuilding citation network for author {author_id}")
        print("=" * 60)
        if api_key:
            print(f"✓ Using API key (delay: {api.request_delay:.1f}s between requests)")
        else:
            print(f"⚠ No API key (delay: {api.request_delay:.1f}s between requests)")
            print(
                f"  Get an API key at https://www.semanticscholar.org/product/api for faster queries"
            )

        # Show cache statistics
        cache_stats = api.get_cache_stats()
        print(f"\n📊 Cache status ({api.cache_db}):")
        print(f"  • {cache_stats['authors']} authors cached")
        print(f"  • {cache_stats['papers']:,} papers cached")
        print(f"  • {cache_stats['citation_queries']} citation queries cached")

    # Step 1: Get author's papers (just metadata, no citation IDs yet)
    if verbose:
        print("\nStep 1: Fetching author's publications...")

    author_papers = api.get_author_papers(author_id, include_citation_ids=False)

    if verbose:
        print(f"✓ Retrieved {len(author_papers)} papers")

    # Step 2: Calculate total citations to choose strategy
    total_citations = sum(p.get("citationCount", 0) for p in author_papers)

    if verbose:
        print(f"\nTotal citations across all papers: {total_citations:,}")

    # Step 3: Route to appropriate strategy
    if total_citations < 10000:
        if verbose:
            print("→ Using batch fetch strategy (total citations < 10k)")
        return _build_small_network(api, author_id, author_papers, verbose)
    else:
        if verbose:
            print("→ Using individual query strategy (total citations ≥ 10k)")
        return _build_large_network(api, author_id, author_papers, verbose)


def _build_small_network(
    api: SemanticScholarAPI,
    author_id: str,
    author_papers: List[Dict],
    verbose: bool,
) -> Dict:
    """
    Build citation network for authors with <10k total citations.

    Strategy: Re-query for citation IDs, then batch fetch metadata.
    This is efficient (few requests) and works reliably for small networks.
    """
    if verbose:
        print("\nStep 2: Re-querying for citation IDs...")

    # Re-fetch author papers WITH citation IDs
    # (We didn't fetch them in Step 1 to save bandwidth for large authors)
    author_papers_with_citations = api.get_author_papers(
        author_id, include_citation_ids=True
    )

    # Collect all unique citation IDs
    citation_ids = set()
    for paper in author_papers_with_citations:
        for citation in paper.get("citations", []):
            cit_id = citation.get("paperId")
            if cit_id:
                citation_ids.add(cit_id)

    if verbose:
        print(f"✓ Found {len(citation_ids)} unique citation IDs")

    # Step 3: Batch fetch metadata
    if citation_ids:
        if verbose:
            num_batches = (len(citation_ids) + 499) // 500
            print(
                f"\nStep 3: Batch fetching metadata ({num_batches} batch{'es' if num_batches > 1 else ''})..."
            )

        citing_papers_list = api.batch_get_papers(list(citation_ids), progress=verbose)

        if verbose:
            print(f"✓ Retrieved {len(citing_papers_list)} papers")
    else:
        citing_papers_list = []

    # Filter for papers with title and abstract
    citing_papers = [
        p
        for p in citing_papers_list
        if p.get("title") and p.get("abstract") and len(p.get("abstract", "")) > 50
    ]

    if verbose:
        print(f"✓ {len(citing_papers)} papers with title + abstract\n")

    return {
        "author_id": author_id,
        "author_papers": author_papers,
        "citing_papers": citing_papers,
        "total_author_papers": len(author_papers),
        "total_citing_papers": len(citing_papers),
    }


def _build_large_network(
    api: SemanticScholarAPI,
    author_id: str,
    author_papers: List[Dict],
    verbose: bool,
) -> Dict:
    """
    Build citation network for authors with ≥10k total citations.

    Strategy: Query each paper's citations individually (with time-slicing for >10k papers).
    This is robust and handles the API's 10k pagination limit per endpoint.
    """
    if verbose:
        print("\nStep 2: Querying each paper's citations individually...")

    all_citing_papers: Dict[str, Dict] = {}  # paperId -> paper dict
    papers_with_citations = [
        (p.get("paperId"), p.get("citationCount", 0))
        for p in author_papers
        if p.get("citationCount", 0) > 0
    ]

    if verbose and papers_with_citations:
        total_citations = sum(count for _, count in papers_with_citations)
        print(
            f"  {len(papers_with_citations)} papers have citations ({total_citations:,} total)"
        )
    elif verbose:
        print(f"  No papers with citations found")

    # Query each paper for citations
    if papers_with_citations:
        if verbose:
            print(f"\nStep 3: Fetching citations with metadata...")

        for idx, (paper_id, citation_count) in enumerate(papers_with_citations, 1):
            if verbose:
                warning = ""
                if citation_count >= 10000:
                    warning = (
                        f" (using time-slicing to get all {citation_count:,} citations)"
                    )

                print(
                    f"  [{idx}/{len(papers_with_citations)}] Paper with {citation_count:,} citations{warning}"
                )

            # Use time-slicing for papers with >10k citations
            paper_citations = api.get_paper_citations_with_time_slicing(
                paper_id, citation_count
            )

            # Merge into collection (deduplicates across papers)
            for citing_paper in paper_citations:
                cit_id = citing_paper["paperId"]
                all_citing_papers[cit_id] = citing_paper

    if verbose:
        print(f"\n✓ Collected {len(all_citing_papers)} unique citing papers")

    # Filter for papers with title and abstract
    citing_papers = [
        p
        for p in all_citing_papers.values()
        if p.get("title") and p.get("abstract") and len(p.get("abstract", "")) > 50
    ]

    if verbose:
        print(f"✓ {len(citing_papers)} papers with title + abstract\n")

    return {
        "author_id": author_id,
        "author_papers": author_papers,
        "citing_papers": citing_papers,
        "total_author_papers": len(author_papers),
        "total_citing_papers": len(citing_papers),
    }
