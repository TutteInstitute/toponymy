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
- Exponential backoff retry logic
- Rate limiting (1 request/second)
- Bulk/batch endpoints (500 papers per request)
- Inline field requests to minimize API calls
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

    def __init__(self, api_key: Optional[str] = None, cache_db: Optional[Path] = None):
        """
        Initialize the Semantic Scholar API client.

        Args:
            api_key: Optional API key for higher rate limits
            cache_db: Path to SQLite cache database (default: semantic_scholar_cache.db)
        """
        self.api_key = api_key
        self.cache_db = cache_db or Path("semantic_scholar_cache.db")
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
            # With API key: 3 sec to handle heavy batch requests (500 papers each)
            self.request_delay = 3.0
        else:
            # Without API key: 10 sec between requests to be very conservative with shared rate limit
            self.request_delay = 10.0

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
            Dictionary with counts of cached authors, papers, and citation queries
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
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.HTTPError, requests.exceptions.Timeout)
        ),
        wait=wait_exponential(
            multiplier=1, min=1, max=60
        ),  # Increased max wait for 429 errors
        stop=stop_after_attempt(8),  # More attempts for transient rate limits
    )
    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make HTTP request with exponential backoff retry logic.

        Retries on HTTP 429 (rate limit), 5xx (server errors), and timeouts.
        Backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
        Special handling for 429: respects Retry-After header if present
        """
        self._rate_limit()

        if method.upper() == "GET":
            response = self.session.get(url, **kwargs)
        elif method.upper() == "POST":
            response = self.session.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Special handling for 429 rate limit errors
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait_time = int(retry_after)
                print(
                    f"⚠ Rate limited. Waiting {wait_time} seconds (from Retry-After header)..."
                )
                time.sleep(wait_time)
                # Retry the request after waiting
                if method.upper() == "GET":
                    response = self.session.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = self.session.post(url, **kwargs)

        response.raise_for_status()
        return response.json()

    def _is_cached(self, table: str, key: str) -> bool:
        """Check if data exists in cache without retrieving it."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        if table == "authors":
            cursor.execute("SELECT 1 FROM authors WHERE author_id = ? LIMIT 1", (key,))
        elif table == "papers":
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1", (key,))
        elif table == "citations":
            cursor.execute(
                "SELECT 1 FROM citations WHERE cache_key = ? LIMIT 1", (key,)
            )
        else:
            conn.close()
            return False

        result = cursor.fetchone()
        conn.close()
        return result is not None

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
        self,
        author_id: str,
        max_papers: Optional[int] = None,
        include_citations: bool = True,
    ) -> List[Dict]:
        """
        Fetch papers for an author with optional inline citations.

        Note: Inline citations are limited to 10,000 per endpoint call (not per paper).
        For most authors this is fine, but if you need all citations from papers with
        10k+ citations, use get_paper_citations() separately for those papers.

        Args:
            author_id: Semantic Scholar author ID
            max_papers: Optional limit on number of papers to fetch
            include_citations: Include inline citations (may be truncated at 10k total)

        Returns:
            List of paper dictionaries, optionally with 'citations' field
        """
        cache_key = f"{author_id}_citations={include_citations}"
        cached = self._get_cached("authors", cache_key)
        if cached:
            assert isinstance(cached, dict), "Expected dict from authors cache"
            papers = cached["papers"]
            if max_papers:
                papers = papers[:max_papers]
            return papers

        papers = []
        offset = 0
        limit = 1000  # API maximum for pagination

        while True:
            url = f"{self.BASE_URL}/author/{author_id}/papers"
            fields = "paperId,title,abstract,year,citationCount"
            if include_citations:
                fields += ",citations.paperId"

            params = {"fields": fields, "offset": offset, "limit": limit}

            data = self._make_request("GET", url, params=params)
            batch = data.get("data", [])
            papers.extend(batch)

            if max_papers and len(papers) >= max_papers:
                papers = papers[:max_papers]
                break

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

            # Extract full citing paper objects and cache them
            for citation in batch:
                citing_paper = citation.get("citingPaper")
                if citing_paper and citing_paper.get("paperId"):
                    # Cache the paper
                    self._cache("papers", citing_paper["paperId"], citing_paper)
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

    def is_citations_cached(
        self,
        paper_id: str,
        citation_count: int,
        max_citations: Optional[int] = None,
        current_year: Optional[int] = None,
    ) -> bool:
        """Check if citations for this paper are fully cached."""
        # For papers < 10k, check single cache key
        effective_count = (
            min(citation_count, max_citations) if max_citations else citation_count
        )

        if effective_count < 10000:
            cache_key = f"{paper_id}_year=None_limit={max_citations}"
            return self._is_cached("citations", cache_key)

        # For papers >= 10k with time-slicing, check if the most recent year is cached
        # If the most recent year is cached, likely the whole set is cached
        if current_year is None:
            current_year = datetime.now().year

        # Check for the current year's cache entry as a proxy
        year_cache_key = f"{paper_id}_year={current_year}_limit=None"
        return self._is_cached("citations", year_cache_key)

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
        self, paper_ids: List[str], progress: bool = True
    ) -> List[Dict]:
        """
        Fetch full paper details using batch endpoint.

        Args:
            paper_ids: List of Semantic Scholar paper IDs
            progress: Show progress bar

        Returns:
            List of paper dictionaries
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
                iterator = tqdm(iterator, desc="Fetching papers")

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
    max_author_papers: Optional[int] = None,
    max_citations_per_paper: Optional[int] = None,
    max_total_citing_papers: Optional[int] = None,
    cache_db: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """
    Build a one-hop citation network for an author.

    This function uses an efficient two-phase approach:
    1. Fetches author papers with inline citations (1 request for most cases)
    2. Only queries individual papers separately if they exceed the inline limit

    For papers with >10k citations, automatic time-slicing is used:
    - Queries year-by-year from present backwards
    - When remaining citations <10k, does one final query for all remaining
    - Bypasses the Semantic Scholar API's 10k pagination limit per endpoint

    Note: If a single year has >10k citations (rare), that year will be truncated.

    Args:
        author_id: Semantic Scholar author ID
        api_key: Optional API key for higher rate limits
        max_author_papers: Limit on author's papers to process (None = all)
        max_citations_per_paper: Max citations to fetch per paper (None = all via time-slicing)
        max_total_citing_papers: Stop after collecting this many unique citing papers (None = all)
        cache_db: Path to SQLite cache database
        verbose: Print progress information

    Returns:
        Dictionary with:
            - author_id: The author ID
            - author_papers: List of author's papers
            - citing_papers: List of papers that cite the author's work
            - total_author_papers: Count of author papers
            - total_citing_papers: Count of citing papers
    """
    api = SemanticScholarAPI(api_key=api_key, cache_db=cache_db)

    if verbose:
        print(f"\nBuilding citation network for author {author_id}")
        print("=" * 60)
        if api_key:
            print(f"✓ Using API key (rate limit: 1 request/3 sec)")
        else:
            print(f"⚠ No API key - using conservative rate limit (1 request/10 sec)")
            print(
                f"  Get an API key at https://www.semanticscholar.org/product/api for faster queries"
            )

        # Show cache statistics
        cache_stats = api.get_cache_stats()
        print(f"\n📊 Cache status ({api.cache_db}):")
        print(f"  • {cache_stats['authors']} authors cached")
        print(f"  • {cache_stats['papers']:,} papers cached")
        print(f"  • {cache_stats['citation_queries']} citation queries cached")

    # Step 1: Get author's papers with inline citations
    if verbose:
        print("\nStep 1: Fetching author's publications with citations...")

    author_papers = api.get_author_papers(
        author_id, max_papers=max_author_papers, include_citations=True
    )

    if verbose:
        print(f"✓ Retrieved {len(author_papers)} papers")

    # Step 2: Extract citations and identify papers that need separate queries
    if verbose:
        print("\nStep 2: Extracting citations...")

    all_citing_papers: Dict[str, Dict] = {}  # paperId -> paper dict
    papers_needing_full_query = []
    total_inline_citations = 0

    for paper in author_papers:
        paper_id = paper.get("paperId")
        citation_count = paper.get("citationCount", 0)
        inline_citations = paper.get("citations", [])

        if inline_citations:
            total_inline_citations += len(inline_citations)

            # If we have fewer inline citations than reported, we need a full query
            if (
                len(inline_citations) < citation_count
                and max_citations_per_paper is None
            ):
                papers_needing_full_query.append((paper_id, citation_count))
            else:
                # Use inline citations (possibly limited by max_citations_per_paper)
                for idx, citation in enumerate(inline_citations):
                    if (
                        max_citations_per_paper is not None
                        and idx >= max_citations_per_paper
                    ):
                        break
                    cit_id = citation.get("paperId")
                    if cit_id and cit_id not in all_citing_papers:
                        # Store just the ID for inline citations (we'll fetch details later)
                        all_citing_papers[cit_id] = {"paperId": cit_id}
        elif citation_count > 0:
            # Paper has citations but we didn't get any inline - need full query
            papers_needing_full_query.append((paper_id, citation_count))

    # If we got no inline citations at all, the API probably didn't return them
    # Fall back to querying all papers with citations
    if total_inline_citations == 0 and papers_needing_full_query:
        if verbose:
            print(f"  No inline citations returned by API")
            print(f"  Will query {len(papers_needing_full_query)} papers individually")
    elif verbose and papers_needing_full_query:
        total_citations_to_fetch = sum(count for _, count in papers_needing_full_query)
        print(f"  Found {len(all_citing_papers)} citations from inline data")
        print(
            f"  {len(papers_needing_full_query)} papers need separate queries ({total_citations_to_fetch:,} total citations)"
        )
    elif verbose:
        print(f"  Extracted {len(all_citing_papers)} citations from inline data")

    # Step 3: Query papers that need full citation lists (WITH METADATA)
    if papers_needing_full_query:
        if verbose:
            print(
                f"\nStep 3: Fetching citations with full metadata for {len(papers_needing_full_query)} papers..."
            )

        for idx, (paper_id, citation_count) in enumerate(papers_needing_full_query, 1):
            if verbose:
                # Check if this paper's citations are cached
                is_cached = api.is_citations_cached(
                    paper_id, citation_count, max_citations_per_paper
                )
                cached_label = " (cached)" if is_cached else ""

                warning = ""
                if citation_count >= 10000 and not is_cached:
                    warning = f" (using time-slicing to get all {citation_count:,} citations with metadata)"

                print(
                    f"  [{idx}/{len(papers_needing_full_query)}] Fetching citations for paper with {citation_count:,} citations{warning}{cached_label}"
                )

            # Use smart time-slicing - now returns full paper dicts
            paper_citations = api.get_paper_citations_with_time_slicing(
                paper_id, citation_count, max_citations=max_citations_per_paper
            )

            # Merge into our collection (papers are already cached in get_paper_citations)
            for citing_paper in paper_citations:
                cit_id = citing_paper["paperId"]
                all_citing_papers[cit_id] = citing_paper

            # Check if we've reached the total limit
            if (
                max_total_citing_papers is not None
                and len(all_citing_papers) >= max_total_citing_papers
            ):
                if verbose:
                    print(
                        f"  Reached limit of {max_total_citing_papers} total citing papers"
                    )
                break

    if verbose:
        print(
            f"\n✓ Collected {len(all_citing_papers)} unique citing papers with metadata"
        )

    # Step 4: Fetch any remaining papers that only have IDs (from inline citations)
    papers_with_full_data = [p for p in all_citing_papers.values() if "title" in p]
    papers_needing_fetch = [
        pid for pid, p in all_citing_papers.items() if "title" not in p
    ]

    if papers_needing_fetch:
        if verbose:
            print(
                f"\nStep 4: Fetching details for {len(papers_needing_fetch)} papers from inline citations..."
            )
        fetched_papers = api.batch_get_papers(papers_needing_fetch, progress=verbose)
        for paper in fetched_papers:
            if paper and paper.get("paperId"):
                all_citing_papers[paper["paperId"]] = paper

    # Apply total limit if specified
    if max_total_citing_papers is not None:
        citing_papers = list(all_citing_papers.values())[:max_total_citing_papers]
    else:
        citing_papers = list(all_citing_papers.values())

    # Filter papers with title and abstract
    citing_papers = [
        p
        for p in citing_papers
        if p.get("title") and p.get("abstract") and len(p.get("abstract", "")) > 50
    ]

    if verbose:
        print(f"✓ Retrieved {len(citing_papers)} papers with title + abstract\n")

    return {
        "author_id": author_id,
        "author_papers": author_papers,
        "citing_papers": citing_papers,
        "total_author_papers": len(author_papers),
        "total_citing_papers": len(citing_papers),
    }
