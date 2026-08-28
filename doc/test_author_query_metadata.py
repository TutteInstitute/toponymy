#!/usr/bin/env python
"""
Quick test to verify author query returns full citation metadata.

Using Steven Ding's author ID (smaller citation count, won't trigger individual queries).
"""

import os
from pathlib import Path
from semantic_scholar_utils import build_citation_network

# Test with Steven Ding's author ID
# (Should have citations but less than 10k to test author query optimization)
TEST_AUTHOR_ID = "9414926"  # Steven Ding

# Use API key if available
api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

print("Testing author query with full citation metadata...")
print(f"Author ID: {TEST_AUTHOR_ID}")
print(f"API Key: {'✓ Present' if api_key else '✗ Not set'}")
print()

# Use a fresh cache to ensure we're actually hitting the API
test_cache = Path("test_cache.db")
if test_cache.exists():
    test_cache.unlink()

# Build the network
result = build_citation_network(
    author_id=TEST_AUTHOR_ID,
    api_key=api_key,
    cache_db=test_cache,
    verbose=True,
)

print("\n" + "=" * 60)
print("RESULTS:")
print(f"  Author papers: {result['total_author_papers']}")
print(f"  Citing papers: {result['total_citing_papers']}")
print()

# Check if citing papers have full metadata
sample_size = min(5, len(result["citing_papers"]))
if sample_size > 0:
    print(f"Sample of {sample_size} citing papers:")
    for i, paper in enumerate(result["citing_papers"][:sample_size], 1):
        print(f"\n  Paper {i}:")
        print(f"    ID: {paper.get('paperId')}")
        print(f"    Title: {paper.get('title', 'MISSING')[:60]}...")
        print(f"    Abstract: {'✓ Present' if paper.get('abstract') else '✗ MISSING'}")
        print(f"    Year: {paper.get('year', 'MISSING')}")
        print(f"    Authors: {len(paper.get('authors', []))} authors")

        # Check if this came from author query or individual query
        # (If all have full metadata and no Step 4 message, author query worked!)

print("\n" + "=" * 60)
print("✓ Test complete!")

# Cleanup
if test_cache.exists():
    test_cache.unlink()
