"""
Generate citation network data for the citation_network_analysis notebook.

This standalone script demonstrates using semantic_scholar_utils.py to build
a citation network from Semantic Scholar. Run this script to:
- Test your API key setup
- Generate citation_network_data.json for use in the notebook
- See a minimal working example of the utility module

The script uses John Healy's citation data (one of Toponymy's authors) as a
convenient, moderately-sized dataset for demonstration. You can substitute
any Semantic Scholar author ID.

Requirements:
    This script requires Toponymy with development dependencies:
        pip install toponymy[dev]

Usage:
    python generate_citation_network.py

API Key (optional but recommended):
  Set the SEMANTIC_SCHOLAR_API_KEY environment variable:
    export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
  Get a free API key at: https://www.semanticscholar.org/product/api
  Without an API key, queries are much slower (10 sec between requests vs 3 sec).

Output (all files created in the same directory):
  - citation_network_data.json: Network data for analysis
  - semantic_scholar_cache.db: SQLite cache for subsequent runs
"""

import json
import os
from pathlib import Path
from semantic_scholar_utils import build_citation_network

# Configuration - all files in the same directory as this script
AUTHOR_ID = "2062756303"  # John Healy
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR / "citation_network_data.json"
CACHE_DB = SCRIPT_DIR / "semantic_scholar_cache.db"


def main():
    """Main execution."""
    # Load API key from environment variable
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        print(f"✓ Loaded API key from SEMANTIC_SCHOLAR_API_KEY environment variable")
    else:
        print(f"⚠ No API key found")
        print(f"  Set SEMANTIC_SCHOLAR_API_KEY environment variable:")
        print(f"    export SEMANTIC_SCHOLAR_API_KEY='your-key-here'")
        print(
            f"  Running without API key (much slower: 1 req/10 sec instead of 1 req/3 sec)"
        )
        print(f"  Get a free key at: https://www.semanticscholar.org/product/api")

    print(f"\nConfiguration:")
    print(f"  AUTHOR_ID: {AUTHOR_ID}")
    print(f"  Cache database: {CACHE_DB}")

    # Build citation network
    network_data = build_citation_network(
        author_id=AUTHOR_ID,
        api_key=api_key,
        cache_db=CACHE_DB,
        verbose=True,
    )

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(network_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Author ID:            {AUTHOR_ID}")
    print(f"Author papers:        {network_data['total_author_papers']}")
    print(f"Citing papers:        {network_data['total_citing_papers']}")
    print(f"\nOutput:")
    print(f"  Database:         {CACHE_DB}")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
