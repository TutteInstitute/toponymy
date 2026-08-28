#!/usr/bin/env python
"""Quick diagnostic to check Semantic Scholar API rate limit status."""

import json
import os
import time
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
url = "https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b"

headers = {}
if api_key:
    headers["x-api-key"] = api_key
    print(f"✓ Using API key: {api_key[:10]}...")
else:
    print("✗ No API key found")

print(f"\nAttempting request to: {url}")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

try:
    response = requests.get(url, headers=headers, timeout=10)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Reason: {response.reason}")
    print(f"Elapsed Time: {response.elapsed.total_seconds():.3f} seconds")
    print(f"URL: {response.url}")
    
    print("\n" + "=" * 60)
    print("REQUEST HEADERS SENT:")
    print("=" * 60)
    for key, value in response.request.headers.items():
        # Partially mask API key for security
        if 'api-key' in key.lower() and value:
            print(f"  {key}: {value[:15]}...{value[-5:]}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("ALL RESPONSE HEADERS:")
    print("=" * 60)
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("RESPONSE BODY:")
    print("=" * 60)
    if response.status_code == 200:
        print("✓ SUCCESS - API is accessible!")
        try:
            data = response.json()
            print(json.dumps(data, indent=2)[:1000])  # First 1000 chars
        except:
            print(response.text[:1000])
    elif response.status_code == 429:
        print("✗ RATE LIMITED - Need to wait")
        print(f"Response text: {response.text}")
    else:
        print(f"? Unexpected status: {response.status_code}")
        print(response.text[:1000])
        
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
