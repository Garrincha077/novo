from __future__ import annotations

from pathlib import Path

import requests

_ORIGINAL_GET = requests.get
_CACHE = Path('output/fred_hy_oas.csv')


def _cached_get(url, *args, **kwargs):
    if 'fred.stlouisfed.org/graph/fredgraph.csv' in str(url) and _CACHE.exists():
        response = requests.Response()
        response.status_code = 200
        response.url = str(url)
        response._content = _CACHE.read_bytes()
        response.headers['Content-Type'] = 'text/csv'
        response.encoding = 'utf-8'
        return response
    return _ORIGINAL_GET(url, *args, **kwargs)


requests.get = _cached_get
