import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BASE = "https://rxnav.nlm.nih.gov/REST"

_failure_log_path: Optional[Path] = None


def set_failure_log(path) -> None:
    """Configure a file to append failed RxNorm lookups (TSV: timestamp, name)."""
    global _failure_log_path
    _failure_log_path = Path(path)


def _record_failure(name: str) -> None:
    if _failure_log_path is None:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _failure_log_path.open("a", encoding="utf-8") as f:
        f.write(f"{ts}\t{name}\n")

# 20 req/sec limit
class _RateLimiter:
    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        gap = self._interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


_limiter = _RateLimiter(rate=20)


def _get(url: str, params: dict) -> dict:
    retries = 3
    for attempt in range(retries):
        _limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning("RxNorm request failed (%s), retrying in %ds", e, wait)
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            # 429 or 5xx should retry, rest of 4xx client errors are bad requests
            if e.response.status_code in (429, 500, 502, 503, 504):
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("RxNorm HTTP %d, retrying in %ds", e.response.status_code, wait)
                time.sleep(wait)
            else:
                raise


def find_rxcui_exact(name: str) -> Optional[str]:
    """Return RXCUI for an exact ingredient name match, or None."""
    data = _get(f"{BASE}/rxcui.json", {"name": name, "allsrc": "0", "search": "1"})
    rxcui = data.get("idGroup", {}).get("rxnormId", [])
    if rxcui:
        return rxcui[0]
    return None


def find_rxcui_approx(name: str) -> Optional[str]:
    """Return RXCUI via approximate match when exact lookup fails.

    Only accepts hits with score == 100 to avoid wrong-drug false positives.
    """
    data = _get(
        f"{BASE}/approximateTerm.json",
        {"term": name, "maxEntries": "5", "option": "0"},
    )
    candidates = data.get("approximateGroup", {}).get("candidate", [])
    for c in candidates:
        if c.get("score") == "100":
            return c["rxcui"]
    return None


def resolve_rxcui(name: str) -> Optional[str]:
    """Resolve ingredient name to RXCUI: exact first, approximate fallback."""
    rxcui = find_rxcui_exact(name)
    if rxcui:
        log.debug("Exact RxNorm match: %s -> %s", name, rxcui)
        return rxcui

    rxcui = find_rxcui_approx(name)
    if rxcui:
        log.debug("Approx RxNorm match: %s -> %s", name, rxcui)
        return rxcui

    log.warning("RxNorm lookup failed: %s", name)
    _record_failure(name)
    return None
