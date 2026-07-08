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
    """Configure a file to append failed RxNorm lookups (TSV: timestamp, name).

    Pass None to disable logging.
    """
    global _failure_log_path
    _failure_log_path = Path(path) if path is not None else None


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


def find_rxcui_approx_candidates(name: str) -> list[dict]:
    """Return distinct real-drug candidates for a name that failed exact lookup.

    Uses RxNorm's spellingsuggestions endpoint (built for typo correction,
    unlike approximateTerm's score, which is an unbounded relevance number
    rather than a confidence percentage and isn't a reliable filter on its
    own). Each suggestion is resolved via exact lookup and deduped by rxcui;
    suggestions that don't resolve to a real rxcui are dropped. More than one
    distinct candidate means the name is a genuine ambiguous typo (it reads
    as close to two or more different drugs), not just a case to filter by
    score.
    """
    data = _get(f"{BASE}/spellingsuggestions.json", {"name": name})
    suggestion_group = data.get("suggestionGroup") or {}
    suggestion_list = suggestion_group.get("suggestionList") or {}
    suggestions = suggestion_list.get("suggestion") or []

    candidates = []
    seen_rxcuis = set()
    for suggestion in suggestions:
        rxcui = find_rxcui_exact(suggestion)
        if rxcui is None or rxcui in seen_rxcuis:
            continue
        seen_rxcuis.add(rxcui)
        candidates.append({"name": suggestion, "rxcui": rxcui})
    return candidates


def find_rxcui_approx(name: str) -> Optional[str]:
    """Return RXCUI via approximate match when exact lookup fails.

    Only resolves when the spelling suggestions point to exactly one real
    drug. An ambiguous typo (multiple plausible candidates, e.g. "metfromin"
    matching both "merbromin" and "metformin") returns None rather than
    guessing between them.
    """
    candidates = find_rxcui_approx_candidates(name)
    if len(candidates) == 1:
        return candidates[0]["rxcui"]
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
