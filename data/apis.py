import requests
import time
import itertools
from requests.exceptions import HTTPError

# ─── Configuration ─────────────────────────────────────────────────────────────
API_KEYS = [
    "RGAPI-152ff7c9-c893-44ac-bbef-a1d4ff50f36f",
    "RGAPI-f856a26d-7c62-4004-ac78-79ca81832e7d",
    "RGAPI-fcd10ccc-f501-4bb9-afc9-93547e052d5e",
]
key_cycle = itertools.cycle(API_KEYS)

PLATFORM   = "na1"
MATCH_REG  = "americas"
QUEUE      = "RANKED_SOLO_5x5"

SLEEP      = 1.2 / len(API_KEYS)

BACKOFF    = 5
MAX_RETRIES = 3  
# ────────────────────────────────────────────────────────────────────────────────


def _get_headers():
    """Rotate through API keys to spread rate‐limit load."""
    return {"X-Riot-Token": next(key_cycle)}


def _get_json(url, params=None):
    """
    GET + JSON parse with:
      - single retry on 429 and 504 errors
      - raise a RuntimeError on any other HTTP or JSON error
    """
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=_get_headers(), params=params)
        try:
            resp.raise_for_status()
            return resp.json()
        except HTTPError as e:
            if resp.status_code == 429 and attempt == 0:
                # rate-limited: back off once, then retry with a fresh key
                time.sleep(BACKOFF)
                continue
            elif resp.status_code == 504 and attempt < MAX_RETRIES - 1:
                # gateway timeout: back off and retry
                time.sleep(BACKOFF)
                continue
            # any other HTTP error, or second 429/504, bubble up
            raise RuntimeError(
                f"[{resp.status_code}] {e} fetching {url} (params={params})"
            ) from e
        except ValueError as e:
            # JSON decoding error
            raise RuntimeError(f"Invalid JSON from {url}: {e}") from e
    # should never reach here
    raise RuntimeError(f"Failed to GET {url} after {MAX_RETRIES} attempts")


def get_dplus_entries(league: str):
    """
    Returns the list of Challenger/Master/Grandmaster entries
    for the given league (e.g. "grandmaster").
    """
    url = (
        f"https://{PLATFORM}.api.riotgames.com"
        f"/lol/league/v4/{league}leagues/by-queue/{QUEUE}"
    )
    data = _get_json(url)
    entries = data.get("entries")
    if entries is None:
        raise RuntimeError(f"No 'entries' key in response from {url!r}")
    time.sleep(SLEEP)
    return entries


def get_tier_entries(tier: str, division: str):
    """
    Returns all entries for a given tier/division (e.g. "DIAMOND", "I").
    """
    url = (
        f"https://{PLATFORM}.api.riotgames.com"
        f"/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    )
    data = _get_json(url)
    time.sleep(SLEEP)
    return data


def get_match_ids(puuid: str):
    """
    Returns up to 200 match IDs for this PUUID, in pages of 100.
    """
    url = (
        f"https://{MATCH_REG}.api.riotgames.com"
        f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
    )
    match_ids = []
    for start in range(0, 200, 100):
        params = {"count": 100, "queue": 420, "start": start}
        page = _get_json(url, params=params)
        if not isinstance(page, list):
            raise RuntimeError(f"Expected list of IDs, got {page!r}")
        match_ids.extend(page)
        time.sleep(SLEEP)
    return match_ids


def get_match_info(match_id: str):
    """
    Full metadata for a given match.
    """
    url = (
        f"https://{MATCH_REG}.api.riotgames.com"
        f"/lol/match/v5/matches/{match_id}"
    )
    data = _get_json(url)
    time.sleep(SLEEP)
    return data


def get_match_timeline(match_id: str):
    """
    Timeline events for a given match.
    """
    url = (
        f"https://{MATCH_REG}.api.riotgames.com"
        f"/lol/match/v5/matches/{match_id}/timeline"
    )
    data = _get_json(url)
    time.sleep(SLEEP)
    return data
