import os
import requests

# ─────────────────────────────────────────────────────────────────────
#  Real-time AQI lookup with WORLDWIDE coverage and STABLE values that
#  match what most "official" AQI dashboards report.
#
#  Why this design:
#    A single WAQI monitoring station can spike (broken sensor, fire,
#    construction site next door) and report e.g. AQI 232 for a city
#    whose actual air is around 60. Most consumer AQI apps (and the
#    official aggregators) instead show a CITY-LEVEL number.
#
#  Strategy (in order):
#    1) Open-Meteo Air Quality (PRIMARY)
#         • Free, no API key, works for ANY city worldwide.
#         • Returns modelled US-EPA AQI from CAMS satellite + reanalysis,
#           anchored to the city's geocoded centre — no single-station
#           outliers.
#    2) WAQI multi-station average (fallback)
#         • If Open-Meteo is unreachable, search WAQI for the city and
#           AVERAGE every station that returned a numeric reading.
#    3) WAQI single-feed (last resort)
#         • Original behaviour — single nearest station.
# ─────────────────────────────────────────────────────────────────────

WAQI_TOKEN = os.environ.get(
    "WAQI_API_TOKEN",
    "e3cb15cbac69deec6b4955987bec64620a6f7ce0",
)

WAQI_FEED_URL   = "https://api.waqi.info/feed/{city}/?token={token}"
WAQI_SEARCH_URL = "https://api.waqi.info/search/"
GEOCODE_URL     = "https://geocoding-api.open-meteo.com/v1/search"
AIRQUALITY_URL  = "https://air-quality-api.open-meteo.com/v1/air-quality"

_TIMEOUT = 8


# ───────────────────────── helpers ──────────────────────────────────

def _geocode(city: str):
    """City name → (lat, lon, pretty_name) via Open-Meteo geocoding."""
    try:
        r = requests.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=_TIMEOUT,
        )
        results = (r.json() or {}).get("results") or []
        if not results:
            return None
        hit = results[0]
        lat = hit.get("latitude")
        lon = hit.get("longitude")
        if lat is None or lon is None:
            return None
        parts = [hit.get("name"), hit.get("admin1"), hit.get("country")]
        pretty = ", ".join(dict.fromkeys([p for p in parts if p]))
        return float(lat), float(lon), pretty
    except Exception:
        return None


def _open_meteo_aqi(lat: float, lon: float):
    """Modelled US-EPA AQI for a coordinate. Returns int or None."""
    try:
        r = requests.get(
            AIRQUALITY_URL,
            params={
                "latitude":  lat,
                "longitude": lon,
                "current":   "us_aqi",
                "timezone":  "auto",
            },
            timeout=_TIMEOUT,
        )
        cur = (r.json() or {}).get("current") or {}
        aqi = cur.get("us_aqi")
        if isinstance(aqi, (int, float)):
            return int(round(aqi))
    except Exception:
        pass
    return None


def _waqi_search_avg(city: str):
    """Average every numeric WAQI station that matches the keyword.
    Returns (aqi, pretty_name) or (None, None)."""
    try:
        r = requests.get(
            WAQI_SEARCH_URL,
            params={"keyword": city, "token": WAQI_TOKEN},
            timeout=_TIMEOUT,
        )
        body = r.json() or {}
        if body.get("status") != "ok":
            return None, None
        # Keep only stations whose station name contains the requested city
        wanted = city.lower().strip()
        readings = []
        sample_name = None
        for s in body.get("data", []):
            name = (s.get("station") or {}).get("name", "")
            if wanted not in name.lower():
                continue
            try:
                v = float(s.get("aqi"))
            except (TypeError, ValueError):
                continue
            readings.append(v)
            sample_name = sample_name or name
        if not readings:
            return None, None
        # Trim a single extreme outlier (broken sensor) when we have ≥4 readings
        if len(readings) >= 4:
            readings.sort()
            readings = readings[:-1]   # drop highest
        avg = sum(readings) / len(readings)
        return int(round(avg)), sample_name or city.title()
    except Exception:
        return None, None


def _waqi_single(city: str):
    """Original single-station feed. Returns (aqi, pretty_name) or (None, None)."""
    try:
        r = requests.get(
            WAQI_FEED_URL.format(city=city, token=WAQI_TOKEN),
            timeout=_TIMEOUT,
        )
        body = r.json() or {}
        if body.get("status") != "ok":
            return None, None
        aqi  = body["data"].get("aqi")
        name = body["data"].get("city", {}).get("name", city.title())
        if isinstance(aqi, (int, float)):
            return int(aqi), name
    except Exception:
        pass
    return None, None


# ───────────────────────── public API ───────────────────────────────

def get_real_aqi(city: str = "delhi"):
    """
    Return (aqi, display_name) for the given city, or (None, None) if not
    resolvable. Works for any city worldwide and avoids single-station
    outliers, so the values line up with what official AQI dashboards show.
    """
    if not city or not str(city).strip():
        return None, None

    city = str(city).strip()

    # 1) PRIMARY — Open-Meteo (city-centre modelled AQI, globally consistent)
    geo = _geocode(city)
    if geo:
        lat, lon, pretty = geo
        aqi = _open_meteo_aqi(lat, lon)
        if aqi is not None:
            return aqi, pretty

    # 2) FALLBACK — WAQI multi-station average (filters single-sensor spikes)
    aqi, name = _waqi_search_avg(city)
    if aqi is not None:
        return aqi, name

    # 3) LAST RESORT — WAQI single station
    aqi, name = _waqi_single(city)
    if aqi is not None:
        return aqi, name

    return None, None
