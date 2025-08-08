#!/usr/bin/env python3
import os
import sys
import time
import math
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

API_BASE_URL = "https://api.taostats.io"
ACCOUNT_HISTORY_ENDPOINT = "/api/account/history/v1"
STAKE_LATEST_ENDPOINT = "/api/dtao/stake_balance/latest/v1"
STAKE_HISTORY_ENDPOINT = "/api/dtao/stake_balance/history/v1"
SUBNET_LATEST_ENDPOINT = "/api/subnet/latest/v1"

DEFAULT_WEEKS = 52
MAX_NETUID = 128  # inclusive 0..128
PLANCK_PER_TAO = 1e9


def get_api_key() -> str:
    load_dotenv()
    # Support both API_KEY and taostats_api for convenience
    api_key = os.getenv("API_KEY") or os.getenv("taostats_api")
    if not api_key:
        print("ERROR: API_KEY (or taostats_api) not found in env/.env", file=sys.stderr)
        sys.exit(1)
    return api_key


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "accept": "application/json",
        "Authorization": api_key,
    }


def paged_get(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    limit = params.get("limit", 200)
    page = params.get("page", 1)
    backoff = 5
    while True:
        params["limit"] = limit
        params["page"] = page
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            time.sleep(retry_after)
            backoff = min(int(backoff * 1.5) + 1, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get("data", []))
        next_page = data.get("pagination", {}).get("next_page")
        if not next_page:
            break
        page = next_page
    return items


def fetch_account_history(address: str, start_ts: int, end_ts: int, api_key: str) -> pd.DataFrame:
    headers = build_headers(api_key)
    url = f"{API_BASE_URL}{ACCOUNT_HISTORY_ENDPOINT}"
    params: Dict[str, Any] = {
        "address": address,
        "timestamp_start": start_ts,
        "timestamp_end": end_ts,
        "order": "timestamp_asc",
        "limit": 200,
        "page": 1,
    }
    items = paged_get(url, headers, params)
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    # Normalize numeric columns to TAO
    for col in ["balance_total", "balance_staked", "balance_free"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / PLANCK_PER_TAO
    # Normalize timestamp to naive datetime
    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_subnet_latest(api_key: str) -> Dict[int, datetime]:
    """Return mapping netuid -> registration datetime (naive)."""
    headers = build_headers(api_key)
    url = f"{API_BASE_URL}{SUBNET_LATEST_ENDPOINT}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    reg_map: Dict[int, datetime] = {}
    for sn in data:
        netuid = int(sn.get("netuid"))
        ts = sn.get("registration_timestamp") or sn.get("timestamp")
        if ts:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.notna(dt):
                reg_map[netuid] = dt.tz_localize(None)
    return reg_map


def fetch_latest_stake_pairs(coldkey: str, api_key: str) -> List[Tuple[int, str]]:
    """Return list of (netuid, hotkey_ss58) that currently have stake for this coldkey."""
    headers = build_headers(api_key)
    url = f"{API_BASE_URL}{STAKE_LATEST_ENDPOINT}"
    params = {"coldkey": coldkey, "limit": 200, "page": 1}
    items = paged_get(url, headers, params)
    pairs: List[Tuple[int, str]] = []
    for it in items:
        try:
            netuid = int(it.get("netuid"))
            hk = it.get("hotkey", {}).get("ss58")
            if hk is None:
                continue
            if 0 <= netuid <= MAX_NETUID:
                pairs.append((netuid, hk))
        except Exception:
            continue
    # Deduplicate
    pairs = sorted(list({(n, h) for (n, h) in pairs}))
    return pairs


def fetch_stake_history(coldkey: str, hotkey: str, netuid: int, api_key: str) -> pd.DataFrame:
    headers = build_headers(api_key)
    url = f"{API_BASE_URL}{STAKE_HISTORY_ENDPOINT}"
    params: Dict[str, Any] = {
        "coldkey": coldkey,
        "hotkey": hotkey,
        "netuid": netuid,
        "limit": 200,
        "page": 1,
        # history API returns newest->oldest typically; we will sort later
    }
    items = paged_get(url, headers, params)
    if not items:
        return pd.DataFrame(columns=["date", "balance_as_tao"])  # empty
    df = pd.DataFrame(items)
    # Normalize numeric
    if "balance_as_tao" in df.columns:
        df["balance_as_tao"] = pd.to_numeric(df["balance_as_tao"], errors="coerce") / PLANCK_PER_TAO
    elif "balance" in df.columns:
        # fall back to alpha (unlikely desired); still convert to TAO-like scale per examples
        df["balance_as_tao"] = pd.to_numeric(df["balance"], errors="coerce") / PLANCK_PER_TAO
    else:
        df["balance_as_tao"] = math.nan
    # Timestamp
    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
    else:
        df["date"] = pd.NaT
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "balance_as_tao"]]


def weekly_index(start_dt: datetime, end_dt: datetime) -> pd.DatetimeIndex:
    # Normalize to dates (no time)
    start = pd.to_datetime(start_dt.date())
    end = pd.to_datetime(end_dt.date())
    # Use Mondays for week labels
    idx = pd.date_range(start=start, end=end, freq="W-MON")
    if len(idx) == 0 or idx[0] > start:
        # Ensure we include the first week before start
        first = (start - pd.offsets.Week(weekday=0))  # Monday on/preceding start
        idx = pd.DatetimeIndex([first]).append(idx)
    return idx


def resample_weekly(df: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex) -> pd.Series:
    if df.empty:
        return pd.Series(index=idx, dtype=float)
    temp = df.set_index("date")[[value_col]].sort_index()
    # Forward-fill to daily then take value at weekly index
    # Efficiently reindex with ffill directly on idx
    series = temp[value_col].reindex(pd.DatetimeIndex(sorted(set(temp.index) | set(idx))))
    series = series.ffill()
    result = series.reindex(idx)
    return result


def compute_weekly_free_balance(address: str, start_dt: datetime, end_dt: datetime, api_key: str, idx: pd.DatetimeIndex) -> pd.Series:
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
    df = fetch_account_history(address, start_ts, end_ts, api_key)
    if df.empty:
        return pd.Series(index=idx, dtype=float)
    df = df[["date", "balance_free"]]
    return resample_weekly(df, "balance_free", idx)


def compute_weekly_subnet_stakes(coldkey: str, start_dt: datetime, end_dt: datetime, api_key: str, idx: pd.DatetimeIndex, subnet_registration: Dict[int, datetime]) -> pd.DataFrame:
    pairs = fetch_latest_stake_pairs(coldkey, api_key)
    # Prepare result frame with columns subnet_0..subnet_128
    cols = [f"subnet_{i}" for i in range(0, MAX_NETUID + 1)]
    result = pd.DataFrame(index=idx, columns=cols, dtype=float)
    result.loc[:, :] = 0.0

    # Aggregate per netuid across hotkeys
    by_netuid: Dict[int, List[pd.Series]] = {}
    for netuid, hotkey in pairs:
        if not (0 <= netuid <= MAX_NETUID):
            continue
        hist = fetch_stake_history(coldkey, hotkey, netuid, api_key)
        series = resample_weekly(hist, "balance_as_tao", idx)
        by_netuid.setdefault(netuid, []).append(series)

    for netuid in range(0, MAX_NETUID + 1):
        weekly_series = None
        if netuid in by_netuid:
            weekly_series = sum(by_netuid[netuid])  # sum across hotkeys
        else:
            weekly_series = pd.Series(0.0, index=idx)
        # Apply NaN before subnet registration
        reg_dt = subnet_registration.get(netuid)
        if reg_dt is not None:
            weekly_series = weekly_series.copy()
            weekly_series[weekly_series.index < pd.to_datetime(reg_dt.date())] = math.nan
        else:
            # If no registration known, keep as 0 except before global min? leave as-is
            pass
        result[f"subnet_{netuid}"] = weekly_series

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate weekly CSV for the last year: free balance and subnet stakes (0..128) for a coldkey.\n"
            "If a subnet did not exist on a date, value is NaN; otherwise 0 when existed but no stake.\n"
            "Reads API key from API_KEY or taostats_api environment/.env."
        )
    )
    parser.add_argument("--address", required=True, help="Coldkey ss58 address")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Number of weeks back (default 52)")
    parser.add_argument(
        "--out", type=str, default="weekly_subnet_balances.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    api_key = get_api_key()

    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(weeks=args.weeks)

    idx = weekly_index(start_dt, end_dt)

    # Fetch subnet registration times for NaN masking
    subnet_registration = fetch_subnet_latest(api_key)

    # Compute free balance
    free_series = compute_weekly_free_balance(args.address, start_dt, end_dt, api_key, idx)

    # Compute subnet stakes (aggregated across hotkeys per subnet)
    subnet_df = compute_weekly_subnet_stakes(args.address, start_dt, end_dt, api_key, idx, subnet_registration)

    # Build final frame
    out_df = pd.DataFrame(index=idx)
    out_df["date"] = out_df.index
    out_df["free_balance_tao"] = free_series
    for col in subnet_df.columns:
        out_df[col] = subnet_df[col]

    out_df.reset_index(drop=True, inplace=True)
    out_path = os.path.abspath(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main() 