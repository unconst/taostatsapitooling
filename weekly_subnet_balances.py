#!/usr/bin/env python3
import os
import sys
import time
import math
import logging
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

logger = logging.getLogger("weekly_subnet_balances")
SESSION = requests.Session()
REQUEST_TIMEOUT = (10, 60)  # (connect, read) seconds


def get_api_key() -> str:
    load_dotenv()
    # Support both API_KEY and taostats_api for convenience
    api_key = os.getenv("API_KEY") or os.getenv("taostats_api")
    if not api_key:
        logger.error("API_KEY (or taostats_api) not found in env/.env")
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
    logger.debug(f"GET {url} params={params}")
    while True:
        params["limit"] = limit
        params["page"] = page
        t1 = time.perf_counter()
        resp = SESSION.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        elapsed = time.perf_counter() - t1
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            logger.warning(f"429 on {url} page={page}. Sleeping {retry_after}s (elapsed {elapsed:.2f}s)")
            time.sleep(retry_after)
            backoff = min(int(backoff * 1.5) + 1, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        page_items = data.get("data", [])
        items.extend(page_items)
        logger.debug(
            f"Fetched {url} page={page} items={len(page_items)} total={len(items)} in {elapsed:.2f}s"
        )
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
    logger.info("Fetching account history (free/staked/total)...")
    items = paged_get(url, headers, params)
    logger.info(f"Account history items: {len(items)}")
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
    logger.info("Fetching subnet registration metadata...")
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
    logger.info(f"Known subnet registrations: {len(reg_map)}")
    return reg_map


def fetch_latest_stake_pairs(coldkey: str, api_key: str) -> List[Tuple[int, str]]:
    """Return list of (netuid, hotkey_ss58) that currently have stake for this coldkey."""
    headers = build_headers(api_key)
    url = f"{API_BASE_URL}{STAKE_LATEST_ENDPOINT}"
    params = {"coldkey": coldkey, "limit": 200, "page": 1}
    logger.info("Discovering current (netuid, hotkey) stake pairs...")
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
    pairs = sorted(list({(n, h) for (n, h) in pairs}))
    unique_nets = sorted({n for n, _ in pairs})
    logger.info(f"Found {len(pairs)} pairs across {len(unique_nets)} subnets: {unique_nets}")
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
    }
    # Custom pagination here to expose per-page logs for performance visibility
    items: List[Dict[str, Any]] = []
    total_pages_seen = 0
    backoff = 5
    while True:
        t1 = time.perf_counter()
        resp = SESSION.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        elapsed = time.perf_counter() - t1
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            logger.warning(
                f"429 stake_history netuid={netuid} hk={hotkey} page={params['page']}. Sleep {retry_after}s (elapsed {elapsed:.2f}s)"
            )
            time.sleep(retry_after)
            backoff = min(int(backoff * 1.5) + 1, 60)
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            logger.error(
                f"HTTP error stake_history netuid={netuid} hk={hotkey} page={params['page']}: {e} body={resp.text[:200]}"
            )
            raise
        data = resp.json()
        page_items = data.get("data", [])
        items.extend(page_items)
        total_pages_seen += 1
        next_page = data.get("pagination", {}).get("next_page")
        logger.info(
            f"stake_history netuid={netuid} hk={hotkey}: page={params['page']} items={len(page_items)} total_items={len(items)} next={next_page} in {elapsed:.2f}s"
        )
        if not next_page:
            break
        params["page"] = next_page
    logger.info(
        f"stake_history netuid={netuid} hk={hotkey}: completed pages={total_pages_seen} total_items={len(items)}"
    )

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
    logger.info(f"Weekly index points: {len(idx)} (from {idx.min().date()} to {idx.max().date()})")
    return idx


def resample_weekly(df: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex) -> pd.Series:
    if df.empty:
        return pd.Series(index=idx, dtype=float)
    temp = df.set_index("date")[[value_col]].sort_index()
    # Drop duplicate timestamps, keep the last observation
    temp = temp[~temp.index.duplicated(keep="last")]
    # Build a combined index and forward-fill values, then pick weekly points
    full_index = temp.index.union(idx).sort_values()
    series = temp[value_col].reindex(full_index).ffill()
    result = series.reindex(idx)
    logger.debug(f"Resampled {len(temp)} points to {len(result)} weekly points")
    return result


def compute_weekly_free_balance(address: str, start_dt: datetime, end_dt: datetime, api_key: str, idx: pd.DatetimeIndex) -> pd.Series:
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
    df = fetch_account_history(address, start_ts, end_ts, api_key)
    logger.info(f"Account history rows: {len(df)}")
    if df.empty:
        return pd.Series(index=idx, dtype=float)
    df = df[["date", "balance_free"]]
    return resample_weekly(df, "balance_free", idx)


def compute_weekly_subnet_stakes(coldkey: str, start_dt: datetime, end_dt: datetime, api_key: str, idx: pd.DatetimeIndex, subnet_registration: Dict[int, datetime], progress_every: int = 10) -> pd.DataFrame:
    pairs = fetch_latest_stake_pairs(coldkey, api_key)
    total_pairs = len(pairs)
    logger.info("Fetching stake histories per (netuid, hotkey) pair...")
    logger.info(f"Total pairs: {total_pairs}")
    t_start = time.perf_counter()
    # Prepare result frame with columns subnet_0..subnet_128
    cols = [f"subnet_{i}" for i in range(0, MAX_NETUID + 1)]
    result = pd.DataFrame(index=idx, columns=cols, dtype=float)
    result.loc[:, :] = 0.0

    # Aggregate per netuid across hotkeys
    by_netuid: Dict[int, List[pd.Series]] = {}
    for i, (netuid, hotkey) in enumerate(pairs, start=1):
        if i == 1 or (progress_every and i % progress_every == 0) or logger.isEnabledFor(logging.DEBUG):
            elapsed = time.perf_counter() - t_start
            avg = elapsed / max(1, i)
            remaining = total_pairs - i
            eta_sec = remaining * avg
            logger.info(
                f"Pair {i}/{total_pairs}: netuid={netuid} hotkey={hotkey} | elapsed={elapsed:.1f}s avg/pair={avg:.2f}s ETA={eta_sec/60:.1f}m"
            )
        p_start = time.perf_counter()
        try:
            hist = fetch_stake_history(coldkey, hotkey, netuid, api_key)
            series = resample_weekly(hist, "balance_as_tao", idx)
            by_netuid.setdefault(netuid, []).append(series)
        except Exception as e:
            logger.warning(f"Failed fetching/resampling for netuid={netuid} hotkey={hotkey}: {e}")
            continue
        finally:
            p_dur = time.perf_counter() - p_start
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Pair {i}/{total_pairs} processed in {p_dur:.2f}s (rows={0 if 'hist' not in locals() else len(hist)})"
                )

    t_fetch = time.perf_counter() - t_start
    logger.info(f"Fetched histories for {sum(len(v) for v in by_netuid.values())} series across {len(by_netuid)} subnets in {t_fetch:.1f}s")

    logger.info("Aggregating weekly stakes per subnet (0..128)...")
    agg_start = time.perf_counter()
    for netuid in range(0, MAX_NETUID + 1):
        if progress_every and netuid % max(1, progress_every) == 0:
            logger.info(f"Aggregating subnet {netuid}/{MAX_NETUID}")
        if netuid in by_netuid:
            weekly_series = sum(by_netuid[netuid])  # sum across hotkeys
        else:
            weekly_series = pd.Series(0.0, index=idx)
        # Apply NaN before subnet registration
        reg_dt = subnet_registration.get(netuid)
        if reg_dt is not None:
            weekly_series = weekly_series.copy()
            weekly_series[weekly_series.index < pd.to_datetime(reg_dt.date())] = math.nan
        result[f"subnet_{netuid}"] = weekly_series
    agg_dur = time.perf_counter() - agg_start
    logger.info(f"Aggregation complete in {agg_dur:.1f}s")

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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--progress-every", type=int, default=10, help="Log progress every N pairs/subnets (default 10)")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = get_api_key()

    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(weeks=args.weeks)

    logger.info(f"Starting run for address={args.address}")
    logger.info(f"Range: {start_dt.date()} to {end_dt.date()} (~{args.weeks} weeks)")

    idx = weekly_index(start_dt, end_dt)

    # Fetch subnet registration times for NaN masking
    subnet_registration = fetch_subnet_latest(api_key)

    # Compute free balance
    free_series = compute_weekly_free_balance(args.address, start_dt, end_dt, api_key, idx)
    logger.info("Computing subnet stakes...")

    # Compute subnet stakes (aggregated across hotkeys per subnet)
    subnet_df = compute_weekly_subnet_stakes(
        args.address,
        start_dt,
        end_dt,
        api_key,
        idx,
        subnet_registration,
        progress_every=max(1, args.progress_every),
    )

    # Build final frame
    out_df = pd.DataFrame(index=idx)
    out_df["date"] = out_df.index
    out_df["free_balance_tao"] = free_series
    for col in subnet_df.columns:
        out_df[col] = subnet_df[col]

    out_df.reset_index(drop=True, inplace=True)
    out_path = os.path.abspath(args.out)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main() 