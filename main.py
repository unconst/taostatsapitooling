import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


API_BASE_URL = "https://api.taostats.io"
ACCOUNT_HISTORY_ENDPOINT = "/api/account/history/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a coldkey's balance history from Taostats and save a plot and CSV.\n"
            "Reads API_KEY from .env or environment."
        )
    )
    parser.add_argument(
        "--address",
        required=True,
        help="SS58 coldkey address to query (e.g., 5HBt...)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start time as ISO date (YYYY-MM-DD) or epoch seconds. Default: 365 days ago"
        ),
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "End time as ISO date (YYYY-MM-DD) or epoch seconds. Default: now"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="balance_history",
        help="Filename prefix for outputs (PNG and CSV).",
    )
    return parser.parse_args()


def to_epoch_seconds(value: Optional[str], fallback: datetime) -> int:
    if value is None:
        return int(fallback.replace(tzinfo=timezone.utc).timestamp())
    # Allow raw epoch seconds
    try:
        if value.isdigit():
            return int(value)
    except AttributeError:
        pass
    # Parse ISO-like date/time
    try:
        # If only date is provided, interpret as midnight UTC
        if len(value) <= 10:
            dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            # Attempt common formats; let pandas handle broader cases if needed
            dt = pd.to_datetime(value, utc=True).to_pydatetime()
        return int(dt.timestamp())
    except Exception:
        return int(fallback.replace(tzinfo=timezone.utc).timestamp())


def get_api_key() -> str:
    load_dotenv()  # Loads from .env if present
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("ERROR: API_KEY not found in environment or .env", file=sys.stderr)
        sys.exit(1)
    return api_key


def paged_get(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    # Ensure paging parameters exist
    if "limit" not in params:
        params["limit"] = 200
    if "page" not in params:
        params["page"] = 1

    while params["page"] is not None:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "30"))
            print(f"Rate limited (429). Sleeping {retry_after}s ...")
            time.sleep(retry_after)
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(f"HTTP error: {e} | body: {response.text}", file=sys.stderr)
            sys.exit(2)
        data = response.json()
        items.extend(data.get("data", []))
        params["page"] = data.get("pagination", {}).get("next_page")
    return items


def fetch_account_history(address: str, start_ts: int, end_ts: int, api_key: str) -> pd.DataFrame:
    headers = {
        "accept": "application/json",
        "Authorization": api_key,
    }
    params: Dict[str, Any] = {
        "address": address,
        "timestamp_start": start_ts,
        "timestamp_end": end_ts,
        "order": "timestamp_asc",
        "limit": 200,
        "page": 1,
    }
    url = f"{API_BASE_URL}{ACCOUNT_HISTORY_ENDPOINT}"
    items = paged_get(url, headers=headers, params=params)
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    return df


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    # Convert amounts from planck-like units to TAO
    for col in ["balance_total", "balance_staked", "balance_free"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1e9
    # Parse timestamp flexibly
    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df["date"] = pd.NaT
    df = df.sort_values("date").reset_index(drop=True)
    # Drop rows with invalid dates to prevent matplotlib errors
    df = df[~df["date"].isna()].copy()
    # Remove timezone info (matplotlib prefers naive datetimes)
    try:
        if getattr(df["date"].dtype, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        # If anything goes wrong, leave as is; plotting may still work if no NaT present
        pass
    return df


def save_outputs(df: pd.DataFrame, out_prefix: str, address: str) -> None:
    csv_path = f"{out_prefix}.csv"
    png_path = f"{out_prefix}.png"

    # Save CSV
    cols = [c for c in ["date", "balance_free", "balance_staked", "balance_total", "timestamp"] if c in df.columns]
    df_to_save = df[cols].copy()
    df_to_save.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(11, 6))
    if "balance_total" in df:
        plt.plot(df["date"], df["balance_total"], label="Total TAO", linewidth=2)
    if "balance_free" in df:
        plt.plot(df["date"], df["balance_free"], label="Free TAO", alpha=0.8)
    if "balance_staked" in df:
        plt.plot(df["date"], df["balance_staked"], label="Staked TAO", alpha=0.8)

    plt.title(f"Balance over time\n{address}")
    plt.xlabel("Date")
    plt.ylabel("TAO")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    now = datetime.now(timezone.utc)
    default_start = now - timedelta(days=365)

    start_ts = to_epoch_seconds(args.start, default_start)
    end_ts = to_epoch_seconds(args.end, now)

    print(
        f"Querying history for address={args.address} start={start_ts} end={end_ts} ..."
    )
    df_raw = fetch_account_history(args.address, start_ts, end_ts, api_key)
    if df_raw.empty:
        print("No data returned for the specified range and address.")
        sys.exit(0)

    df = normalize_history_df(df_raw)
    save_outputs(df, args.out_prefix, args.address)


if __name__ == "__main__":
    main() 