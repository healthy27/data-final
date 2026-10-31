"""Conservative DART collection: no year, period, or consolidation fallback."""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_final.data import validate_panel

BASE = "https://opendart.fss.or.kr/api/"
ACCOUNTS = {
    "revenue": ({"ifrs-full_Revenue", "ifrs_Revenue"}, {"매출액", "수익(매출액)"}, {"IS", "CIS"}),
    "operating_income": (
        {"dart_OperatingIncomeLoss"},
        {"영업이익", "영업이익(손실)"},
        {"IS", "CIS"},
    ),
    "net_income": (
        {"ifrs-full_ProfitLoss", "ifrs_ProfitLoss"},
        {"당기순이익", "당기순이익(손실)"},
        {"IS", "CIS"},
    ),
    "assets": ({"ifrs-full_Assets", "ifrs_Assets"}, {"자산총계"}, {"BS"}),
    "liabilities": ({"ifrs-full_Liabilities", "ifrs_Liabilities"}, {"부채총계"}, {"BS"}),
}


class DartError(RuntimeError):
    """Safe error: never contains authenticated request URLs."""


def amount(value) -> float:
    if value is None or str(value).strip() in {"", "-", "—"}:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    number = float(text)
    if not np.isfinite(number):
        raise ValueError("Nonfinite source amount")
    return number


def select_account(rows: list[dict], metric: str) -> tuple[float, dict]:
    ids, names, statements = ACCOUNTS[metric]
    eligible = [
        r for r in rows if r.get("sj_div") in statements and r.get("currency", "KRW") == "KRW"
    ]
    candidates = [r for r in eligible if r.get("account_id") in ids]
    method = "account_id"
    if not candidates:
        candidates = [r for r in eligible if re.sub(r"\s+", "", r.get("account_nm", "")) in names]
        method = "exact_name"
    try:
        values = {amount(r.get("thstrm_amount")) for r in candidates}
        values = {x for x in values if np.isfinite(x)}
    except (TypeError, ValueError):
        return np.nan, {"status": "invalid_amount", "metric": metric}
    if len(values) != 1:
        return np.nan, {"status": "missing" if not values else "ambiguous", "metric": metric}
    return values.pop(), {
        "status": "ok",
        "metric": metric,
        "method": method,
        "account_ids": sorted({r.get("account_id", "") for r in candidates}),
        "receipt_numbers": sorted({r.get("rcept_no", "") for r in candidates}),
    }


def employee_count(rows: list[dict]) -> tuple[float, str]:
    """Only sum unique business-unit × male/female detail rows; reject unclear totals."""
    if not rows:
        return np.nan, "missing"
    detail = [
        r
        for r in rows
        if r.get("sexdstn", "").strip() in {"남", "여", "남자", "여자"}
        and r.get("fo_bbm", "").strip() not in {"합계", "총계", "계"}
    ]
    if not detail:
        return np.nan, "unrecognized_employee_layout"
    keys = [
        (r.get("fo_bbm", "").strip(), r.get("sexdstn", "").strip().replace("자", ""))
        for r in detail
    ]
    if len(set(keys)) != len(keys):
        return np.nan, "duplicate_employee_groups"
    try:
        values = [amount(r.get("sm")) for r in detail]
    except (ValueError, TypeError):
        return np.nan, "invalid_employee_count"
    if any(not np.isfinite(x) or x < 0 or x != int(x) for x in values):
        return np.nan, "missing_or_invalid_employee_group"
    return float(sum(values)), "detail_sum_review_required"


class DartClient:
    def __init__(self, cache: Path, key: str | None = None):
        self.key = key or os.environ.get("DART_API_KEY")
        if not self.key:
            raise DartError("Set a newly issued DART_API_KEY environment variable first")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get(self, endpoint: str, params: dict) -> bytes:
        try:
            response = self.session.get(
                BASE + endpoint, params={**params, "crtfc_key": self.key}, timeout=(10, 30)
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            raise DartError(f"DART transport failure: {endpoint}; retry later") from None
        finally:
            time.sleep(0.2)

    def fetch(self, endpoint: str, params: dict) -> dict:
        # API key is excluded from cache names, metadata, outputs and errors.
        token = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{token}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            try:
                payload = json.loads(self._get(endpoint, params))
            except (ValueError, UnicodeDecodeError):
                raise DartError(f"Invalid DART JSON: {endpoint}") from None
            status = payload.get("status")
            if status not in {"000", "013"}:
                raise DartError(f"DART status {status}: {endpoint}")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "endpoint": endpoint,
                        "params": params,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                    indent=2,
                )
            )
        if payload.get("status") not in {"000", "013"}:
            raise DartError(f"Invalid cached DART status: {endpoint}")
        return payload

    def corporations(self) -> pd.DataFrame:
        path = self.cache / "corpCode.zip"
        if not path.exists():
            raw = self._get("corpCode.xml", {})
            try:
                with ZipFile(BytesIO(raw)) as archive:
                    archive.read("CORPCODE.xml")
            except Exception:
                raise DartError("DART corporation download is not a valid ZIP") from None
            path.write_bytes(raw)
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("CORPCODE.xml"))
        return pd.DataFrame(
            [{child.tag: child.text for child in row} for row in root.findall("list")]
        )


def collect(universe_path: Path, years: list[int], output: Path, cache: Path) -> dict:
    universe = pd.read_csv(universe_path, dtype=str)
    if "stock_code" not in universe or universe.stock_code.duplicated().any():
        raise ValueError("Universe requires unique stock_code values")
    if not universe.stock_code.str.fullmatch(r"\d{6}").all():
        raise ValueError("Stock codes must be six digits")
    client = DartClient(cache)
    master = client.corporations()
    rows, audit = [], []
    for stock in universe.stock_code:
        found = master[master.stock_code.eq(stock)]
        if len(found) != 1:
            raise ValueError(f"Ambiguous/missing corporation for stock_code={stock}")
        code, name = found.iloc[0][["corp_code", "corp_name"]]
        for year in sorted(set(years)):
            params = {"corp_code": code, "bsns_year": str(year), "reprt_code": "11011"}
            financial = client.fetch("fnlttSinglAcntAll.json", {**params, "fs_div": "OFS"})
            source = financial.get("list", [])
            if any(
                str(r.get("bsns_year")) != str(year)
                or r.get("corp_code") != code
                or r.get("reprt_code") != "11011"
                for r in source
            ):
                raise ValueError(f"DART identity/period mismatch: {code}, {year}")
            row = dict(
                corp_code=code,
                company=name,
                year=year,
                fs_div="OFS",
                reprt_code="11011",
                data_kind="dart",
            )
            for metric in ACCOUNTS:
                row[metric], detail = select_account(source, metric)
                audit.append({"corp_code": code, "year": year, **detail})
            emp = client.fetch("empSttus.json", params).get("list", [])
            if any(
                r.get("corp_code") != code
                or (r.get("stlm_dt") and str(r["stlm_dt"])[:4] != str(year))
                for r in emp
            ):
                raise ValueError("Employee corporation/period mismatch")
            row["employees"], status = employee_count(emp)
            audit.append(
                {
                    "corp_code": code,
                    "year": year,
                    "metric": "employees",
                    "status": status,
                    "receipt_numbers": sorted({r.get("rcept_no", "") for r in emp}),
                }
            )
            rows.append(row)
    if not rows:
        raise ValueError("No company-years requested")
    result = validate_panel(pd.DataFrame(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    output.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return {
        "rows": len(result),
        "output": str(output),
        "audit_records": len(audit),
        "warning": "Current snapshot; not point-in-time data. Review employee scope and mappings.",
    }
