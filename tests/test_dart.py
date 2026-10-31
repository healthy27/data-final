import numpy as np
import pytest
import requests

from data_final.dart import DartClient, DartError, amount, employee_count, select_account


def test_amount_formats_preserve_missing_and_negative():
    assert amount("1,234") == 1234
    assert amount("(1,234)") == -1234
    assert np.isnan(amount("-"))
    with pytest.raises(ValueError):
        amount("inf")


def test_ambiguous_financial_values_not_summed():
    row = dict(sj_div="IS", account_id="ifrs-full_ProfitLoss", thstrm_amount="100")
    value, audit = select_account([row, {**row, "thstrm_amount": "200"}], "net_income")
    assert np.isnan(value)
    assert audit["status"] == "ambiguous"
    value, _ = select_account([row, {**row, "sj_div": "CIS"}], "net_income")
    assert value == 100


def test_employee_totals_not_counted_twice_or_missing_as_zero():
    rows = [
        dict(fo_bbm="제조", sexdstn="남", sm="100"),
        dict(fo_bbm="제조", sexdstn="여", sm="50"),
        dict(fo_bbm="합계", sexdstn="합계", sm="150"),
    ]
    assert employee_count(rows)[0] == 150
    assert np.isnan(employee_count(rows + rows[:1])[0])
    rows[0]["sm"] = "-"
    assert np.isnan(employee_count(rows)[0])


def test_wrong_statement_or_currency_does_not_supply_target():
    rows = [
        dict(sj_div="CF", account_id="ifrs-full_ProfitLoss", thstrm_amount="100"),
        dict(sj_div="IS", currency="USD", account_id="ifrs-full_ProfitLoss", thstrm_amount="100"),
    ]
    assert np.isnan(select_account(rows, "net_income")[0])


def test_network_errors_do_not_expose_api_key(tmp_path, monkeypatch):
    client = DartClient(tmp_path, key="test-secret-do-not-log")

    def fail(*args, **kwargs):
        raise requests.ConnectionError("URL includes test-secret-do-not-log")

    monkeypatch.setattr(client.session, "get", fail)
    with pytest.raises(DartError) as error:
        client.fetch("empSttus.json", {"corp_code": "00000001"})
    assert "test-secret" not in str(error.value)


def test_cache_does_not_include_key_or_repeat_network_call(tmp_path, monkeypatch):
    client = DartClient(tmp_path, key="fake-secret")
    calls = []

    def response(endpoint, params):
        calls.append(endpoint)
        return b'{"status":"013","message":"no data"}'

    monkeypatch.setattr(client, "_get", response)
    for _ in range(2):
        assert client.fetch("empSttus.json", {"bsns_year": "2022"})["status"] == "013"
    assert len(calls) == 1
    assert all("fake-secret" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_collector_never_falls_back_to_another_year_or_scope(tmp_path, monkeypatch):
    import pandas as pd

    from data_final.dart import collect

    monkeypatch.setenv("DART_API_KEY", "fixture-not-a-real-key")
    universe = tmp_path / "universe.csv"
    universe.write_text("stock_code\n005930\n")
    monkeypatch.setattr(
        DartClient,
        "corporations",
        lambda self: pd.DataFrame(
            [{"stock_code": "005930", "corp_code": "00126380", "corp_name": "fixture"}]
        ),
    )
    calls = []

    def fetch(self, endpoint, params):
        calls.append((endpoint, params))
        return {"status": "013"}

    monkeypatch.setattr(DartClient, "fetch", fetch)
    output = tmp_path / "panel.csv"
    collect(universe, [2022], output, tmp_path / "cache")
    assert len(calls) == 2
    assert all(params["bsns_year"] == "2022" for _, params in calls)
    assert all(params["reprt_code"] == "11011" for _, params in calls)
    assert calls[0][1]["fs_div"] == "OFS"
    result = pd.read_csv(output)
    assert result.net_income.isna().all()
    assert output.with_suffix(".audit.json").exists()
