import urllib.error
import pytest
import main as m


def test_offline_fallback_metadata():
    result = m._build_offline_fallback_pr(
        "FalkorDB/FalkorDB",
        "test_network_failure"
    )

    assert result["ingestion_source"] == "fallback_offline"
    assert result["provenance_type"] == "offline_fallback"
    assert result["adr_status"] == "synthetic_fallback"
    assert result["fallback_reason"] == "test_network_failure"
    assert result["is_merged"] is False
    assert result["merged_at_epoch"] is None
    assert result["merged_at_iso"] is None


def test_network_error_triggers_fallback(monkeypatch):
    module_globals = m._fetch_live_github_pr.__globals__

    def simulated_failure(*args, **kwargs):
        raise urllib.error.URLError("simulated_network_failure")

    if "urlopen" in module_globals:
        monkeypatch.setitem(module_globals, "urlopen", simulated_failure)
    elif "urllib" in module_globals and hasattr(module_globals["urllib"], "request"):
        monkeypatch.setattr(
            module_globals["urllib"].request,
            "urlopen",
            simulated_failure
        )
    elif "request" in module_globals and hasattr(module_globals["request"], "urlopen"):
        monkeypatch.setattr(
            module_globals["request"],
            "urlopen",
            simulated_failure
        )
    else:
        pytest.fail("Could not locate GitHub HTTP opener")

    result = m._fetch_live_github_pr("FalkorDB/FalkorDB", 3063)

    assert result["ingestion_source"] == "fallback_offline"
    assert result["provenance_type"] == "offline_fallback"
    assert result["adr_status"] == "synthetic_fallback"
    assert result["fallback_reason"].startswith("network_")


def test_invalid_pr_number_is_rejected():
    with pytest.raises(Exception) as exc:
        m._fetch_live_github_pr("FalkorDB/FalkorDB", 0)

    assert getattr(exc.value, "status_code", None) == 400
