import inspect
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from fastapi.testclient import TestClient
from embeddings import get_embedding
import graph_tools as gt
from agent_swarm import run_incident_investigation, AgentState
import main as main_app

WARMUP_RUNS = int(os.environ.get("BENCHMARK_WARMUP_RUNS", 5))
BENCHMARK_RUNS = int(os.environ.get("BENCHMARK_QUERY_RUNS", 50))
E2E_RUNS = int(os.environ.get("BENCHMARK_E2E_RUNS", 20))

DEFAULT_BENCHMARK_ERROR_LOG = (
    "CheckoutService HTTP 502 Bad Gateway: upstream PaymentService rejected by "
    "AuthTokenService (401 JWT signature mismatch)"
)


def nearest_rank_percentile(sorted_vals: list[float], pct: float) -> float:
    """
    Computes the nearest-rank percentile: index = ceil(pct * n) - 1.
    For n=20 and pct=0.95: ceil(19.0) - 1 = 18 (the 19th element, distinct from Max at index 19).
    """
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = max(0, min(n - 1, math.ceil(pct * n) - 1))
    return sorted_vals[idx]


@dataclass
class BenchmarkArtifactTracker:
    """Tracks exact benchmark-created IDs so cleanup never touches user or seeded data."""
    run_prefix: str = field(default_factory=lambda: f"bench-{uuid.uuid4().hex[:8]}")
    memory_ids: set[str] = field(default_factory=set)
    session_ids: set[str] = field(default_factory=set)
    turn_ids: set[str] = field(default_factory=set)
    triage_session_ids: set[str] = field(default_factory=set)

    def new_memory_id(self, tag: str = "q6") -> str:
        mid = f"MEM-{self.run_prefix}-{tag}-{uuid.uuid4().hex[:6]}"
        self.memory_ids.add(mid)
        return mid

    def new_session_id(self, tag: str = "e2e") -> str:
        sid = f"sess-{self.run_prefix}-{tag}-{uuid.uuid4().hex[:6]}"
        self.session_ids.add(sid)
        self.triage_session_ids.add(sid)
        return sid

    def register_from_state(self, state: dict | None):
        if not state:
            return
        sid = state.get("session_id")
        if sid and sid != "sess-baseline-001":
            self.session_ids.add(sid)
            self.triage_session_ids.add(sid)
        mid = state.get("remediation_data", {}).get("stored_memory_id")
        if mid and mid != "MEM-301":
            self.memory_ids.add(mid)
        for section in ("triage_data", "memory_data", "remediation_data"):
            for tel in state.get(section, {}).get("tool_telemetry", []):
                if tel.get("tool") == "record_session_turn":
                    tid = tel.get("data", {}).get("turn_id")
                    if tid:
                        self.turn_ids.add(tid)

    def cleanup_single_iteration(
        self,
        session_id: str | None = None,
        memory_id: str | None = None,
        turn_ids: list[str] | None = None,
        tenant_id: str = "acme_corp",
    ):
        """Targeted cleanup of a single iteration's artifacts (outside the benchmark timer)."""
        graph = gt.get_graph(tenant_id)
        if memory_id and memory_id != "MEM-301":
            graph.query(
                "MATCH (m:ResolutionMemory {memory_id: $mid}) DETACH DELETE m",
                {"mid": memory_id},
            )
            self.memory_ids.discard(memory_id)

        if turn_ids:
            graph.query(
                "MATCH (t:ConversationTurn) WHERE t.turn_id IN $tids DETACH DELETE t",
                {"tids": list(turn_ids)},
            )
            for tid in turn_ids:
                self.turn_ids.discard(tid)

        if session_id and session_id != "sess-baseline-001":
            graph.query(
                "MATCH (t:ConversationTurn {session_id: $sid}) DETACH DELETE t",
                {"sid": session_id},
            )
            graph.query(
                "MATCH (tr:TriageResult {session_id: $sid}) DETACH DELETE tr",
                {"sid": session_id},
            )
            graph.query(
                "MATCH (s:Session {session_id: $sid}) DETACH DELETE s",
                {"sid": session_id},
            )
            self.session_ids.discard(session_id)
            self.triage_session_ids.discard(session_id)

    def cleanup_all(self, tenant_id: str = "acme_corp"):
        """Targeted cleanup of any remaining tracked artifacts created by this benchmark run."""
        graph = gt.get_graph(tenant_id)
        if self.memory_ids:
            graph.query(
                "MATCH (m:ResolutionMemory) WHERE m.memory_id IN $mids DETACH DELETE m",
                {"mids": list(self.memory_ids)},
            )
            self.memory_ids.clear()

        if self.turn_ids:
            graph.query(
                "MATCH (t:ConversationTurn) WHERE t.turn_id IN $tids DETACH DELETE t",
                {"tids": list(self.turn_ids)},
            )
            self.turn_ids.clear()

        if self.triage_session_ids:
            graph.query(
                "MATCH (tr:TriageResult) WHERE tr.session_id IN $sids DETACH DELETE tr",
                {"sids": list(self.triage_session_ids)},
            )
            self.triage_session_ids.clear()

        if self.session_ids:
            graph.query(
                "MATCH (t:ConversationTurn) WHERE t.session_id IN $sids DETACH DELETE t",
                {"sids": list(self.session_ids)},
            )
            graph.query(
                "MATCH (s:Session) WHERE s.session_id IN $sids DETACH DELETE s",
                {"sids": list(self.session_ids)},
            )
            self.session_ids.clear()


def _canonical_node_identity(node_obj: Any) -> tuple[tuple[str, ...], str]:
    """Extracts a deterministic (labels, canonical_key) tuple for a FalkorDB node object."""
    labels = tuple(sorted(node_obj.labels)) if hasattr(node_obj, "labels") and node_obj.labels else ("Node",)
    props = dict(node_obj.properties) if hasattr(node_obj, "properties") else {}
    primary_key = (
        props.get("name")
        or props.get("pr_id")
        or props.get("alert_id")
        or props.get("memory_id")
        or props.get("adr_id")
        or props.get("thread_id")
        or props.get("step_id")
        or props.get("deploy_id")
        or props.get("turn_id")
        or props.get("session_id")
        or str(getattr(node_obj, "id", ""))
    )
    return labels, str(primary_key)


def _canonical_node_fingerprint(node_obj: Any) -> tuple[tuple[str, ...], str, tuple[tuple[str, str], ...]]:
    """Captures labels, primary key, and non-vector property key/value pairs for invariant verification."""
    labels, primary_key = _canonical_node_identity(node_obj)
    props = dict(node_obj.properties) if hasattr(node_obj, "properties") else {}
    prop_items = tuple(
        sorted((str(k), repr(v)) for k, v in props.items() if k != "error_embedding")
    )
    return labels, primary_key, prop_items


def get_graph_snapshot(tenant_id: str = "acme_corp") -> dict:
    """
    Captures verified graph invariants:
    - Total node and edge counts
    - IncidentAlert statuses and mttr_minutes
    - Microservice statuses
    - Record ID sets for ResolutionMemory, Session, ConversationTurn, and TriageResult
    - Full node property fingerprints (excluding large vector embeddings)
    - Full relationship triples (source_identity, relation_type, target_identity)
    """
    graph = gt.get_graph(tenant_id)
    n_res = graph.query("MATCH (n) RETURN count(n)")
    e_res = graph.query("MATCH ()-[r]->() RETURN count(r)")
    a_res = graph.query("MATCH (a:IncidentAlert) RETURN a.alert_id, a.status, a.mttr_minutes")
    svc_res = graph.query("MATCH (s:Microservice) RETURN s.name, s.status")
    m_res = graph.query("MATCH (m:ResolutionMemory) RETURN collect(m.memory_id)")
    s_res = graph.query("MATCH (s:Session) RETURN collect(s.session_id)")
    t_res = graph.query("MATCH (t:ConversationTurn) RETURN collect(t.turn_id)")
    tr_res = graph.query("MATCH (tr:TriageResult) RETURN collect(tr.session_id)")
    topo_res = graph.query("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m")

    node_fingerprints = set()
    edge_triples = set()
    for row in topo_res.result_set:
        n_obj, r_obj, m_obj = row[0], row[1], row[2]
        if n_obj is not None:
            node_fingerprints.add(_canonical_node_fingerprint(n_obj))
        if m_obj is not None:
            node_fingerprints.add(_canonical_node_fingerprint(m_obj))
        if r_obj is not None and n_obj is not None and m_obj is not None:
            rel_type = str(getattr(r_obj, "relation", "CONNECTED"))
            edge_triples.add((
                _canonical_node_identity(n_obj),
                rel_type,
                _canonical_node_identity(m_obj),
            ))

    return {
        "nodes": n_res.result_set[0][0] if n_res.result_set else 0,
        "edges": e_res.result_set[0][0] if e_res.result_set else 0,
        "alert_states": {r[0]: (r[1], r[2]) for r in a_res.result_set},
        "service_states": {r[0]: r[1] for r in svc_res.result_set},
        "memories": set(m_res.result_set[0][0]) if m_res.result_set else set(),
        "sessions": set(s_res.result_set[0][0]) if s_res.result_set else set(),
        "turns": set(t_res.result_set[0][0]) if t_res.result_set else set(),
        "triage_results": set(tr_res.result_set[0][0]) if tr_res.result_set else set(),
        "node_fingerprints": node_fingerprints,
        "edge_triples": edge_triples,
    }


def restore_graph_snapshot(snapshot: dict, tenant_id: str = "acme_corp"):
    """Restores exact pre-benchmark IncidentAlert and Microservice statuses and verifies restoration."""
    graph = gt.get_graph(tenant_id)
    for alert_id, (status, mttr) in snapshot.get("alert_states", {}).items():
        graph.query(
            "MATCH (a:IncidentAlert {alert_id: $aid}) SET a.status = $st, a.mttr_minutes = $mttr",
            {"aid": alert_id, "st": status, "mttr": mttr},
        )
    for svc_name, status in snapshot.get("service_states", {}).items():
        graph.query(
            "MATCH (s:Microservice {name: $name}) SET s.status = $st",
            {"name": svc_name, "st": status},
        )


def safe_cleanup_and_restore(
    tracker: BenchmarkArtifactTracker,
    snap_acme: dict | None = None,
    snap_beta: dict | None = None,
):
    """
    Guarantees that restore_graph_snapshot runs for both tenants even if tracker.cleanup_all raises.
    Never suppresses exceptions from cleanup or restoration.
    """
    try:
        tracker.cleanup_all("acme_corp")
    finally:
        try:
            if snap_acme is not None:
                restore_graph_snapshot(snap_acme, "acme_corp")
        finally:
            if snap_beta is not None:
                restore_graph_snapshot(snap_beta, "beta_health")


def prepare_and_verify_initial_incident_state(tenant_id: str = "acme_corp") -> bool:
    """
    Sets ALERT-901 to 'open' and CheckoutService to 'down' before a benchmarked run begins,
    and verifies the graph is in the expected initial outage state.
    """
    graph = gt.get_graph(tenant_id)
    graph.query(
        """
        MATCH (a:IncidentAlert {alert_id: 'ALERT-901'})
        SET a.status = 'open', a.mttr_minutes = 0
        WITH a
        MATCH (s:Microservice)
        WHERE s.name IN ['CheckoutService', 'OrderService', 'PaymentService', 'AuthTokenService']
        SET s.status = CASE WHEN s.name = 'CheckoutService' THEN 'down' ELSE 'degraded' END
        """
    )
    check = graph.query(
        """
        MATCH (a:IncidentAlert {alert_id: 'ALERT-901'})-[:TRIGGERED_ON]->(s:Microservice {name: 'CheckoutService'})
        RETURN a.status, s.status
        """
    )
    if not check.result_set:
        return False
    return check.result_set[0][0] == "open" and check.result_set[0][1] == "down"


def verify_persisted_investigation_in_db(
    session_id: str,
    memory_id: str,
    tenant_id: str = "acme_corp",
    alert_id: str = "ALERT-901",
) -> bool:
    """Verifies that the investigation physically persisted its state and healed services in FalkorDB."""
    graph = gt.get_graph(tenant_id)
    res = graph.query(
        """
        MATCH (a:IncidentAlert {alert_id: $aid})-[:TRIGGERED_ON]->(s:Microservice {name: 'CheckoutService'})
        MATCH (m:ResolutionMemory {memory_id: $mid})-[:RESOLVED]->(a)
        MATCH (tr:TriageResult {session_id: $sid})-[:EVALUATED]->(a)
        MATCH (sess:Session {session_id: $sid})
        OPTIONAL MATCH (t:ConversationTurn {session_id: $sid})-[:IN_SESSION]->(sess)
        RETURN a.status, s.status, m.memory_id, tr.suspect_pr, tr.blast_radius_count, count(t)
        """,
        {"aid": alert_id, "mid": memory_id, "sid": session_id},
    )
    if not res.result_set:
        return False
    row = res.result_set[0]
    healed_res = graph.query(
        """
        MATCH (svc:Microservice)
        WHERE svc.name IN ['CheckoutService', 'OrderService', 'PaymentService', 'AuthTokenService']
        RETURN collect(DISTINCT svc.status)
        """
    )
    statuses = healed_res.result_set[0][0] if healed_res.result_set else []
    return (
        row[0] == "resolved"
        and row[1] == "healthy"
        and row[2] == memory_id
        and row[3] == "PR-402"
        and row[4] == 4
        and row[5] >= 2
        and statuses == ["healthy"]
    )


def validate_investigation_state(
    state: AgentState,
    session_id: str,
    memory_id: str,
    alert_id: str = "ALERT-901",
) -> bool:
    """
    Strict correctness validator:
    - Verifies all 4 agent stages genuinely executed.
    - Rejects any run where fallback dictionaries were used.
    - Verifies non-empty DB rows for root cause, blast radius, criticality, handoffs, memory recall, experts, and session history.
    - Verifies physical persistence in FalkorDB.
    """
    expected_agents = [
        "Orchestrator Agent",
        "Triage & Code Analyst Agent",
        "Memory & Knowledge Agent",
        "Remediation Agent",
    ]
    if state.get("executed_agents") != expected_agents:
        return False

    triage = state.get("triage_data", {})
    memory = state.get("memory_data", {})
    remed = state.get("remediation_data", {})

    if (
        triage.get("used_fallback")
        or triage.get("used_crit_fallback")
        or triage.get("used_br_fallback")
        or memory.get("used_fallback")
        or remed.get("used_expert_fallback")
    ):
        return False

    triage_tel = triage.get("tool_telemetry", [])
    mem_tel = memory.get("tool_telemetry", [])
    rem_tel = remed.get("tool_telemetry", [])
    if len(triage_tel) != 5 or len(mem_tel) != 3 or len(rem_tel) != 3:
        return False

    # 1. Root cause (Q1)
    if not triage_tel[0].get("data") or triage_tel[0]["data"][0].get("pr_id") != "PR-402":
        return False
    # 2. Blast radius (Q2)
    br_rows = triage_tel[1].get("data", [])
    if len(br_rows) != 4 or triage.get("blast_radius_count") != 4:
        return False
    # 3. Criticality (Q3)
    crit_rows = triage_tel[2].get("data", [])
    if len(crit_rows) < 4 or "criticality" not in crit_rows[0]:
        return False
    # 4. Handoff write & Turn write
    if triage_tel[3].get("data", {}).get("blast_radius_count") != 4:
        return False
    if not triage_tel[4].get("data", {}).get("turn_id"):
        return False
    # 5. Handoff read, Memory recall (Q5), and Turn write
    ho_read = mem_tel[0].get("data", {})
    if not memory.get("handoff_verified") or ho_read.get("blast_radius_count") != 4 or ho_read.get("suspect_pr") != "PR-402":
        return False
    if not mem_tel[1].get("data") or mem_tel[1]["data"][0].get("memory_id") != "MEM-301":
        return False
    if len(memory.get("runbook_steps", [])) != 3 or not mem_tel[2].get("data", {}).get("turn_id"):
        return False
    # 6. Experts (Q8), Store Memory (Q6), Session History (Q7)
    exp_rows = rem_tel[0].get("data", [])
    if not exp_rows or exp_rows[0].get("expert") != "Priya Nair" or not exp_rows[0].get("is_owner"):
        return False
    if rem_tel[1].get("data", {}).get("stored_memory_id") != memory_id:
        return False
    hist_rows = rem_tel[2].get("data", [])
    ref_entities = [r.get("referenced_entity") for r in hist_rows]
    if len(hist_rows) < 2 or "PR-402" not in ref_entities or "MEM-301" not in ref_entities:
        return False

    return verify_persisted_investigation_in_db(session_id, memory_id, alert_id=alert_id)


def _inspect_main_endpoint_sleep_policy() -> str:
    """Inspects main.trigger_incident_stream source to disclose whether presentation sleeps exist."""
    try:
        src = inspect.getsource(main_app.trigger_incident_stream)
        if "sleep(0.6)" in src or "sleep(0.9)" in src or "sleep(0.4)" in src:
            return "INCLUDES presentation asyncio.sleep() delays in main.py"
        if "sleep(0)" in src:
            return "Zero presentation delay (asyncio.sleep(0) in main.py)"
    except Exception:
        pass
    return "In-process ASGI stream"


def _is_mode_valid_for_comparison(mode_stats: dict, expected_runs: int) -> tuple[bool, str]:
    """
    Verifies that a benchmark mode completed all expected runs with zero validation,
    exception, or cleanup failures and has a positive duration.
    """
    if expected_runs <= 0:
        return False, f"expected_runs={expected_runs} <= 0"
    if mode_stats.get("ok", 0) != expected_runs:
        return False, f"ok={mode_stats.get('ok', 0)}/{expected_runs}"
    if mode_stats.get("val_fail", 0) != 0:
        return False, f"val_fail={mode_stats.get('val_fail')}"
    if mode_stats.get("exc_fail", 0) != 0:
        return False, f"exc_fail={mode_stats.get('exc_fail')}"
    if mode_stats.get("cleanup_fail", 0) != 0:
        return False, f"cleanup_fail={mode_stats.get('cleanup_fail')}"
    if mode_stats.get("sample_count", 0) != expected_runs:
        return False, f"sample_count={mode_stats.get('sample_count', 0)}/{expected_runs}"
    if mode_stats.get("avg", 0.0) <= 0.0:
        return False, "avg <= 0.0ms"
    return True, "valid"


def format_speedup_comparison(
    label: str,
    baseline_mode: dict,
    target_mode: dict,
    expected_runs: int,
) -> str:
    """
    Computes and formats a speedup comparison only if both compared modes are 100% valid.
    Otherwise explicitly reports INVALID / NOT COMPARABLE.
    """
    base_ok, base_reason = _is_mode_valid_for_comparison(baseline_mode, expected_runs)
    targ_ok, targ_reason = _is_mode_valid_for_comparison(target_mode, expected_runs)
    if not base_ok or not targ_ok:
        reasons = []
        if not base_ok:
            reasons.append(f"baseline [{baseline_mode.get('name', 'Base')}] {base_reason}")
        if not targ_ok:
            reasons.append(f"target [{target_mode.get('name', 'Target')}] {targ_reason}")
        return f"• {label:68s} INVALID / NOT COMPARABLE ({'; '.join(reasons)})"

    base_avg = baseline_mode["avg"]
    targ_avg = target_mode["avg"]
    ratio = base_avg / targ_avg
    pct_change = 100.0 * (base_avg - targ_avg) / base_avg
    direction = "faster" if ratio >= 1.0 else "slower (thread dispatch overhead exceeds local query savings)"
    return (
        f"• {label:68s} {ratio:5.2f}x ({base_avg:5.2f}ms -> {targ_avg:5.2f}ms, "
        f"{pct_change:+.1f}% | {direction})"
    )


def run_single_query_benchmark(tracker: BenchmarkArtifactTracker) -> list[dict]:
    """Benchmarks all 10 queries with per-iteration state isolation for mutating Query 6."""
    # Prime connection pool and warm embedding cache before query measurements
    gt.get_db()
    get_embedding(DEFAULT_BENCHMARK_ERROR_LOG)

    def _run_q6_isolated() -> tuple[float, bool]:
        mid = tracker.new_memory_id("q6")
        if not prepare_and_verify_initial_incident_state("acme_corp"):
            raise RuntimeError("Pre-Q6 incident state invalid")
        try:
            t0 = time.perf_counter()
            res = gt.store_resolution_memory(
                alert_id="ALERT-901",
                summary="Benchmark rollback verification",
                steps_taken="Benchmark step execution",
                mttr_minutes=2,
                tenant_id="acme_corp",
                custom_memory_id=mid,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = gt.get_graph("acme_corp")
            chk = graph.query(
                "MATCH (m:ResolutionMemory {memory_id: $mid})-[:RESOLVED]->(a:IncidentAlert {alert_id: 'ALERT-901'}) RETURN a.status",
                {"mid": mid},
            )
            valid = (
                bool(chk.result_set)
                and chk.result_set[0][0] == "resolved"
                and res["data"]["stored_memory_id"] == mid
            )
            return elapsed_ms, valid
        finally:
            try:
                tracker.cleanup_single_iteration(memory_id=mid, tenant_id="acme_corp")
            finally:
                if not prepare_and_verify_initial_incident_state("acme_corp"):
                    raise RuntimeError("Post-Q6 incident state restoration failed")

    queries: list[tuple[str, Callable[[], dict] | None, Callable[[dict], bool] | None]] = [
        (
            "Query 1: 4-Hop Root Cause + PR Risk Sort",
            lambda: gt.find_root_cause("ALERT-901", "acme_corp"),
            lambda r: bool(r.get("data")) and r["data"][0]["pr_id"] == "PR-402" and r["data"][0]["risk_score"] == 4,
        ),
        (
            "Query 2: 6-Hop Transitive Blast Radius",
            lambda: gt.get_blast_radius("AuthTokenService", "acme_corp"),
            lambda r: bool(r.get("data")) and len(r["data"]) == 4,
        ),
        (
            "Query 3: Infrastructure Criticality",
            lambda: gt.get_critical_services("acme_corp"),
            lambda r: bool(r.get("data")) and len(r["data"]) >= 4,
        ),
        (
            "Query 4: Shortest Dependency Path",
            lambda: gt.get_shortest_dependency_path("APIGateway", "AuthTokenService", "acme_corp"),
            lambda r: bool(r.get("data")) and r["data"][0]["hops"] == 3,
        ),
        (
            "Query 5: Hybrid Vector + Graph Memory",
            lambda: gt.recall_incident_memory("401 JWT signature mismatch on AuthTokenService", tenant_id="acme_corp"),
            lambda r: bool(r.get("data")) and r["data"][0]["memory_id"] == "MEM-301" and len(r["data"][0]["ordered_runbook_steps"]) == 3,
        ),
        ("Query 6: Store Resolution Memory & Heal", None, None),
        (
            "Query 7: Multi-Agent Session History",
            lambda: gt.get_session_history("sess-baseline-001", "acme_corp"),
            lambda r: bool(r.get("data")) and len(r["data"]) >= 2,
        ),
        (
            "Query 8: Domain Expert Ranking",
            lambda: gt.get_service_experts("CheckoutService", "acme_corp"),
            lambda r: bool(r.get("data")) and r["data"][0]["expert"] == "Aarav Sharma",
        ),
        (
            "Query 9: ADR + Slack Provenance",
            lambda: gt.get_decision_provenance("CheckoutService", "acme_corp"),
            lambda r: bool(r.get("data")) and r["data"][0]["adr_id"] == "ADR-019",
        ),
        (
            "Query 10: 2-Hop Neighborhood Changes",
            lambda: gt.get_recent_neighborhood_changes("CheckoutService", "acme_corp"),
            lambda r: bool(r.get("data")) and len(r["data"]) >= 2,
        ),
    ]

    results_summary = []
    for name, fn, validator in queries:
        times = []
        ok = 0
        val_fail = 0
        exc_fail = 0

        if fn is None:
            for _ in range(WARMUP_RUNS):
                _run_q6_isolated()
            for _ in range(BENCHMARK_RUNS):
                try:
                    ms, is_valid = _run_q6_isolated()
                    if is_valid:
                        ok += 1
                        times.append(ms)
                    else:
                        val_fail += 1
                except Exception:
                    exc_fail += 1
        else:
            for _ in range(WARMUP_RUNS):
                fn()
            for _ in range(BENCHMARK_RUNS):
                try:
                    t0 = time.perf_counter()
                    res = fn()
                    ms = (time.perf_counter() - t0) * 1000
                    if validator is not None and validator(res):
                        ok += 1
                        times.append(ms)
                    else:
                        val_fail += 1
                except Exception:
                    exc_fail += 1

        times.sort()
        avg_ms = statistics.mean(times) if times else 0.0
        med_ms = statistics.median(times) if times else 0.0
        p95_ms = nearest_rank_percentile(times, 0.95)
        min_ms = times[0] if times else 0.0
        max_ms = times[-1] if times else 0.0
        stat_entry = {
            "name": name,
            "avg": avg_ms,
            "med": med_ms,
            "p95": p95_ms,
            "min": min_ms,
            "max": max_ms,
            "ok": ok,
            "val_fail": val_fail,
            "exc_fail": exc_fail,
            "sample_count": len(times),
        }
        results_summary.append(stat_entry)
        print(
            f"{name:40s} | {avg_ms:5.2f}ms | {med_ms:5.2f}ms | {p95_ms:5.2f}ms | {min_ms:5.2f}ms | {max_ms:5.2f}ms | {ok:2d}/{val_fail:1d}/{exc_fail:1d}"
        )
    return results_summary


def _execute_single_e2e_iteration(
    tracker: BenchmarkArtifactTracker,
    parallel: bool,
    use_pooling: bool,
    use_cache: bool,
    use_langgraph_engine: bool,
    tag: str = "e2e",
) -> tuple[str, float, float, int]:
    """
    Executes a single isolated 4-agent workflow iteration.
    Returns (status_code, elapsed_ms, total_db_ms, db_client_instantiations)
    where status_code is 'ok', 'val_fail', 'exc_fail', or 'cleanup_fail'.
    """
    sid = tracker.new_session_id(tag)
    mid = tracker.new_memory_id(tag)
    state = None
    status_code = "ok"
    elapsed_ms = 0.0
    total_db_ms = 0.0
    actual_conns = 0

    try:
        if not prepare_and_verify_initial_incident_state("acme_corp"):
            return ("val_fail", 0.0, 0.0, 0)

        if use_pooling:
            gt.get_db()
        if use_cache:
            # Ensure @lru_cache is warm for the target incident error log before timing
            get_embedding(DEFAULT_BENCHMARK_ERROR_LOG)

        t0 = time.perf_counter()
        state = run_incident_investigation(
            "ALERT-901",
            "acme_corp",
            parallel=parallel,
            use_pooling=use_pooling,
            use_cache=use_cache,
            session_id=sid,
            custom_memory_id=mid,
            use_langgraph_engine=use_langgraph_engine,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tracker.register_from_state(state)

        expected_conns = 0 if use_pooling else 11
        expected_emb = 0 if use_cache else 1
        expected_closed = 0 if use_pooling else 11
        remed = state.get("remediation_data", {})
        actual_conns = remed.get("db_client_instantiations", -1)
        actual_emb = remed.get("uncached_embeddings", -1)
        actual_closed = remed.get("unpooled_clients_closed", -1)
        actual_unclosed = remed.get("unclosed_client_count", -1)
        cleanup_errs = remed.get("cleanup_errors", ["err"])
        total_db_ms = float(state.get("total_db_ms", 0.0))

        if (
            validate_investigation_state(state, sid, mid)
            and actual_conns == expected_conns
            and actual_emb == expected_emb
            and actual_closed == expected_closed
            and actual_unclosed == 0
            and len(cleanup_errs) == 0
        ):
            status_code = "ok"
        else:
            status_code = "val_fail"
    except Exception:
        status_code = "exc_fail"
    finally:
        try:
            tracker.cleanup_single_iteration(sid, mid, tenant_id="acme_corp")
        except Exception:
            status_code = "cleanup_fail"
        finally:
            if not prepare_and_verify_initial_incident_state("acme_corp"):
                status_code = "cleanup_fail"

    return (status_code, elapsed_ms, total_db_ms, actual_conns)


def _summarize_mode_Accumulator(acc: dict) -> dict:
    wall_times = sorted(acc["wall_times"])
    db_sums = acc["db_sums"]
    conns = acc["conns"]
    return {
        "name": acc["name"],
        "avg": statistics.mean(wall_times) if wall_times else 0.0,
        "med": statistics.median(wall_times) if wall_times else 0.0,
        "p95": nearest_rank_percentile(wall_times, 0.95),
        "min": wall_times[0] if wall_times else 0.0,
        "max": wall_times[-1] if wall_times else 0.0,
        "db_avg": statistics.mean(db_sums) if db_sums else 0.0,
        "conns": int(statistics.mean(conns)) if conns else 0,
        "ok": acc["ok"],
        "val_fail": acc["val_fail"],
        "exc_fail": acc["exc_fail"],
        "cleanup_fail": acc["cleanup_fail"],
        "sample_count": len(wall_times),
    }


def benchmark_e2e_mode(
    tracker: BenchmarkArtifactTracker,
    mode_name: str,
    parallel: bool,
    use_pooling: bool = True,
    use_cache: bool = True,
    use_langgraph_engine: bool = False,
) -> dict:
    """
    Standalone single-mode E2E benchmark runner (preserved for external callers/tests).
    """
    for _ in range(WARMUP_RUNS):
        _execute_single_e2e_iteration(
            tracker, parallel, use_pooling, use_cache, use_langgraph_engine, tag="warmup"
        )

    acc = {
        "name": mode_name,
        "wall_times": [],
        "db_sums": [],
        "conns": [],
        "ok": 0,
        "val_fail": 0,
        "exc_fail": 0,
        "cleanup_fail": 0,
    }
    for _ in range(E2E_RUNS):
        st, ms, db_ms, c_cnt = _execute_single_e2e_iteration(
            tracker, parallel, use_pooling, use_cache, use_langgraph_engine, tag="e2e"
        )
        if st == "ok":
            acc["ok"] += 1
            acc["wall_times"].append(ms)
            acc["db_sums"].append(db_ms)
            acc["conns"].append(c_cnt)
        elif st == "val_fail":
            acc["val_fail"] += 1
        elif st == "cleanup_fail":
            acc["cleanup_fail"] += 1
        else:
            acc["exc_fail"] += 1

    return _summarize_mode_Accumulator(acc)


def benchmark_interleaved_e2e_modes(
    tracker: BenchmarkArtifactTracker, mode_configs: list[dict]
) -> dict[str, dict]:
    """
    Executes warm-up runs for all E2E modes first, then runs E2E_RUNS measured rounds
    in rotating interleaved order to eliminate execution-order and thermal/JIT drift bias.
    """
    accumulators: dict[str, dict] = {}
    for cfg in mode_configs:
        key = cfg["key"]
        accumulators[key] = {
            "name": cfg["name"],
            "wall_times": [],
            "db_sums": [],
            "conns": [],
            "ok": 0,
            "val_fail": 0,
            "exc_fail": 0,
            "cleanup_fail": 0,
        }

    # 1. Dedicated warm-up phase across all modes
    for _ in range(WARMUP_RUNS):
        for cfg in mode_configs:
            _execute_single_e2e_iteration(
                tracker,
                parallel=cfg["parallel"],
                use_pooling=cfg["use_pooling"],
                use_cache=cfg["use_cache"],
                use_langgraph_engine=cfg["use_langgraph_engine"],
                tag="warmup",
            )

    # 2. Interleaved measured rounds (rotates starting mode each round)
    n_modes = len(mode_configs)
    for round_idx in range(E2E_RUNS):
        for offset in range(n_modes):
            cfg = mode_configs[(round_idx + offset) % n_modes]
            acc = accumulators[cfg["key"]]
            st, ms, db_ms, c_cnt = _execute_single_e2e_iteration(
                tracker,
                parallel=cfg["parallel"],
                use_pooling=cfg["use_pooling"],
                use_cache=cfg["use_cache"],
                use_langgraph_engine=cfg["use_langgraph_engine"],
                tag="e2e",
            )
            if st == "ok":
                acc["ok"] += 1
                acc["wall_times"].append(ms)
                acc["db_sums"].append(db_ms)
                acc["conns"].append(c_cnt)
            elif st == "val_fail":
                acc["val_fail"] += 1
            elif st == "cleanup_fail":
                acc["cleanup_fail"] += 1
            else:
                acc["exc_fail"] += 1

    return {k: _summarize_mode_Accumulator(v) for k, v in accumulators.items()}


def benchmark_http_sse_endpoint(tracker: BenchmarkArtifactTracker) -> dict:
    """Measures in-process FastAPI ASGI HTTP/SSE streaming latency with per-run isolation and cleanup."""
    client = TestClient(main_app.app)
    http_times = []
    ok = 0
    val_fail = 0
    exc_fail = 0
    cleanup_fail = 0

    for _ in range(WARMUP_RUNS):
        w_sid = None
        w_mid = None
        try:
            prepare_and_verify_initial_incident_state("acme_corp")
            last_state = None
            with client.stream(
                "POST",
                "/api/incident/trigger",
                json={"alert_id": "ALERT-901", "tenant_id": "acme_corp"},
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        last_state = json.loads(line[6:]).get("state")
            if last_state:
                tracker.register_from_state(last_state)
                w_sid = last_state.get("session_id")
                w_mid = last_state.get("remediation_data", {}).get("stored_memory_id")
        finally:
            try:
                tracker.cleanup_single_iteration(w_sid, w_mid, tenant_id="acme_corp")
            finally:
                prepare_and_verify_initial_incident_state("acme_corp")

    for _ in range(E2E_RUNS):
        created_sid = None
        created_mid = None
        iter_ok = False
        try:
            if not prepare_and_verify_initial_incident_state("acme_corp"):
                val_fail += 1
                continue
            t0 = time.perf_counter()
            last_state = None
            with client.stream(
                "POST",
                "/api/incident/trigger",
                json={"alert_id": "ALERT-901", "tenant_id": "acme_corp"},
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        last_state = payload.get("state")
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if last_state:
                tracker.register_from_state(last_state)
                created_sid = last_state.get("session_id")
                created_mid = last_state.get("remediation_data", {}).get("stored_memory_id")

            if (
                last_state
                and created_sid
                and created_mid
                and validate_investigation_state(last_state, created_sid, created_mid)
            ):
                iter_ok = True
            else:
                val_fail += 1
        except Exception:
            exc_fail += 1
        finally:
            cleanup_ok = True
            try:
                tracker.cleanup_single_iteration(created_sid, created_mid, tenant_id="acme_corp")
            except Exception:
                cleanup_ok = False
            finally:
                if not prepare_and_verify_initial_incident_state("acme_corp"):
                    cleanup_ok = False
            if not cleanup_ok:
                cleanup_fail += 1
            elif iter_ok:
                ok += 1
                http_times.append(elapsed_ms)

    http_times.sort()
    return {
        "name": "6. In-Process ASGI HTTP/SSE (TestClient)",
        "avg": statistics.mean(http_times) if http_times else 0.0,
        "med": statistics.median(http_times) if http_times else 0.0,
        "p95": nearest_rank_percentile(http_times, 0.95),
        "min": http_times[0] if http_times else 0.0,
        "max": http_times[-1] if http_times else 0.0,
        "ok": ok,
        "val_fail": val_fail,
        "exc_fail": exc_fail,
        "cleanup_fail": cleanup_fail,
        "sample_count": len(http_times),
    }


def verify_graph_snapshots_match(before: dict, after: dict, tenant_label: str):
    """
    Compares all captured graph invariants and raises a detailed AssertionError
    if any count, status, ID set, node property fingerprint, or edge triple changed.
    """
    for key in (
        "nodes",
        "edges",
        "alert_states",
        "service_states",
        "memories",
        "sessions",
        "turns",
        "triage_results",
        "node_fingerprints",
        "edge_triples",
    ):
        if before.get(key) != after.get(key):
            raise AssertionError(
                f"Graph invariant mismatch on {tenant_label} for key '{key}':\n"
                f"  Before: {before.get(key)}\n"
                f"  After:  {after.get(key)}"
            )


def run_suite():
    tracker = BenchmarkArtifactTracker()
    snap_acme_before = get_graph_snapshot("acme_corp")
    snap_beta_before = get_graph_snapshot("beta_health")
    sleep_disclosure = _inspect_main_endpoint_sleep_policy()

    print("=" * 116)
    print(f"CORTEX AUDITED BENCHMARK SUITE (N={BENCHMARK_RUNS}/Query, N={E2E_RUNS}/E2E, Uniform Warm-up={WARMUP_RUNS})")
    print("Methodology: Per-iteration state reset outside timer; interleaved E2E rounds; nearest-rank P95 = ceil(0.95 * n) - 1")
    print("Cache Policy: 'Cached' modes measure warm @lru_cache hits (0 recomputations); 'Uncached' recomputes 768-d vector/run")
    print("=" * 116)
    print(f"{'QUERY NAME':40s} | {'AVG':>7s} | {'MEDIAN':>7s} | {'P95':>7s} | {'MIN':>7s} | {'MAX':>7s} | {'OK/VF/EX':>8s}")
    print("-" * 116)

    try:
        query_results = run_single_query_benchmark(tracker)

        print("=" * 116)
        print(f"\nCONTROLLED 4-AGENT WORKFLOW COMPARISON ({E2E_RUNS} INTERLEAVED ISOLATED RUNS PER MODE, 11 QUERIES/RUN)")
        print("-" * 116)
        print(f"{'EXECUTION MODE':48s} | {'AVG':>7s} | {'MEDIAN':>7s} | {'P95':>7s} | {'MIN':>7s} | {'MAX':>7s} | {'DB SUM':>7s} | {'OK/VF/EX':>8s}")
        print("-" * 116)

        mode_configs = [
            {
                "key": "seq_unpooled_uncached",
                "name": "1. Sequential, Unpooled, Uncached (11 Clients)",
                "parallel": False,
                "use_pooling": False,
                "use_cache": False,
                "use_langgraph_engine": False,
            },
            {
                "key": "seq_pooled_uncached",
                "name": "2. Sequential, Pooled, Uncached (0 New Clients)",
                "parallel": False,
                "use_pooling": True,
                "use_cache": False,
                "use_langgraph_engine": False,
            },
            {
                "key": "seq_pooled_cached",
                "name": "3. Sequential, Pooled, Cached (Warm LRU)",
                "parallel": False,
                "use_pooling": True,
                "use_cache": True,
                "use_langgraph_engine": False,
            },
            {
                "key": "par_pooled_cached",
                "name": "4. Parallel, Pooled, Cached (3-Thread Pool)",
                "parallel": True,
                "use_pooling": True,
                "use_cache": True,
                "use_langgraph_engine": False,
            },
            {
                "key": "langgraph_par_pooled_cached",
                "name": "5. Compiled LangGraph .invoke() (Parallel)",
                "parallel": True,
                "use_pooling": True,
                "use_cache": True,
                "use_langgraph_engine": True,
            },
        ]

        e2e_results = benchmark_interleaved_e2e_modes(tracker, mode_configs)
        m1 = e2e_results["seq_unpooled_uncached"]
        m2 = e2e_results["seq_pooled_uncached"]
        m3 = e2e_results["seq_pooled_cached"]
        m4 = e2e_results["par_pooled_cached"]
        m5 = e2e_results["langgraph_par_pooled_cached"]
        m6_http = benchmark_http_sse_endpoint(tracker)

        for m in (m1, m2, m3, m4, m5):
            total_fails = m["val_fail"] + m["cleanup_fail"]
            print(
                f"{m['name']:48s} | {m['avg']:5.2f}ms | {m['med']:5.2f}ms | {m['p95']:5.2f}ms | "
                f"{m['min']:5.2f}ms | {m['max']:5.2f}ms | {m['db_avg']:5.2f}ms | {m['ok']:2d}/{total_fails:1d}/{m['exc_fail']:1d}"
            )

        http_fails = m6_http["val_fail"] + m6_http["cleanup_fail"]
        print(
            f"{m6_http['name']:48s} | {m6_http['avg']:5.2f}ms | {m6_http['med']:5.2f}ms | {m6_http['p95']:5.2f}ms | "
            f"{m6_http['min']:5.2f}ms | {m6_http['max']:5.2f}ms | {'N/A':>7s} | {m6_http['ok']:2d}/{http_fails:1d}/{m6_http['exc_fail']:1d}"
        )
        print("-" * 116)
        print("SINGLE-VARIABLE & COMBINED ISOLATED COMPARISONS (Requires 100% Validated Runs):")
        print(
            format_speedup_comparison(
                "Pure Connection Pooling Effect (Mode 1 vs. Mode 2, both Seq/Uncached):",
                m1,
                m2,
                E2E_RUNS,
            )
        )
        print(
            format_speedup_comparison(
                "Pure Embedding Cache Effect (Mode 2 vs. Mode 3, both Seq/Pooled):",
                m2,
                m3,
                E2E_RUNS,
            )
        )
        print(
            format_speedup_comparison(
                "Combined Pooling + Caching Effect (Mode 1 vs. Mode 3, both Seq):",
                m1,
                m3,
                E2E_RUNS,
            )
        )
        print(
            format_speedup_comparison(
                "Pure Concurrency Effect (Mode 3 Seq vs. Mode 4 Parallel, both Pooled):",
                m3,
                m4,
                E2E_RUNS,
            )
        )
        print(
            format_speedup_comparison(
                "Combined Optimization (Mode 1 Unpooled Seq vs. Mode 4 Pooled Parallel):",
                m1,
                m4,
                E2E_RUNS,
            )
        )
        print("-" * 116)
        print("• Historical UI-Paced Reference: 2540.00 ms (NOT a backend baseline; included ~2,400ms presentation asyncio.sleep)")
        print(f"• Endpoint Sleep Policy Detected in main.py: {sleep_disclosure}")

        all_queries_ok = all(
            q["ok"] == BENCHMARK_RUNS and q["val_fail"] == 0 and q["exc_fail"] == 0
            for q in query_results
        )
        if not all_queries_ok:
            print("⚠️ WARNING: One or more single-query benchmarks did not complete all validated runs.")
        print("=" * 116)

    finally:
        safe_cleanup_and_restore(tracker, snap_acme_before, snap_beta_before)

    snap_acme_after = get_graph_snapshot("acme_corp")
    snap_beta_after = get_graph_snapshot("beta_health")
    verify_graph_snapshots_match(snap_acme_before, snap_acme_after, "tenant_acme_corp")
    verify_graph_snapshots_match(snap_beta_before, snap_beta_after, "tenant_beta_health")
    print(
        f"✅ VERIFIED GRAPH INVARIANTS PRESERVED: tenant_acme_corp ({snap_acme_after['nodes']} nodes, "
        f"{snap_acme_after['edges']} edges, {len(snap_acme_after['edge_triples'])} relationship triples, "
        f"pre-existing ALERT-901 state {snap_acme_after['alert_states'].get('ALERT-901')} restored) & "
        f"tenant_beta_health ({snap_beta_after['nodes']} nodes, {snap_beta_after['edges']} edges) 100% intact."
    )


if __name__ == "__main__":
    run_suite()