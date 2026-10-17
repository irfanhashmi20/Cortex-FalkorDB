import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from falkordb import FalkorDB
from fastapi.testclient import TestClient
import graph_tools as gt
from agent_swarm import run_incident_investigation, _run_tasks
import main as main_app
from benchmark import (
    nearest_rank_percentile,
    BenchmarkArtifactTracker,
    get_graph_snapshot,
    restore_graph_snapshot,
    safe_cleanup_and_restore,
    prepare_and_verify_initial_incident_state,
    validate_investigation_state,
)


def test_percentile_calculation_20_elements():
    """Verifies nearest-rank P95 on 20 elements selects the 19th item (index 18), not Max (index 19)."""
    sample_20 = [float(i) for i in range(1, 21)]
    assert nearest_rank_percentile(sample_20, 0.50) == 10.0
    assert nearest_rank_percentile(sample_20, 0.95) == 19.0
    assert nearest_rank_percentile(sample_20, 0.95) < max(sample_20)


def test_unpooled_client_cleanup_and_error_reporting():
    """Verifies real FalkorDB/Redis client cleanup, retention of failed clients, and non-masking of primary errors."""
    client = FalkorDB(host=gt.FALKORDB_HOST, port=gt.FALKORDB_PORT)
    assert hasattr(client, "connection")
    assert hasattr(client.connection, "close")
    assert hasattr(client.connection, "connection_pool")
    assert hasattr(client.connection.connection_pool, "disconnect")
    gt.close_falkordb_client(client)

    with pytest.raises(AttributeError):
        gt.close_falkordb_client(object())  # type: ignore[arg-type]

    # 1. Partial failure test: valid client is removed, broken client is retained in _unpooled_clients
    policy = gt.InvestigationRuntimePolicy(use_pooling=False, use_cache=False)
    valid_client = FalkorDB(host=gt.FALKORDB_HOST, port=gt.FALKORDB_PORT)
    broken_client = object()
    policy.record_client_instantiation(valid_client, is_unpooled=True)
    policy.record_client_instantiation(broken_client, is_unpooled=True)  # type: ignore[arg-type]
    assert policy.unclosed_client_count == 2

    with pytest.raises(RuntimeError, match="Failed to close 1 unpooled FalkorDB client"):
        policy.close_unpooled_clients(raise_on_error=True)

    assert policy.unpooled_clients_closed == 1
    assert policy.unclosed_client_count == 1
    assert policy._unpooled_clients == [broken_client]
    assert len(policy.cleanup_errors) == 1

    # 2. Verify cleanup errors do not replace an active investigation exception
    snap_before = get_graph_snapshot("acme_corp")
    tracker = BenchmarkArtifactTracker()
    sid_err = tracker.new_session_id("err-mask")
    orig_find_rc = gt.find_root_cause
    orig_close = gt.close_falkordb_client
    try:
        def _failing_find_rc(*args, **kwargs):
            gt.get_graph("acme_corp")  # Instantiates an unpooled client
            raise ValueError("Primary investigation failure")

        def _failing_close(c):
            orig_close(c)
            raise OSError("Simulated socket close failure")

        gt.find_root_cause = _failing_find_rc
        gt.close_falkordb_client = _failing_close
        with pytest.raises(ValueError, match="Primary investigation failure"):
            run_incident_investigation(
                "ALERT-901", "acme_corp", parallel=False, use_pooling=False, session_id=sid_err
            )
    finally:
        gt.find_root_cause = orig_find_rc
        gt.close_falkordb_client = orig_close
        safe_cleanup_and_restore(tracker, snap_before)

    # 3. Full E2E unpooled vs pooled verification
    sid_cold = tracker.new_session_id("cold")
    mid_cold = tracker.new_memory_id("cold")
    sid_warm = tracker.new_session_id("warm")
    mid_warm = tracker.new_memory_id("warm")
    try:
        assert prepare_and_verify_initial_incident_state("acme_corp")
        state_cold = run_incident_investigation(
            "ALERT-901",
            "acme_corp",
            parallel=False,
            use_pooling=False,
            use_cache=False,
            session_id=sid_cold,
            custom_memory_id=mid_cold,
        )
        tracker.register_from_state(state_cold)
        assert validate_investigation_state(state_cold, sid_cold, mid_cold)
        assert state_cold["remediation_data"]["db_client_instantiations"] == 11
        assert state_cold["remediation_data"]["unpooled_clients_closed"] == 11
        assert state_cold["remediation_data"]["unclosed_client_count"] == 0
        assert state_cold["remediation_data"]["cleanup_errors"] == []
        assert state_cold["remediation_data"]["uncached_embeddings"] == 1

        assert prepare_and_verify_initial_incident_state("acme_corp")
        state_warm = run_incident_investigation(
            "ALERT-901",
            "acme_corp",
            parallel=True,
            use_pooling=True,
            use_cache=True,
            session_id=sid_warm,
            custom_memory_id=mid_warm,
        )
        tracker.register_from_state(state_warm)
        assert validate_investigation_state(state_warm, sid_warm, mid_warm)
        assert state_warm["remediation_data"]["db_client_instantiations"] == 0
        assert state_warm["remediation_data"]["unpooled_clients_closed"] == 0
        assert state_warm["remediation_data"]["unclosed_client_count"] == 0
        assert state_warm["remediation_data"]["cleanup_errors"] == []
        assert state_warm["remediation_data"]["uncached_embeddings"] == 0
    finally:
        safe_cleanup_and_restore(tracker, snap_before)
    assert get_graph_snapshot("acme_corp") == snap_before


def test_parallel_tasks_wait_for_all_futures_before_cleanup():
    """Proves _run_tasks waits for all sibling worker threads to finish before propagating an exception."""
    sibling_finished = threading.Event()
    sibling_completed_before_return = False

    def _fast_failing_task():
        raise ValueError("Immediate task failure")

    def _slow_sibling_task():
        nonlocal sibling_completed_before_return
        time.sleep(0.08)
        sibling_completed_before_return = True
        sibling_finished.set()
        return "ok"

    with pytest.raises(ValueError, match="Immediate task failure"):
        _run_tasks(True, [_fast_failing_task, _slow_sibling_task])

    assert sibling_finished.is_set()
    assert sibling_completed_before_return is True


def test_concurrent_policy_isolation_and_normal_api_unaffected():
    """
    Uses two isolated incident nodes (ALERT-CONC-1 and ALERT-CONC-2) so concurrent
    unpooled/uncached and pooled/cached investigations do not race on shared incident state.
    """
    snap_before = get_graph_snapshot("acme_corp")
    tracker = BenchmarkArtifactTracker()
    sid_unpooled = tracker.new_session_id("conc-unpooled")
    mid_unpooled = tracker.new_memory_id("conc-unpooled")
    sid_pooled = tracker.new_session_id("conc-pooled")
    mid_pooled = tracker.new_memory_id("conc-pooled")
    graph = gt.get_graph("acme_corp")

    try:
        assert prepare_and_verify_initial_incident_state("acme_corp")
        gt.get_db()

        graph.query(
            """
            MATCH (src:IncidentAlert {alert_id: 'ALERT-901'})-[:TRIGGERED_ON]->(chk:Microservice {name: 'CheckoutService'})
            MATCH (src)-[:CAUSED_BY]->(pr:PullRequest {pr_id: 'PR-402'})
            CREATE (a1:IncidentAlert {
              alert_id: 'ALERT-CONC-1',
              severity: src.severity,
              severity_rank: src.severity_rank,
              error_log: src.error_log,
              error_signature: src.error_signature,
              error_embedding: src.error_embedding,
              timestamp_epoch: src.timestamp_epoch,
              timestamp_iso: src.timestamp_iso,
              status: 'open',
              mttr_minutes: 0
            })-[:TRIGGERED_ON]->(chk)
            CREATE (a1)-[:CAUSED_BY]->(pr)
            CREATE (a2:IncidentAlert {
              alert_id: 'ALERT-CONC-2',
              severity: src.severity,
              severity_rank: src.severity_rank,
              error_log: src.error_log,
              error_signature: src.error_signature,
              error_embedding: src.error_embedding,
              timestamp_epoch: src.timestamp_epoch,
              timestamp_iso: src.timestamp_iso,
              status: 'open',
              mttr_minutes: 0
            })-[:TRIGGERED_ON]->(chk)
            CREATE (a2)-[:CAUSED_BY]->(pr)
            """
        )

        def _run_unpooled():
            return run_incident_investigation(
                "ALERT-CONC-1",
                "acme_corp",
                parallel=True,
                use_pooling=False,
                use_cache=False,
                session_id=sid_unpooled,
                custom_memory_id=mid_unpooled,
            )

        def _run_pooled():
            return run_incident_investigation(
                "ALERT-CONC-2",
                "acme_corp",
                parallel=True,
                use_pooling=True,
                use_cache=True,
                session_id=sid_pooled,
                custom_memory_id=mid_pooled,
            )

        def _run_normal_api():
            client = TestClient(main_app.app)
            resp = client.get("/api/graph?tenant_id=acme_corp")
            return resp.status_code, len(resp.json().get("nodes", []))

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_unpooled = ex.submit(_run_unpooled)
            f_pooled = ex.submit(_run_pooled)
            f_api = ex.submit(_run_normal_api)
            st_unpooled = f_unpooled.result()
            st_pooled = f_pooled.result()
            api_status, api_nodes = f_api.result()

        tracker.register_from_state(st_unpooled)
        tracker.register_from_state(st_pooled)

        assert validate_investigation_state(st_unpooled, sid_unpooled, mid_unpooled, alert_id="ALERT-CONC-1")
        assert validate_investigation_state(st_pooled, sid_pooled, mid_pooled, alert_id="ALERT-CONC-2")
        assert st_unpooled["remediation_data"]["db_client_instantiations"] == 11
        assert st_unpooled["remediation_data"]["unpooled_clients_closed"] == 11
        assert st_unpooled["remediation_data"]["unclosed_client_count"] == 0
        assert st_unpooled["remediation_data"]["uncached_embeddings"] == 1
        assert st_pooled["remediation_data"]["db_client_instantiations"] == 0
        assert st_pooled["remediation_data"]["unpooled_clients_closed"] == 0
        assert st_pooled["remediation_data"]["unclosed_client_count"] == 0
        assert st_pooled["remediation_data"]["uncached_embeddings"] == 0
        assert api_status == 200 and api_nodes >= 30
        assert gt._ACTIVE_POLICY.get() is None
    finally:
        try:
            graph.query("MATCH (a:IncidentAlert) WHERE a.alert_id IN ['ALERT-CONC-1', 'ALERT-CONC-2'] DETACH DELETE a")
        finally:
            safe_cleanup_and_restore(tracker, snap_before)
    assert get_graph_snapshot("acme_corp") == snap_before


def test_restoration_guaranteed_when_cleanup_raises():
    """Proves safe_cleanup_and_restore restores snapshot even if tracker.cleanup_all raises."""
    snap_before = get_graph_snapshot("acme_corp")
    prepare_and_verify_initial_incident_state("acme_corp")

    class FailingTracker(BenchmarkArtifactTracker):
        def cleanup_all(self, tenant_id: str = "acme_corp"):
            raise RuntimeError("Simulated cleanup failure")

    bad_tracker = FailingTracker()
    gt.get_graph("acme_corp").query(
        "MATCH (a:IncidentAlert {alert_id: 'ALERT-901'}) SET a.status = 'resolved'"
    )
    with pytest.raises(RuntimeError, match="Simulated cleanup failure"):
        safe_cleanup_and_restore(bad_tracker, snap_before)

    assert get_graph_snapshot("acme_corp") == snap_before


def test_query_1_root_cause():
    snap_before = get_graph_snapshot("acme_corp")
    try:
        prepare_and_verify_initial_incident_state("acme_corp")
        res = gt.find_root_cause("ALERT-901", "acme_corp")
        assert len(res["data"]) > 0
        assert res["data"][0]["suspect_service"] == "AuthTokenService"
        assert res["data"][0]["pr_id"] == "PR-402"
        assert res["data"][0]["risk_score"] == 4
    finally:
        restore_graph_snapshot(snap_before, "acme_corp")


def test_query_2_blast_radius():
    res = gt.get_blast_radius("AuthTokenService", "acme_corp")
    affected = [r["affected_service"] for r in res["data"]]
    assert len(affected) == 4
    assert "PaymentService" in affected
    assert "CheckoutService" in affected


def test_query_3_criticality():
    res = gt.get_critical_services("acme_corp")
    assert len(res["data"]) >= 4
    assert "service" in res["data"][0]


def test_query_4_shortest_path():
    res = gt.get_shortest_dependency_path("APIGateway", "AuthTokenService", "acme_corp")
    assert len(res["data"]) > 0
    assert res["data"][0]["hops"] == 3


def test_query_5_hybrid_memory():
    res = gt.recall_incident_memory(
        "CheckoutService HTTP 502 Bad Gateway: upstream PaymentService rejected by AuthTokenService (401 JWT signature mismatch)"
    )
    assert len(res["data"]) > 0
    assert res["data"][0]["memory_id"] == "MEM-301"
    assert len(res["data"][0]["ordered_runbook_steps"]) == 3


def test_query_6_store_resolution_memory_isolated():
    snap_before = get_graph_snapshot("acme_corp")
    tracker = BenchmarkArtifactTracker()
    mid = tracker.new_memory_id("testq6")
    try:
        assert prepare_and_verify_initial_incident_state("acme_corp")
        res = gt.store_resolution_memory(
            alert_id="ALERT-901",
            summary="Test rollback resolution",
            steps_taken="Step 1, Step 2",
            mttr_minutes=2,
            tenant_id="acme_corp",
            custom_memory_id=mid,
        )
        assert res["data"]["stored_memory_id"] == mid
        assert res["data"]["alert_status"] == "resolved"
    finally:
        safe_cleanup_and_restore(tracker, snap_before)
    assert get_graph_snapshot("acme_corp") == snap_before


def test_query_7_session_history_isolated():
    snap_before = get_graph_snapshot("acme_corp")
    tracker = BenchmarkArtifactTracker()
    sid = tracker.new_session_id("testq7")
    try:
        t1 = gt.record_session_turn(
            session_id=sid,
            agent_name="Triage & Code Analyst Agent",
            role="action",
            content="Isolated PR-402.",
            ref_id="PR-402",
            tenant_id="acme_corp",
        )
        t2 = gt.record_session_turn(
            session_id=sid,
            agent_name="Remediation Agent",
            role="action",
            content="Verified rollback completion.",
            ref_id="MEM-301",
            tenant_id="acme_corp",
        )
        tracker.turn_ids.add(t1["data"]["turn_id"])
        tracker.turn_ids.add(t2["data"]["turn_id"])

        hist_res = gt.get_session_history(sid, "acme_corp")
        assert len(hist_res["data"]) == 2
        entities = [row["referenced_entity"] for row in hist_res["data"]]
        assert "PR-402" in entities
        assert "MEM-301" in entities
    finally:
        safe_cleanup_and_restore(tracker, snap_before)
    assert get_graph_snapshot("acme_corp") == snap_before


def test_query_8_experts():
    res = gt.get_service_experts("CheckoutService", "acme_corp")
    assert len(res["data"]) > 0
    assert res["data"][0]["expert"] == "Aarav Sharma"


def test_query_9_provenance():
    res = gt.get_decision_provenance("CheckoutService", "acme_corp")
    assert len(res["data"]) > 0
    assert res["data"][0]["adr_id"] == "ADR-019"


def test_query_10_recent_neighborhood_changes():
    res = gt.get_recent_neighborhood_changes("CheckoutService", "acme_corp")
    assert len(res["data"]) >= 2
    services = [r["service"] for r in res["data"]]
    assert "CheckoutService" in services


def test_multigraph_tenant_isolation():
    acme_res = gt.get_critical_services("acme_corp")
    beta_res = gt.get_critical_services("beta_health")
    acme_names = {x["service"] for x in acme_res["data"]}
    beta_names = {x["service"] for x in beta_res["data"]}
    assert "CheckoutService" in acme_names
    assert "PatientPortal" in beta_names
    assert acme_names.isdisjoint(beta_names)


def test_non_destructive_cleanup_preserves_user_records_and_resolved_state():
    """
    Proves that if ALERT-901 is already resolved and user-created memory/session nodes exist,
    running the benchmark tracker and restoring snapshot preserves 100% of pre-existing state.
    """
    orig_snap = get_graph_snapshot("acme_corp")
    graph = gt.get_graph("acme_corp")
    user_mem_id = "MEM-USER-PROTECTED-999"
    user_sess_id = "sess-user-protected-999"
    user_turn_id = "TURN-USER-PROTECTED-999"

    try:
        graph.query(
            """
            MATCH (a:IncidentAlert {alert_id: 'ALERT-901'})-[:TRIGGERED_ON]->(s:Microservice {name: 'CheckoutService'})
            SET a.status = 'resolved', a.mttr_minutes = 5, s.status = 'healthy'
            MERGE (m:ResolutionMemory {memory_id: $mid})
            ON CREATE SET m.solution_summary = 'User created memory'
            MERGE (m)-[:RESOLVED]->(a)
            MERGE (sess:Session {session_id: $sid})
            MERGE (t:ConversationTurn {turn_id: $tid, session_id: $sid})-[:IN_SESSION]->(sess)
            """,
            {"mid": user_mem_id, "sid": user_sess_id, "tid": user_turn_id},
        )
        snap_with_user_data = get_graph_snapshot("acme_corp")
        assert snap_with_user_data["alert_states"]["ALERT-901"] == ("resolved", 5)
        assert user_mem_id in snap_with_user_data["memories"]
        assert user_turn_id in snap_with_user_data["turns"]

        tracker = BenchmarkArtifactTracker()
        sid = tracker.new_session_id("safety")
        mid = tracker.new_memory_id("safety")
        try:
            prepare_and_verify_initial_incident_state("acme_corp")
            state = run_incident_investigation(
                "ALERT-901", "acme_corp", parallel=True, session_id=sid, custom_memory_id=mid
            )
            tracker.register_from_state(state)
            assert validate_investigation_state(state, sid, mid)
        finally:
            safe_cleanup_and_restore(tracker, snap_with_user_data)

        snap_after = get_graph_snapshot("acme_corp")
        assert snap_after == snap_with_user_data
    finally:
        graph.query("MATCH (m:ResolutionMemory {memory_id: $mid}) DETACH DELETE m", {"mid": user_mem_id})
        graph.query("MATCH (t:ConversationTurn {turn_id: $tid}) DETACH DELETE t", {"tid": user_turn_id})
        graph.query("MATCH (s:Session {session_id: $sid}) DETACH DELETE s", {"sid": user_sess_id})
        restore_graph_snapshot(orig_snap, "acme_corp")


def test_sequential_and_parallel_modes_produce_identical_verified_results():
    """Verifies functional and graph-state equivalence between sequential and parallel modes."""
    snap_before = get_graph_snapshot("acme_corp")
    tracker = BenchmarkArtifactTracker()
    sid_seq = tracker.new_session_id("seq")
    mid_seq = tracker.new_memory_id("seq")
    sid_par = tracker.new_session_id("par")
    mid_par = tracker.new_memory_id("par")

    try:
        assert prepare_and_verify_initial_incident_state("acme_corp")
        state_seq = run_incident_investigation(
            "ALERT-901",
            "acme_corp",
            parallel=False,
            use_pooling=True,
            use_cache=True,
            session_id=sid_seq,
            custom_memory_id=mid_seq,
        )
        tracker.register_from_state(state_seq)
        assert validate_investigation_state(state_seq, sid_seq, mid_seq)
        tracker.cleanup_single_iteration(sid_seq, mid_seq, tenant_id="acme_corp")

        assert prepare_and_verify_initial_incident_state("acme_corp")
        state_par = run_incident_investigation(
            "ALERT-901",
            "acme_corp",
            parallel=True,
            use_pooling=True,
            use_cache=True,
            session_id=sid_par,
            custom_memory_id=mid_par,
        )
        tracker.register_from_state(state_par)
        assert validate_investigation_state(state_par, sid_par, mid_par)

        assert state_seq["triage_data"]["suspect_pr"] == state_par["triage_data"]["suspect_pr"] == "PR-402"
        assert state_seq["triage_data"]["blast_radius_count"] == state_par["triage_data"]["blast_radius_count"] == 4
        assert set(state_seq["triage_data"]["blast_radius_services"]) == set(state_par["triage_data"]["blast_radius_services"])
        assert state_seq["memory_data"]["memory_id"] == state_par["memory_data"]["memory_id"] == "MEM-301"
        assert state_seq["remediation_data"]["top_expert"] == state_par["remediation_data"]["top_expert"] == "Priya Nair"
        assert state_seq["query_count"] == state_par["query_count"] == 11
        assert state_seq["executed_agents"] == state_par["executed_agents"]
    finally:
        safe_cleanup_and_restore(tracker, snap_before)
    assert get_graph_snapshot("acme_corp") == snap_before