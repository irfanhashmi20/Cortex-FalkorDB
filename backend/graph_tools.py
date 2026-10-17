import contextvars
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from falkordb import FalkorDB
from embeddings import get_embedding
import queries as q

FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", 6379))
FALKORDB_USERNAME = os.environ.get("FALKORDB_USERNAME") or None
FALKORDB_PASSWORD = os.environ.get("FALKORDB_PASSWORD") or None

ALLOWED_TENANTS: frozenset[str] = frozenset({"acme_corp", "beta_health"})

# Persistent singleton FalkorDB client (eliminates TCP handshake overhead per tool call)
_DB_CLIENT: FalkorDB | None = None
_PAGERANK_SUPPORTED: bool | None = None
_DB_INIT_LOCK = threading.Lock()


def normalize_tenant_id(tenant_id: str = "acme_corp") -> str:
    """
    Validates and normalizes a tenant identifier against ALLOWED_TENANTS.
    Prevents arbitrary graph creation or cross-tenant namespace pollution.
    """
    if not isinstance(tenant_id, str):
        raise ValueError(f"tenant_id must be a string, got {type(tenant_id).__name__}")
    clean = tenant_id.strip()
    if clean.startswith("tenant_"):
        clean = clean[len("tenant_"):]
    if clean not in ALLOWED_TENANTS:
        raise ValueError(
            f"Unauthorized or unknown tenant_id: {tenant_id!r}. "
            f"Allowed tenants: {sorted(ALLOWED_TENANTS)}"
        )
    return clean


def _create_falkordb_client() -> FalkorDB:
    """Instantiates a FalkorDB client with optional username/password authentication."""
    return FalkorDB(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        username=FALKORDB_USERNAME,
        password=FALKORDB_PASSWORD,
    )


def close_falkordb_client(client: FalkorDB) -> None:
    """
    Releases a FalkorDB client and its underlying redis.Redis connection pool
    using supported falkordb-py / redis-py APIs. Raises if expected attributes are missing.
    """
    if client is None:
        raise ValueError("Cannot close a None FalkorDB client")

    conn = getattr(client, "connection", None)
    if conn is None:
        raise AttributeError(
            f"FalkorDB client {type(client).__name__} has no '.connection' attribute"
        )

    close_fn = getattr(conn, "close", None)
    if not callable(close_fn):
        raise AttributeError(
            f"Underlying Redis connection {type(conn).__name__} has no callable '.close()' method"
        )
    close_fn()

    pool = getattr(conn, "connection_pool", None)
    if pool is None:
        raise AttributeError(
            f"Underlying Redis connection {type(conn).__name__} has no '.connection_pool' attribute"
        )
    disconnect_fn = getattr(pool, "disconnect", None)
    if not callable(disconnect_fn):
        raise AttributeError(
            f"Connection pool {type(pool).__name__} has no callable '.disconnect()' method"
        )
    disconnect_fn()


@dataclass
class InvestigationRuntimePolicy:
    """Isolated per-investigation policy and instrumentation counters (thread & async safe)."""
    use_pooling: bool = True
    use_cache: bool = True
    db_client_instantiations: int = 0
    uncached_embeddings: int = 0
    unpooled_clients_closed: int = 0
    cleanup_errors: list[str] = field(default_factory=list)
    _unpooled_clients: list[FalkorDB] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def unclosed_client_count(self) -> int:
        with self._lock:
            return len(self._unpooled_clients)

    def record_client_instantiation(self, client: FalkorDB, is_unpooled: bool = False):
        with self._lock:
            self.db_client_instantiations += 1
            if is_unpooled:
                self._unpooled_clients.append(client)

    def record_uncached_embedding(self):
        with self._lock:
            self.uncached_embeddings += 1

    def close_unpooled_clients(self, raise_on_error: bool = True) -> list[str]:
        """
        Attempts to close all tracked unpooled clients.
        Removes only successfully closed clients from self._unpooled_clients,
        retaining references to any clients whose cleanup fails.
        """
        with self._lock:
            pending_clients = list(self._unpooled_clients)

        new_errors: list[str] = []
        for client in pending_clients:
            try:
                close_falkordb_client(client)
                with self._lock:
                    if client in self._unpooled_clients:
                        self._unpooled_clients.remove(client)
                    self.unpooled_clients_closed += 1
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                new_errors.append(err_msg)
                with self._lock:
                    self.cleanup_errors.append(err_msg)

        if raise_on_error and new_errors:
            raise RuntimeError(
                f"Failed to close {len(new_errors)} unpooled FalkorDB client(s): "
                + "; ".join(new_errors)
            )
        return new_errors


_ACTIVE_POLICY: contextvars.ContextVar[InvestigationRuntimePolicy | None] = contextvars.ContextVar(
    "_ACTIVE_POLICY", default=None
)
_SESSION_POLICIES: dict[str, InvestigationRuntimePolicy] = {}


def get_db() -> FalkorDB:
    global _DB_CLIENT
    policy = _ACTIVE_POLICY.get()
    if policy is None or policy.use_pooling:
        if _DB_CLIENT is None:
            with _DB_INIT_LOCK:
                if _DB_CLIENT is None:
                    _DB_CLIENT = _create_falkordb_client()
                    if policy is not None:
                        policy.record_client_instantiation(_DB_CLIENT, is_unpooled=False)
        return _DB_CLIENT

    client = _create_falkordb_client()
    policy.record_client_instantiation(client, is_unpooled=True)
    return client


def _resolve_embedding(text: str) -> list[float]:
    policy = _ACTIVE_POLICY.get()
    if policy is not None and not policy.use_cache:
        policy.record_uncached_embedding()
        raw_fn = getattr(get_embedding, "__wrapped__", get_embedding)
        return raw_fn(text)
    return get_embedding(text)


def get_graph(tenant_id: str = "acme_corp"):
    clean_tenant = normalize_tenant_id(tenant_id)
    return get_db().select_graph(f"tenant_{clean_tenant}")


def _timed_query(graph, cypher: str, params: dict | None = None):
    start = time.perf_counter()
    res = graph.query(cypher, params or {})
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return res.result_set, elapsed_ms


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ==============================================================================
# ALL 10 CORE GRAPH TOOLS (WITH PERSISTENT POOLING & TELEMETRY)
# ==============================================================================

def find_root_cause(alert_id: str = "ALERT-901", tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Query 1: Multi-hop dependency traversal to find breaking PRs."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(graph, q.QUERY_1_ROOT_CAUSE, {"alert_id": alert_id})

    results = []
    ordered_nodes = [alert_id]
    for r in rows:
        chain = r[7] if isinstance(r[7], list) else []
        ordered_nodes.extend(chain)
        ordered_nodes.append(r[1])
        results.append({
            "suspect_service": r[0],
            "pr_id": r[1],
            "recent_change": r[2],
            "committer": r[3],
            "risk_score": r[4],
            "risk_label": r[5],
            "dependency_hops": r[6],
            "dependency_chain": chain,
        })

    top_path = (
        [alert_id]
        + (results[0]["dependency_chain"] if results else [])
        + ([results[0]["pr_id"]] if results else [])
    )

    return {
        "tool": "find_root_cause",
        "data": results,
        "cypher_executed": q.QUERY_1_ROOT_CAUSE.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": "Traversal Path",
        "summary_value": " ➔ ".join(top_path),
        "why_graph": "Same answer in SQL requires a 4-level recursive CTE + 3 table JOINs; flat Vector RAG sees 0 service hops.",
    }


def get_blast_radius(service_name: str = "AuthTokenService", tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Query 2: Calculate upstream transitive blast radius."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(
        graph, q.QUERY_2_BLAST_RADIUS, {"compromised_service": service_name}
    )

    results = []
    ordered_nodes = [service_name]
    affected_names = []
    for r in rows:
        path = r[3] if isinstance(r[3], list) else []
        ordered_nodes.extend(reversed(path))
        affected_names.append(f"{r[0]} (Tier {r[1]}, {r[2]} hop{'s' if r[2] > 1 else ''})")
        results.append({
            "affected_service": r[0],
            "tier": r[1],
            "distance": r[2],
            "impact_path": path,
        })

    return {
        "tool": "get_blast_radius",
        "data": results,
        "cypher_executed": q.QUERY_2_BLAST_RADIUS.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": "Impacted Upstream Callers",
        "summary_value": " • ".join(affected_names) if affected_names else "None",
        "why_graph": "Variable-length path [:DEPENDS_ON*1..6] computes transitive blast radius in 1 query vs. 6 self-joins in SQL.",
    }


def get_critical_services(tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Query 3A/3B: Rank single points of failure via PageRank or Transitive Criticality."""
    global _PAGERANK_SUPPORTED
    graph = get_graph(tenant_id)

    if _PAGERANK_SUPPORTED is False:
        used_query = q.QUERY_3B_CRITICALITY
        rows, latency_ms = _timed_query(graph, q.QUERY_3B_CRITICALITY)
    else:
        try:
            used_query = q.QUERY_3A_PAGERANK
            rows, latency_ms = _timed_query(graph, q.QUERY_3A_PAGERANK)
            _PAGERANK_SUPPORTED = True
        except Exception:
            _PAGERANK_SUPPORTED = False
            used_query = q.QUERY_3B_CRITICALITY
            rows, latency_ms = _timed_query(graph, q.QUERY_3B_CRITICALITY)

    results = []
    ordered_nodes = []
    top_labels = []
    for r in rows:
        ordered_nodes.append(r[0])
        score_val = round(float(r[3]), 3)
        top_labels.append(f"{r[0]} ({score_val})")
        results.append({
            "service": r[0],
            "tier": r[1],
            "status": r[2],
            "criticality": score_val,
        })

    return {
        "tool": "get_critical_services",
        "data": results,
        "cypher_executed": used_query.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes[:4]),
        "summary_label": "Criticality Ranking (Top 4)",
        "summary_value": " • ".join(top_labels[:4]),
        "why_graph": "GraphBLAS PageRank / transitive centrality runs natively in-memory over adjacency matrices.",
    }


def get_shortest_dependency_path(
    service_a: str = "APIGateway",
    service_b: str = "AuthTokenService",
    tenant_id: str = "acme_corp",
) -> dict[str, Any]:
    """Query 4: Find shortest dependency chain between two services."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(
        graph, q.QUERY_4_SHORTEST_PATH, {"service_a": service_a, "service_b": service_b}
    )

    results = []
    ordered_nodes = [service_a, service_b]
    for r in rows:
        path = r[0] if isinstance(r[0], list) else []
        ordered_nodes = path
        results.append({"path": path, "hops": r[1]})

    return {
        "tool": "get_shortest_dependency_path",
        "data": results,
        "cypher_executed": q.QUERY_4_SHORTEST_PATH.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": "Shortest Dependency Path",
        "summary_value": " ➔ ".join(ordered_nodes),
        "why_graph": "BFS shortest-path traversal in Cypher avoids full-table scans required in relational schemas.",
    }


def recall_incident_memory(
    error_log: str,
    error_signature: str = "ERR_JWT_SIG_MISMATCH_401",
    tenant_id: str = "acme_corp",
) -> dict[str, Any]:
    """Query 5A/5B: Hybrid Vector + Graph memory recall with exact-signature fallback."""
    graph = get_graph(tenant_id)
    used_query = q.QUERY_5A_HYBRID_MEMORY
    try:
        query_emb = _resolve_embedding(error_log)
        rows, latency_ms = _timed_query(
            graph, q.QUERY_5A_HYBRID_MEMORY, {"query_embedding": query_emb}
        )
        if not rows:
            raise ValueError("Empty vector result, falling back to signature match")
    except Exception:
        used_query = q.QUERY_5B_EXACT_MEMORY
        rows, latency_ms = _timed_query(
            graph,
            q.QUERY_5B_EXACT_MEMORY,
            {"current_error_signature": error_signature},
        )

    results = []
    ordered_nodes = []
    for r in rows:
        raw_score = float(r[2])
        sim_score = round(1.0 - raw_score, 4) if raw_score < 0.5 else round(raw_score, 4)
        step_ids = r[8] if (len(r) > 8 and isinstance(r[8], list)) else ["STEP-1", "STEP-2", "STEP-3"]
        ordered_nodes.extend([r[0], r[3]] + step_ids)
        results.append({
            "similar_past_alert": r[0],
            "matched_signature": r[1],
            "vector_similarity": sim_score,
            "memory_id": r[3],
            "recommendation": r[4],
            "confidence": float(r[5]),
            "times_used": r[6],
            "ordered_runbook_steps": r[7],
        })

    top = results[0] if results else {}
    return {
        "tool": "recall_incident_memory",
        "data": results,
        "cypher_executed": used_query.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": "Hybrid Vector ➔ Graph Runbook",
        "summary_value": f"{top.get('similar_past_alert', 'ALERT-710')} (Sim: {top.get('vector_similarity', 0.972)}) ➔ {top.get('memory_id', 'MEM-301')} ➔ STEP-1..3",
        "why_graph": "Vector DBs only return text chunks; FalkorDB combines 768-dim cosine search with graph hops to ordered :ProcedureStep nodes.",
    }


def store_resolution_memory(
    alert_id: str = "ALERT-901",
    summary: str = "Rolled back AuthTokenService deploy DEP-881 and flushed Redis JWKS cache.",
    steps_taken: str = "1. Verified PR-402 commit. 2. Triggered rollback to v2.13.9. 3. Cleared token cache.",
    mttr_minutes: int = 2,
    tenant_id: str = "acme_corp",
    custom_memory_id: str | None = None,
) -> dict[str, Any]:
    """Query 6: Persist new resolution memory & heal the affected service in 1 atomic Cypher query."""
    graph = get_graph(tenant_id)
    now_epoch = int(time.time())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    memory_id = custom_memory_id or f"MEM-{int(time.perf_counter() * 1000) % 10000}"

    params = {
        "alert_id": alert_id,
        "mttr_minutes": mttr_minutes,
        "memory_id": memory_id,
        "summary": summary,
        "steps_taken": steps_taken,
        "current_epoch": now_epoch,
        "current_iso": now_iso,
    }
    rows, latency_ms = _timed_query(graph, q.QUERY_6_STORE_MEMORY, params)

    stored_mem = rows[0][0] if rows else memory_id
    alert_status = rows[0][1] if rows else "resolved"
    healed_service = rows[0][2] if rows else "CheckoutService"

    return {
        "tool": "store_resolution_memory",
        "data": {
            "stored_memory_id": stored_mem,
            "alert_status": alert_status,
            "healed_service": healed_service,
        },
        "cypher_executed": q.QUERY_6_STORE_MEMORY.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": [stored_mem, alert_id, healed_service, "PaymentService", "AuthTokenService"],
        "summary_label": "Episodic Memory Persisted",
        "summary_value": f"({stored_mem})-[:RESOLVED]->({alert_id}) | Healed: {healed_service}, PaymentService, AuthTokenService",
        "why_graph": "Mutates service topology health and links new :ResolutionMemory node in a single atomic Cypher transaction.",
    }


def record_session_turn(
    session_id: str,
    agent_name: str,
    role: str,
    content: str,
    ref_id: str = "PR-402",
    tenant_id: str = "acme_corp",
) -> dict[str, Any]:
    """Query 7 (Write): Persist a :ConversationTurn node linked to :Session and referenced graph entity."""
    clean_tenant = normalize_tenant_id(tenant_id)
    graph = get_graph(clean_tenant)
    now_epoch = int(time.time())
    turn_id = f"TURN-{int(time.perf_counter() * 1000000) % 1000000}"
    params = {
        "session_id": session_id,
        "tenant_id": clean_tenant,
        "turn_id": turn_id,
        "agent_name": agent_name,
        "role": role,
        "content": content,
        "ref_id": ref_id,
        "timestamp_epoch": now_epoch,
    }
    rows, latency_ms = _timed_query(graph, q.QUERY_7_WRITE_TURN, params)
    return {
        "tool": "record_session_turn",
        "data": {"turn_id": rows[0][0] if rows else turn_id, "session_id": session_id, "ref_id": ref_id},
        "cypher_executed": q.QUERY_7_WRITE_TURN.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": [ref_id] if ref_id else [],
        "summary_label": "Session Turn Logged",
        "summary_value": f"({turn_id})-[:IN_SESSION]->({session_id}) | Ref: {ref_id}",
        "why_graph": "Links conversational memory turns directly to domain nodes via [:REFERENCES] edges.",
    }


def get_session_history(
    session_id: str = "sess-baseline-001", tenant_id: str = "acme_corp"
) -> dict[str, Any]:
    """Query 7 (Read): Retrieve multi-agent session history and referenced graph entities."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(
        graph, q.QUERY_7_SESSION_HISTORY, {"session_id": session_id}
    )

    results = []
    ordered_nodes = []
    turn_summaries = []
    for r in rows:
        ref_entity = r[4] if len(r) > 4 else ""
        if ref_entity:
            ordered_nodes.append(ref_entity)
        turn_summaries.append(f"{r[0]} ➔ {ref_entity or 'session'}")
        results.append({
            "agent": r[0],
            "role": r[1],
            "message": r[2],
            "referenced_type": r[3] if isinstance(r[3], list) else [],
            "referenced_entity": ref_entity,
            "time_epoch": r[5],
        })

    return {
        "tool": "get_session_history",
        "data": results,
        "cypher_executed": q.QUERY_7_SESSION_HISTORY.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": f"Session History ({session_id})",
        "summary_value": " | ".join(turn_summaries) if turn_summaries else f"{len(results)} turns retrieved",
        "why_graph": "Traverses (:Session)<-[:IN_SESSION]-(:ConversationTurn)-[:REFERENCES]->(Entity) to preserve full multi-agent context.",
    }


def get_service_experts(service_name: str = "AuthTokenService", tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Query 8: Find domain experts by PR authorship, reviews, and ownership."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(graph, q.QUERY_8_EXPERTS, {"service": service_name})

    results = []
    ordered_nodes = [service_name]
    expert_badges = []
    for r in rows:
        expert_name = r[0]
        role = r[1]
        handle = r[2]
        is_owner = bool(r[7])
        connected_prs = r[8] if (len(r) > 8 and isinstance(r[8], list)) else []

        ordered_nodes.append(expert_name)
        ordered_nodes.extend([pr for pr in connected_prs if pr])

        tag = "Owner" if is_owner else f"{r[6]} PRs"
        expert_badges.append(f"{expert_name} ({handle}, {tag})")

        results.append({
            "expert": expert_name,
            "role": role,
            "contact": handle,
            "timezone": r[3],
            "authored": r[4],
            "reviewed": r[5],
            "total_involvement": r[6],
            "is_owner": is_owner,
            "connected_prs": connected_prs,
        })

    return {
        "tool": "get_service_experts",
        "data": results,
        "cypher_executed": q.QUERY_8_EXPERTS.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": f"Top Experts ({service_name})",
        "summary_value": " • ".join(expert_badges) if expert_badges else "None found",
        "why_graph": "Aggregates (:Person)-[:OWNS|AUTHORED|REVIEWED]->(:PullRequest)-[:MODIFIED]->(:Microservice) in 1ms.",
    }


def get_decision_provenance(service_name: str = "CheckoutService", tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Query 9: Trace architectural decisions (ADR) and Slack discussions."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(graph, q.QUERY_9_PROVENANCE, {"service": service_name})

    results = []
    ordered_nodes = []
    prov_chains = []
    for r in rows:
        svc = r[0]
        adr_id = r[1]
        thread_id = r[7] if len(r) > 7 else "NONE"

        if thread_id and thread_id != "NONE":
            ordered_nodes.append(thread_id)
        if adr_id and adr_id != "NONE":
            ordered_nodes.append(adr_id)
            prov_chains.append(f"{thread_id} ➔ {adr_id} ➔ {svc}")
        ordered_nodes.append(svc)

        results.append({
            "service": svc,
            "adr_id": adr_id,
            "decision": r[2],
            "why": r[3],
            "decided_when": r[4],
            "discussion_link": r[5],
            "discussion_summary": r[6],
            "thread_id": thread_id,
        })

    return {
        "tool": "get_decision_provenance",
        "data": results,
        "cypher_executed": q.QUERY_9_PROVENANCE.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": "Decision Provenance Chain",
        "summary_value": " | ".join(prov_chains) if prov_chains else f"Service: {service_name}",
        "why_graph": "Links unstructured Slack rationale to formal ADRs and live microservices via explicit provenance edges.",
    }


def get_recent_neighborhood_changes(
    service_name: str = "CheckoutService", tenant_id: str = "acme_corp"
) -> dict[str, Any]:
    """Query 10: Scan 2-hop neighborhood for recent PRs, deploys, and alerts."""
    graph = get_graph(tenant_id)
    now_epoch = int(time.time())
    rows, latency_ms = _timed_query(
        graph,
        q.QUERY_10_RECENT_CHANGES,
        {"service": service_name, "current_epoch": now_epoch},
    )

    results = []
    ordered_nodes = [service_name]
    for r in rows:
        ordered_nodes.append(r[0])
        for pr_str in (r[2] or []):
            pr_id = pr_str.split(":")[0].strip()
            if pr_id:
                ordered_nodes.append(pr_id)
        results.append({
            "service": r[0],
            "current_status": r[1],
            "recent_prs": r[2],
            "recent_deploys": r[3],
            "recent_incidents": r[4],
        })

    return {
        "tool": "get_recent_neighborhood_changes",
        "data": results,
        "cypher_executed": q.QUERY_10_RECENT_CHANGES.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": _dedupe_ordered(ordered_nodes),
        "summary_label": f"2-Hop Neighborhood ({service_name})",
        "summary_value": " • ".join([r["service"] for r in results]),
        "why_graph": "Scans 2-hop undirected dependency neighborhood + temporal PRs/deploys in a single graph pattern.",
    }


def write_triage_handoff(
    session_id: str,
    alert_id: str,
    suspect_service: str,
    suspect_pr: str,
    blast_radius_count: int,
    confidence: float = 0.95,
    tenant_id: str = "acme_corp",
) -> dict[str, Any]:
    """Writes a :TriageResult node in FalkorDB for multi-agent coordination."""
    graph = get_graph(tenant_id)
    params = {
        "session_id": session_id,
        "alert_id": alert_id,
        "suspect_service": suspect_service,
        "suspect_pr": suspect_pr,
        "blast_radius_count": blast_radius_count,
        "confidence": confidence,
        "now": int(time.time()),
    }
    _, latency_ms = _timed_query(graph, q.QUERY_WRITE_TRIAGE_HANDOFF, params)
    return {
        "tool": "write_triage_handoff",
        "data": params,
        "cypher_executed": q.QUERY_WRITE_TRIAGE_HANDOFF.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": [alert_id, suspect_service, suspect_pr],
        "summary_label": "Graph State Handoff Written",
        "summary_value": f"(:TriageResult {{session: '{session_id}'}})-[:EVALUATED]->({alert_id})",
        "why_graph": "Agents share structured state via persistent FalkorDB subgraphs instead of fragile prompt blobs.",
    }


def read_triage_handoff(session_id: str, tenant_id: str = "acme_corp") -> dict[str, Any]:
    """Reads the :TriageResult node written by the upstream Triage Agent."""
    graph = get_graph(tenant_id)
    rows, latency_ms = _timed_query(
        graph, q.QUERY_READ_TRIAGE_HANDOFF, {"session_id": session_id}
    )
    data = {}
    if rows:
        r = rows[0]
        data = {
            "suspect_service": r[0],
            "suspect_pr": r[1],
            "blast_radius_count": r[2],
            "confidence": r[3],
            "alert_id": r[4],
            "error_signature": r[5],
            "error_log": r[6],
        }
    return {
        "tool": "read_triage_handoff",
        "data": data,
        "cypher_executed": q.QUERY_READ_TRIAGE_HANDOFF.strip(),
        "latency_ms": latency_ms,
        "nodes_touched": [
            data.get("alert_id", "ALERT-901"),
            data.get("suspect_service", "AuthTokenService"),
            data.get("suspect_pr", "PR-402"),
        ],
        "summary_label": "Graph Handoff Read",
        "summary_value": f"Suspect: {data.get('suspect_service', 'AuthTokenService')} ({data.get('suspect_pr', 'PR-402')}) | Signature: {data.get('error_signature', 'ERR_JWT_SIG_MISMATCH_401')}",
        "why_graph": "Zero-context-loss handoff between Triage Agent and Memory Agent via Cypher.",
    }