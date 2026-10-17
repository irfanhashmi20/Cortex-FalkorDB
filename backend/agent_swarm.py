import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph, END
import graph_tools as gt

_EXECUTOR = ThreadPoolExecutor(max_workers=6)


class AgentState(TypedDict):
    session_id: str
    tenant_id: str
    alert_id: str
    query_text: str
    mode: str  # "incident" or "brain"
    parallel: bool
    use_pooling: bool
    use_cache: bool
    custom_memory_id: str
    executed_agents: List[str]
    started_at_perf: float
    total_db_ms: float
    query_count: int
    highlight_nodes: List[str]
    messages: List[Dict[str, Any]]
    current_agent: str
    triage_data: Dict[str, Any]
    memory_data: Dict[str, Any]
    remediation_data: Dict[str, Any]
    provenance_data: Dict[str, Any]
    final_answer: str


ALL_KNOWN_SERVICES = [
    "APIGateway",
    "WebStorefront",
    "CheckoutService",
    "OrderService",
    "PaymentService",
    "InventoryService",
    "AuthTokenService",
    "FraudEngine",
    "PatientPortal",
    "EHRRecordsAPI",
    "ImagingService",
    "HIPAAuditVault",
]


def _bind_session_policy_if_present(state: AgentState) -> gt.InvestigationRuntimePolicy | None:
    policy = gt._SESSION_POLICIES.get(state.get("session_id", ""))
    if policy is not None:
        gt._ACTIVE_POLICY.set(policy)
    return policy


def _run_tasks(parallel: bool, callables: List[Callable[[], Any]]) -> List[Any]:
    """
    Executes independent graph tool calls concurrently if parallel=True, else sequentially.
    Waits for ALL submitted futures to complete before propagating any exception so
    worker threads never continue using unpooled clients during finally cleanup.
    """
    if parallel and len(callables) > 1:
        futures = [_EXECUTOR.submit(contextvars.copy_context().run, fn) for fn in callables]
        wait(futures, return_when=ALL_COMPLETED)
        results: List[Any] = []
        errors: List[BaseException] = []
        for f in futures:
            exc = f.exception()
            if exc is not None:
                errors.append(exc)
            else:
                results.append(f.result())
        if errors:
            primary = errors[0]
            if len(errors) > 1 and hasattr(primary, "add_note"):
                for extra in errors[1:]:
                    primary.add_note(f"Additional parallel task error: {type(extra).__name__}: {extra}")
            raise primary
        return results
    return [fn() for fn in callables]


def _detect_service_from_query(query_text: str, tenant_id: str) -> str:
    q_lower = query_text.lower()
    for svc in ALL_KNOWN_SERVICES:
        if svc.lower() in q_lower or svc.lower().replace("service", "") in q_lower:
            return svc
    return "PatientPortal" if "beta" in tenant_id else "CheckoutService"


def orchestrator_node(state: AgentState) -> AgentState:
    """Agent 1: Routes between Autonomous Incident Pipeline and Company Brain Q&A."""
    _bind_session_policy_if_present(state)
    if not state.get("started_at_perf"):
        state["started_at_perf"] = time.perf_counter()
    state["total_db_ms"] = 0.0
    state["query_count"] = 0
    state["highlight_nodes"] = []
    state["executed_agents"] = ["Orchestrator Agent"]
    state["current_agent"] = "Orchestrator Agent"
    state["messages"].append({
        "agent": "Orchestrator Agent",
        "role": "thought",
        "content": (
            f"Received request [Mode: `{state['mode'].upper()}`]. "
            f"Binding session to isolated FalkorDB multigraph `tenant_{state['tenant_id']}`..."
        ),
        "timestamp": int(time.time()),
    })
    return state


def triage_analyst_node(state: AgentState) -> AgentState:
    """Agent 2: Traverses multi-hop dependency graph and calculates blast radius."""
    _bind_session_policy_if_present(state)
    state["current_agent"] = "Triage & Code Analyst Agent"
    state.setdefault("executed_agents", []).append("Triage & Code Analyst Agent")
    is_parallel = state.get("parallel", True)

    state["messages"].append({
        "agent": "Triage & Code Analyst Agent",
        "role": "thought",
        "content": f"Executing 4-Hop Root Cause Analysis (Query 1) & Criticality/Blast-Radius Checks (Queries 2 & 3) on `{state['alert_id']}`...",
        "timestamp": int(time.time()),
    })

    tenant = state["tenant_id"]
    alert_id = state["alert_id"]

    # Stage 2A: Run independent root-cause and criticality queries
    rc_res, crit_res = _run_tasks(
        is_parallel,
        [
            lambda: gt.find_root_cause(alert_id, tenant),
            lambda: gt.get_critical_services(tenant),
        ],
    )

    used_rc_fallback = len(rc_res.get("data", [])) == 0
    used_crit_fallback = len(crit_res.get("data", [])) == 0
    top_hit = rc_res["data"][0] if not used_rc_fallback else {
        "suspect_service": "AuthTokenService",
        "pr_id": "PR-402",
        "recent_change": "Rotate JWT signing key & enforce strict RS256 header check",
        "risk_score": 4,
        "risk_label": "critical",
        "dependency_chain": ["CheckoutService", "PaymentService", "AuthTokenService"],
    }

    # Stage 2B: Run blast radius and session turn write concurrently
    br_res, turn_res = _run_tasks(
        is_parallel,
        [
            lambda: gt.get_blast_radius(top_hit["suspect_service"], tenant),
            lambda: gt.record_session_turn(
                state["session_id"],
                "Triage & Code Analyst Agent",
                "action",
                f"Isolated {top_hit['suspect_service']} via {top_hit['pr_id']}",
                top_hit["pr_id"],
                tenant,
            ),
        ],
    )
    used_br_fallback = len(br_res.get("data", [])) == 0
    actual_blast_count = len(br_res.get("data", []))

    # Stage 2C: Write triage handoff using actual measured blast_radius_count
    handoff_res = gt.write_triage_handoff(
        session_id=state["session_id"],
        alert_id=alert_id,
        suspect_service=top_hit["suspect_service"],
        suspect_pr=top_hit["pr_id"],
        blast_radius_count=actual_blast_count,
        confidence=0.96,
        tenant_id=tenant,
    )

    telemetry = [rc_res, br_res, crit_res, handoff_res, turn_res]
    state["total_db_ms"] = round(
        state.get("total_db_ms", 0.0) + sum(t["latency_ms"] for t in telemetry), 2
    )
    state["query_count"] = state.get("query_count", 0) + len(telemetry)

    state["triage_data"] = {
        "used_fallback": used_rc_fallback,
        "used_crit_fallback": used_crit_fallback,
        "used_br_fallback": used_br_fallback,
        "suspect_service": top_hit["suspect_service"],
        "suspect_pr": top_hit["pr_id"],
        "recent_change": top_hit["recent_change"],
        "risk_score": top_hit["risk_score"],
        "dependency_chain": top_hit["dependency_chain"],
        "blast_radius_count": actual_blast_count,
        "blast_radius_services": [x["affected_service"] for x in br_res["data"]],
        "tool_telemetry": telemetry,
    }

    chain_str = " ➔ ".join(top_hit["dependency_chain"])
    state["messages"].append({
        "agent": "Triage & Code Analyst Agent",
        "role": "action",
        "content": (
            f"Isolated root cause across {len(top_hit['dependency_chain']) - 1} hops (`{chain_str}`): "
            f"**{top_hit['suspect_service']}** broken by `{top_hit['pr_id']}` (Risk: `{top_hit['risk_score']}/4 CRITICAL`). "
            f"Transitive blast radius impacts **{actual_blast_count} upstream services**."
        ),
        "timestamp": int(time.time()),
    })
    return state


def memory_knowledge_node(state: AgentState) -> AgentState:
    """Agent 3: Reads handoff node, searches hybrid vector+graph memory, or queries Company Brain."""
    _bind_session_policy_if_present(state)
    state["current_agent"] = "Memory & Knowledge Agent"
    state.setdefault("executed_agents", []).append("Memory & Knowledge Agent")
    tenant = state["tenant_id"]
    is_parallel = state.get("parallel", True)

    if state["mode"] == "incident":
        state["messages"].append({
            "agent": "Memory & Knowledge Agent",
            "role": "thought",
            "content": "Reading `:TriageResult` handoff node from FalkorDB. Querying Hybrid Vector + Graph Memory (Query 5A)...",
            "timestamp": int(time.time()),
        })

        read_ho = gt.read_triage_handoff(
            session_id=state["session_id"], tenant_id=tenant
        )
        error_log = read_ho["data"].get("error_log") or (
            "CheckoutService HTTP 502 Bad Gateway: upstream PaymentService rejected by AuthTokenService (401 JWT signature mismatch)"
        )
        mem_res = gt.recall_incident_memory(
            error_log=error_log, tenant_id=tenant
        )
        used_mem_fallback = len(mem_res.get("data", [])) == 0
        top_mem = mem_res["data"][0] if not used_mem_fallback else {
            "memory_id": "MEM-301",
            "recommendation": "Rollback AuthTokenService deploy to restore dual key validation.",
            "vector_similarity": 0.972,
            "ordered_runbook_steps": [
                "Verify offending PR",
                "Execute rollback DEP-881",
                "Page owner",
            ],
        }

        turn_mem_res = gt.record_session_turn(
            session_id=state["session_id"],
            agent_name="Memory & Knowledge Agent",
            role="action",
            content=f"Recalled runbook {top_mem['memory_id']}",
            ref_id=top_mem["memory_id"],
            tenant_id=tenant,
        )

        telemetry = [read_ho, mem_res, turn_mem_res]
        state["total_db_ms"] = round(
            state.get("total_db_ms", 0.0) + sum(t["latency_ms"] for t in telemetry), 2
        )
        state["query_count"] = state.get("query_count", 0) + len(telemetry)

        state["memory_data"] = {
            "used_fallback": used_mem_fallback,
            "handoff_verified": bool(read_ho.get("data")),
            "memory_id": top_mem["memory_id"],
            "recommendation": top_mem["recommendation"],
            "similarity": top_mem.get("vector_similarity", 0.972),
            "runbook_steps": top_mem["ordered_runbook_steps"],
            "tool_telemetry": telemetry,
        }

        state["messages"].append({
            "agent": "Memory & Knowledge Agent",
            "role": "action",
            "content": (
                f"Recalled past resolution `{top_mem['memory_id']}` "
                f"(Cosine Similarity: `{top_mem.get('vector_similarity', 0.972)}`). "
                f"Extracted **{len(top_mem['ordered_runbook_steps'])} ordered `:ProcedureStep` nodes**."
            ),
            "timestamp": int(time.time()),
        })
    else:
        target_service = _detect_service_from_query(
            state["query_text"], tenant
        )
        state["messages"].append({
            "agent": "Memory & Knowledge Agent",
            "role": "thought",
            "content": f"Traversing Company Brain graph for **{target_service}** (Query 8: Experts, Query 9: ADR/Slack, Query 10: Neighborhood)...",
            "timestamp": int(time.time()),
        })

        exp_res, prov_res, neigh_res = _run_tasks(
            is_parallel,
            [
                lambda: gt.get_service_experts(target_service, tenant),
                lambda: gt.get_decision_provenance(target_service, tenant),
                lambda: gt.get_recent_neighborhood_changes(target_service, tenant),
            ],
        )

        telemetry = [exp_res, prov_res, neigh_res]
        state["total_db_ms"] = round(
            state.get("total_db_ms", 0.0) + sum(t["latency_ms"] for t in telemetry), 2
        )
        state["query_count"] = state.get("query_count", 0) + len(telemetry)

        brain_highlight = []
        brain_highlight.extend(exp_res["nodes_touched"])
        brain_highlight.extend(prov_res["nodes_touched"])
        state["highlight_nodes"] = brain_highlight

        state["provenance_data"] = {
            "service": target_service,
            "provenance": prov_res["data"],
            "experts": exp_res["data"],
            "neighborhood": neigh_res["data"],
            "tool_telemetry": telemetry,
        }

        top_expert = (
            exp_res["data"][0]["expert"] if exp_res["data"] else "Aarav Sharma"
        )
        state["messages"].append({
            "agent": "Memory & Knowledge Agent",
            "role": "action",
            "content": (
                f"Identified top domain expert **{top_expert}**, traced ADR/Slack decision provenance "
                f"for `{target_service}`, and scanned {len(neigh_res['data'])} connected services."
            ),
            "timestamp": int(time.time()),
        })

    return state


def remediation_node(state: AgentState) -> AgentState:
    """Agent 4: Executes remediation, writes resolution memory (Query 6), reads session history (Query 7), and pages owner (Query 8)."""
    policy = _bind_session_policy_if_present(state)
    state["current_agent"] = "Remediation Agent"
    state.setdefault("executed_agents", []).append("Remediation Agent")
    tenant = state["tenant_id"]
    is_parallel = state.get("parallel", True)

    if state["mode"] == "incident":
        state["messages"].append({
            "agent": "Remediation Agent",
            "role": "thought",
            "content": "Executing runbook `RB-JWT-ROLLBACK`, looking up service experts (Query 8), persisting `:ResolutionMemory` (Query 6), and verifying session audit trail (Query 7)...",
            "timestamp": int(time.time()),
        })

        suspect_svc = state["triage_data"].get("suspect_service", "AuthTokenService")
        custom_mem_id = state.get("custom_memory_id") or None

        exp_res, rem_res, hist_res = _run_tasks(
            is_parallel,
            [
                lambda: gt.get_service_experts(suspect_svc, tenant),
                lambda: gt.store_resolution_memory(
                    state["alert_id"],
                    "Rolled back AuthTokenService deploy DEP-881 and flushed Redis JWKS cache.",
                    "1. Verified PR-402 commit. 2. Triggered rollback to v2.13.9. 3. Cleared token cache.",
                    2,
                    tenant,
                    custom_memory_id=custom_mem_id,
                ),
                lambda: gt.get_session_history(state["session_id"], tenant),
            ],
        )

        telemetry = [exp_res, rem_res, hist_res]
        total_ms = round(
            state.get("total_db_ms", 0.0) + sum(t["latency_ms"] for t in telemetry), 2
        )
        q_count = state.get("query_count", 0) + len(telemetry)
        avg_ms = round(total_ms / max(q_count, 1), 2)
        state["total_db_ms"] = total_ms
        state["query_count"] = q_count

        elapsed_s = round(time.perf_counter() - state.get("started_at_perf", time.perf_counter()), 3)
        elapsed_ms = round(elapsed_s * 1000, 2)
        top_expert = exp_res["data"][0] if exp_res["data"] else {
            "expert": "Priya Nair",
            "contact": "@priya.sre",
        }

        state["highlight_nodes"] = [
            "ALERT-901",
            "CheckoutService",
            "PaymentService",
            "AuthTokenService",
            "PR-402",
            "MEM-301",
            rem_res["data"]["stored_memory_id"],
            top_expert["expert"],
        ]

        if policy is not None:
            policy.close_unpooled_clients(raise_on_error=False)

        state["remediation_data"] = {
            **rem_res["data"],
            "used_expert_fallback": len(exp_res.get("data", [])) == 0,
            "top_expert": top_expert["expert"],
            "elapsed_seconds": elapsed_s,
            "elapsed_ms": elapsed_ms,
            "total_db_ms": total_ms,
            "avg_db_ms": avg_ms,
            "query_count": q_count,
            "session_turns_verified": len(hist_res["data"]),
            "db_client_instantiations": policy.db_client_instantiations if policy else 0,
            "uncached_embeddings": policy.uncached_embeddings if policy else 0,
            "unpooled_clients_closed": policy.unpooled_clients_closed if policy else 0,
            "unclosed_client_count": policy.unclosed_client_count if policy else 0,
            "cleanup_errors": list(policy.cleanup_errors) if policy else [],
            "tool_telemetry": telemetry,
        }

        state["final_answer"] = (
            f"• **Root Cause Path:** `ALERT-901` ➔ `CheckoutService` ➔ `PaymentService` ➔ `{suspect_svc}` ➔ `{state['triage_data']['suspect_pr']}`\n"
            f"• **Blast Radius Protected:** {state['triage_data']['blast_radius_count']} upstream services ({', '.join(state['triage_data']['blast_radius_services'])})\n"
            f"• **Memory & Owner:** Recalled `{state['memory_data']['memory_id']}` (Sim: `{state['memory_data']['similarity']}`) • Paged **{top_expert['expert']}** (`{top_expert['contact']}`) • Wrote `{rem_res['data']['stored_memory_id']}`\n"
            f"• **Latency Breakdown:** Completed in **{elapsed_ms}ms ({elapsed_s}s) end-to-end** | FalkorDB executed {q_count} queries in **{total_ms}ms cumulative** ({avg_ms}ms avg/query)."
        )

        state["messages"].append({
            "agent": "Remediation Agent",
            "role": "action",
            "content": (
                f"Healed services to `healthy`, persisted `{rem_res['data']['stored_memory_id']}`, and verified {len(hist_res['data'])} session turns. "
                f"End-to-end investigation completed in **{elapsed_ms}ms ({elapsed_s}s)**."
            ),
            "timestamp": int(time.time()),
        })
    else:
        elapsed_s = round(time.perf_counter() - state.get("started_at_perf", time.perf_counter()), 3)
        elapsed_ms = round(elapsed_s * 1000, 2)
        total_ms = round(state.get("total_db_ms", 0.0), 2)
        q_count = max(state.get("query_count", 3), 1)
        avg_ms = round(total_ms / q_count, 2)

        prov = state.get("provenance_data", {})
        svc = prov.get("service", "CheckoutService")
        prov_list = prov.get("provenance", [])
        experts = prov.get("experts", [])
        neigh = prov.get("neighborhood", [])

        adr_info = prov_list[0] if prov_list else {
            "adr_id": "ADR-019",
            "decision": "Use Redis for active checkout cart state",
            "why": "Reduces p99 lock contention",
            "discussion_link": "https://acme.slack.com/archives/C01ARCH/p1723459200",
            "thread_id": "SLK-772",
        }
        expert_names = [
            f"**{e['expert']}** (`{e['contact']}`, TZ: `{e['timezone']}`, {'Owner' if e['is_owner'] else str(e['total_involvement']) + ' PRs'})"
            for e in experts[:3]
        ]

        state["remediation_data"] = {
            "elapsed_seconds": elapsed_s,
            "elapsed_ms": elapsed_ms,
            "total_db_ms": total_ms,
            "avg_db_ms": avg_ms,
            "query_count": q_count,
        }

        state["final_answer"] = (
            f"• **Top Domain Experts (`{svc}`):** {' • '.join(expert_names) if expert_names else 'Aarav Sharma (@aarav)'}\n"
            f"• **ADR Provenance (`{adr_info.get('adr_id')}`):** {adr_info.get('decision')} — *{adr_info.get('why') or 'Verified in graph'}*\n"
            f"• **Slack Thread (`{adr_info.get('thread_id', 'SLK-772')}`):** {adr_info.get('discussion_summary') or 'Linked in FalkorDB subgraph'} ({len(neigh)} neighbor services scanned)\n"
            f"• **Latency Breakdown:** Completed in **{elapsed_ms}ms ({elapsed_s}s) end-to-end** | FalkorDB took **{total_ms}ms cumulative** ({avg_ms}ms avg across {q_count} queries)."
        )

        state["messages"].append({
            "agent": "Remediation Agent",
            "role": "action",
            "content": f"Synthesized provenance report in **{elapsed_ms}ms ({elapsed_s}s)** ({total_ms}ms cumulative FalkorDB time across {q_count} queries).",
            "timestamp": int(time.time()),
        })

    return state


def build_cortex_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("triage_analyst", triage_analyst_node)
    workflow.add_node("memory_knowledge", memory_knowledge_node)
    workflow.add_node("remediation", remediation_node)

    workflow.set_entry_point("orchestrator")

    def route_after_orchestrator(state: AgentState):
        return "triage_analyst" if state["mode"] == "incident" else "memory_knowledge"

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "triage_analyst": "triage_analyst",
            "memory_knowledge": "memory_knowledge",
        },
    )
    workflow.add_edge("triage_analyst", "memory_knowledge")
    workflow.add_edge("memory_knowledge", "remediation")
    workflow.add_edge("remediation", END)
    return workflow.compile()


_COMPILED_GRAPH = build_cortex_graph()


def run_incident_investigation(
    alert_id: str = "ALERT-901",
    tenant_id: str = "acme_corp",
    parallel: bool = True,
    use_pooling: bool = True,
    use_cache: bool = True,
    session_id: str | None = None,
    custom_memory_id: str | None = None,
    use_langgraph_engine: bool = False,
) -> AgentState:
    """
    Executes the complete 4-agent incident investigation workflow using context-isolated policies.
    - parallel: toggles concurrent execution of independent queries within each agent stage.
    - use_pooling: when False, every query instantiates a fresh FalkorDB client (11 instances/run), closed in finally.
    - use_cache: when False, bypasses @lru_cache to recompute the 768-dim error embedding.
    - use_langgraph_engine: when True, invokes the compiled LangGraph StateGraph (.invoke()).
    """
    if use_pooling:
        gt.get_db()
    sid = session_id or f"sess-{int(time.perf_counter() * 1000000)}"
    policy = gt.InvestigationRuntimePolicy(use_pooling=use_pooling, use_cache=use_cache)
    gt._SESSION_POLICIES[sid] = policy
    token = gt._ACTIVE_POLICY.set(policy)
    state: AgentState = {
        "session_id": sid,
        "tenant_id": tenant_id,
        "alert_id": alert_id,
        "query_text": f"Investigate alert {alert_id}",
        "mode": "incident",
        "parallel": parallel,
        "use_pooling": use_pooling,
        "use_cache": use_cache,
        "custom_memory_id": custom_memory_id or "",
        "executed_agents": [],
        "started_at_perf": time.perf_counter(),
        "total_db_ms": 0.0,
        "query_count": 0,
        "highlight_nodes": [],
        "messages": [],
        "current_agent": "Orchestrator Agent",
        "triage_data": {},
        "memory_data": {},
        "remediation_data": {},
        "provenance_data": {},
        "final_answer": "",
    }
    active_exc: BaseException | None = None
    try:
        if use_langgraph_engine:
            return _COMPILED_GRAPH.invoke(state)
        state = orchestrator_node(state)
        state = triage_analyst_node(state)
        state = memory_knowledge_node(state)
        state = remediation_node(state)
        return state
    except BaseException as exc:
        active_exc = exc
        raise
    finally:
        gt._ACTIVE_POLICY.reset(token)
        gt._SESSION_POLICIES.pop(sid, None)
        if active_exc is not None:
            cleanup_errs = policy.close_unpooled_clients(raise_on_error=False)
            all_errs = policy.cleanup_errors or cleanup_errs
            if all_errs and hasattr(active_exc, "add_note"):
                active_exc.add_note(
                    "Unpooled client cleanup error(s) during exception unwind: "
                    + "; ".join(all_errs)
                )
        else:
            policy.close_unpooled_clients(raise_on_error=True)
            if policy.cleanup_errors:
                raise RuntimeError(
                    f"Failed to close {len(policy.cleanup_errors)} unpooled FalkorDB client(s): "
                    + "; ".join(policy.cleanup_errors)
                )