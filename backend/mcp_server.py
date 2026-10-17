import json
from mcp.server.fastmcp import FastMCP
import graph_tools as gt

mcp = FastMCP("cortex-falkordb-mcp")


@mcp.tool()
def find_root_cause(alert_id: str = "ALERT-901", tenant_id: str = "acme_corp") -> str:
    """Query 1 (Track 01): Multi-hop FalkorDB traversal to isolate the root cause PR of an incident."""
    return json.dumps(gt.find_root_cause(alert_id, tenant_id), indent=2)


@mcp.tool()
def get_blast_radius(service_name: str = "AuthTokenService", tenant_id: str = "acme_corp") -> str:
    """Query 2 (Track 01): Calculate the upstream transitive blast radius of a failing microservice."""
    return json.dumps(gt.get_blast_radius(service_name, tenant_id), indent=2)


@mcp.tool()
def get_critical_services(tenant_id: str = "acme_corp") -> str:
    """Query 3 (Track 01): Rank single points of failure via GraphBLAS PageRank / transitive criticality."""
    return json.dumps(gt.get_critical_services(tenant_id), indent=2)


@mcp.tool()
def get_shortest_dependency_path(
    service_a: str = "APIGateway",
    service_b: str = "AuthTokenService",
    tenant_id: str = "acme_corp",
) -> str:
    """Query 4 (Track 01): Find the shortest dependency chain between any two microservices."""
    return json.dumps(gt.get_shortest_dependency_path(service_a, service_b, tenant_id), indent=2)


@mcp.tool()
def recall_incident_memory(error_log: str, tenant_id: str = "acme_corp") -> str:
    """Query 5 (Track 02): Hybrid 768-dim Vector + Graph traversal to retrieve past incident runbooks."""
    return json.dumps(gt.recall_incident_memory(error_log, tenant_id=tenant_id), indent=2)


@mcp.tool()
def store_resolution_memory(
    alert_id: str = "ALERT-901",
    summary: str = "Rolled back AuthTokenService deploy DEP-881",
    steps_taken: str = "Verified PR-402 and rolled back deployment",
    mttr_minutes: int = 2,
    tenant_id: str = "acme_corp",
) -> str:
    """Query 6 (Track 02): Persist a new ResolutionMemory node and heal affected services in FalkorDB."""
    return json.dumps(
        gt.store_resolution_memory(alert_id, summary, steps_taken, mttr_minutes, tenant_id),
        indent=2,
    )


@mcp.tool()
def get_session_history(session_id: str = "sess-baseline-001", tenant_id: str = "acme_corp") -> str:
    """Query 7 (Track 02): Retrieve persistent multi-agent conversation turns and referenced graph entities."""
    return json.dumps(gt.get_session_history(session_id, tenant_id), indent=2)


@mcp.tool()
def get_service_experts(service_name: str = "CheckoutService", tenant_id: str = "acme_corp") -> str:
    """Query 8 (Track 03): Identify domain experts and owners for a microservice from the Company Brain graph."""
    return json.dumps(gt.get_service_experts(service_name, tenant_id), indent=2)


@mcp.tool()
def get_decision_provenance(service_name: str = "CheckoutService", tenant_id: str = "acme_corp") -> str:
    """Query 9 (Track 03): Trace Architecture Decision Records (ADRs) and Slack threads for a service."""
    return json.dumps(gt.get_decision_provenance(service_name, tenant_id), indent=2)


@mcp.tool()
def get_recent_neighborhood_changes(
    service_name: str = "CheckoutService", tenant_id: str = "acme_corp"
) -> str:
    """Query 10 (Track 03): Scan the 2-hop dependency neighborhood for recent PRs, deploys, and alerts."""
    return json.dumps(gt.get_recent_neighborhood_changes(service_name, tenant_id), indent=2)


if __name__ == "__main__":
    mcp.run()