import uuid

import graph_tools as gt


def test_real_falkordb_tenant_isolation():
    """Verify a node in one real FalkorDB graph is invisible in another."""
    test_id = f"CORTEX_ISO_TEST_{uuid.uuid4().hex}"

    acme = gt.get_graph("acme_corp")
    beta = gt.get_graph("beta_health")

    try:
        acme.query(
            "CREATE (:IsolationTest {id: $test_id})",
            {"test_id": test_id},
        )

        acme_result = acme.query(
            "MATCH (n:IsolationTest {id: $test_id}) RETURN count(n)",
            {"test_id": test_id},
        )
        beta_result = beta.query(
            "MATCH (n:IsolationTest {id: $test_id}) RETURN count(n)",
            {"test_id": test_id},
        )

        assert acme_result.result_set[0][0] == 1
        assert beta_result.result_set[0][0] == 0

    finally:
        acme.query(
            "MATCH (n:IsolationTest {id: $test_id}) DELETE n",
            {"test_id": test_id},
        )
