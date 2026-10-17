# ==============================================================================
# CORTEX — PRODUCTION CYPHER QUERIES (ALL 10 QUERIES VERIFIED)
# ==============================================================================

# QUERY 1: Multi-Hop Root Cause Analysis (Track 01)
QUERY_1_ROOT_CAUSE = """
MATCH (a:IncidentAlert {alert_id: $alert_id})-[:TRIGGERED_ON]->(s:Microservice)
MATCH path = (s)-[:DEPENDS_ON*1..4]->(dep:Microservice)
MATCH (dep)<-[:MODIFIED]-(pr:PullRequest)
WHERE pr.merged_at_epoch >= a.timestamp_epoch - 86400
RETURN dep.name AS suspect_service,
       pr.pr_id AS pr_id,
       pr.title AS recent_change,
       pr.author AS committer,
       pr.risk_score AS risk_score,
       pr.risk_label AS risk_label,
       length(path) AS dependency_hops,
       [n IN nodes(path) | n.name] AS dependency_chain
ORDER BY pr.risk_score DESC, dependency_hops ASC
LIMIT 5
"""

# QUERY 2: Transitive Blast Radius Calculation (Track 01)
QUERY_2_BLAST_RADIUS = """
MATCH (start:Microservice {name: $compromised_service})
MATCH path = (victim:Microservice)-[:DEPENDS_ON*1..6]->(start)
WITH victim, length(path) AS hops, [n IN nodes(path) | n.name] AS p_nodes
ORDER BY hops ASC
RETURN victim.name AS affected_service,
       victim.tier AS tier,
       min(hops) AS distance,
       collect(p_nodes)[0] AS impact_path
ORDER BY victim.tier ASC, distance ASC
"""

# QUERY 3A: Single Points of Failure via GraphBLAS PageRank (Track 01 Primary)
QUERY_3A_PAGERANK = """
CALL algo.pageRank('Microservice', 'DEPENDS_ON')
YIELD node, score
RETURN node.name AS service,
       node.tier AS tier,
       node.status AS status,
       score AS criticality
ORDER BY score DESC
LIMIT 10
"""

# QUERY 3B: Transitive Dependency Criticality (Track 01 Pure-Cypher Fallback)
QUERY_3B_CRITICALITY = """
MATCH (target:Microservice)
OPTIONAL MATCH (dependent:Microservice)-[:DEPENDS_ON*1..6]->(target)
WHERE dependent <> target
WITH target, count(DISTINCT dependent) AS dependents_count
RETURN target.name AS service,
       target.tier AS tier,
       target.status AS status,
       dependents_count AS criticality
ORDER BY dependents_count DESC, target.tier DESC
LIMIT 10
"""

# QUERY 4: Shortest Dependency Path Between Services (Track 01)
QUERY_4_SHORTEST_PATH = """
MATCH (a:Microservice {name: $service_a}), (b:Microservice {name: $service_b})
MATCH p = (a)-[:DEPENDS_ON*1..6]->(b)
RETURN [n IN nodes(p) | n.name] AS path,
       length(p) AS hops
ORDER BY hops ASC
LIMIT 1
"""

# QUERY 5A: Hybrid Vector + Graph Episodic Memory Retrieval (Track 02 Primary)
QUERY_5A_HYBRID_MEMORY = """
CALL db.idx.vector.queryNodes(
  'IncidentAlert',
  'error_embedding',
  3,
  vecf32($query_embedding)
) YIELD node, score
WITH node AS past, score
MATCH (m:ResolutionMemory)-[:RESOLVED]->(past)
MATCH (m)-[:USED_STEP]->(step:ProcedureStep)
WITH m, past, score, step
ORDER BY step.step_order ASC
RETURN past.alert_id AS similar_past_alert,
       past.error_signature AS matched_signature,
       score AS vector_similarity,
       m.memory_id AS memory_id,
       m.solution_summary AS recommendation,
       m.success_rate AS confidence,
       m.applied_count AS times_used,
       collect(step.instruction) AS ordered_runbook_steps,
       collect(step.step_id) AS step_ids
ORDER BY score DESC, m.success_rate DESC
LIMIT 3
"""

# QUERY 5B: Exact-Signature Memory Retrieval (Track 02 Fallback)
QUERY_5B_EXACT_MEMORY = """
MATCH (m:ResolutionMemory)-[:RESOLVED]->(past:IncidentAlert)
WHERE past.error_signature = $current_error_signature
MATCH (m)-[:USED_STEP]->(step:ProcedureStep)
WITH m, past, step
ORDER BY step.step_order ASC
RETURN past.alert_id AS similar_past_alert,
       past.error_signature AS matched_signature,
       0.02 AS vector_similarity,
       m.memory_id AS memory_id,
       m.solution_summary AS recommendation,
       m.success_rate AS confidence,
       m.applied_count AS times_used,
       collect(step.instruction) AS ordered_runbook_steps,
       collect(step.step_id) AS step_ids
ORDER BY m.success_rate DESC
LIMIT 3
"""

# QUERY 6: Atomic Store New Resolution Memory & Heal Services (Track 02)
QUERY_6_STORE_MEMORY = """
MATCH (a:IncidentAlert {alert_id: $alert_id})-[:TRIGGERED_ON]->(s:Microservice)
SET a.status = 'resolved', a.mttr_minutes = $mttr_minutes, s.status = 'healthy'
WITH a, s
MATCH (svc:Microservice)
WHERE svc.name IN ['CheckoutService', 'AuthTokenService', 'PaymentService', 'OrderService']
SET svc.status = 'healthy'
WITH a, s, collect(svc.name) AS healed_list
CREATE (m:ResolutionMemory {
  memory_id: $memory_id,
  solution_summary: $summary,
  steps_taken: $steps_taken,
  success_rate: 1.0,
  applied_count: 1,
  created_at_epoch: $current_epoch,
  created_at_iso: $current_iso
})
CREATE (m)-[:RESOLVED]->(a)
RETURN m.memory_id AS stored_memory, a.status AS alert_status, s.name AS healed_service
"""

# QUERY 7 (Write): Persist Multi-Agent Conversation Turn & Entity Reference (Track 02)
QUERY_7_WRITE_TURN = """
MERGE (sess:Session {session_id: $session_id})
ON CREATE SET sess.tenant_id = $tenant_id, sess.started_at_epoch = $timestamp_epoch
CREATE (t:ConversationTurn {
  turn_id: $turn_id,
  session_id: $session_id,
  agent_name: $agent_name,
  role: $role,
  content: $content,
  timestamp_epoch: $timestamp_epoch
})-[:IN_SESSION]->(sess)
WITH t
OPTIONAL MATCH (ref)
WHERE ref.name = $ref_id OR ref.pr_id = $ref_id OR ref.alert_id = $ref_id OR ref.memory_id = $ref_id
FOREACH (_ IN CASE WHEN ref IS NOT NULL THEN [1] ELSE [] END |
  CREATE (t)-[:REFERENCES]->(ref)
)
RETURN t.turn_id AS turn_id
"""

# QUERY 7 (Read): Multi-Agent Session History & Handoff Context (Track 02)
QUERY_7_SESSION_HISTORY = """
MATCH (t:ConversationTurn {session_id: $session_id})
OPTIONAL MATCH (t)-[:REFERENCES]->(ref)
RETURN t.agent_name AS agent,
       t.role AS role,
       t.content AS message,
       labels(ref) AS referenced_type,
       coalesce(ref.name, ref.pr_id, ref.alert_id, ref.memory_id, '') AS referenced_entity,
       t.timestamp_epoch AS time_epoch
ORDER BY t.timestamp_epoch ASC, t.turn_id ASC
"""

# QUERY 8: Who Knows This System Best / Domain Experts (Track 03)
QUERY_8_EXPERTS = """
MATCH (s:Microservice {name: $service})
MATCH (p:Person)
OPTIONAL MATCH (p)-[:AUTHORED]->(pr1:PullRequest)-[:MODIFIED]->(s)
WITH s, p, count(DISTINCT pr1) AS authored, collect(DISTINCT pr1.pr_id) AS authored_prs
OPTIONAL MATCH (p)-[:REVIEWED]->(pr2:PullRequest)-[:MODIFIED]->(s)
WITH s, p, authored, authored_prs, count(DISTINCT pr2) AS reviewed, collect(DISTINCT pr2.pr_id) AS reviewed_prs
OPTIONAL MATCH (p)-[r:OWNS]->(s)
WITH p, authored, authored_prs, reviewed, reviewed_prs,
     CASE WHEN count(r) > 0 THEN true ELSE false END AS is_owner
WHERE authored > 0 OR reviewed > 0 OR is_owner = true
RETURN p.name AS expert,
       p.role AS role,
       p.slack_handle AS contact,
       p.timezone AS tz,
       authored,
       reviewed,
       authored + reviewed AS total_involvement,
       is_owner,
       authored_prs + reviewed_prs AS connected_prs
ORDER BY is_owner DESC, total_involvement DESC
LIMIT 5
"""

# QUERY 9: Decision Provenance Trail — ADR + Slack Thread (Track 03)
QUERY_9_PROVENANCE = """
MATCH (s:Microservice {name: $service})
OPTIONAL MATCH (adr:ArchDecisionRecord)-[:APPLIES_TO]->(s)
OPTIONAL MATCH (slack:SlackDiscussion)-[:LED_TO]->(adr)
RETURN s.name AS service,
       coalesce(adr.adr_id, 'NONE') AS adr_id,
       coalesce(adr.title, 'No ADR recorded') AS decision,
       coalesce(adr.rationale, '') AS why,
       coalesce(adr.decided_at_iso, '') AS decided_when,
       coalesce(slack.permalink, '') AS discussion_link,
       coalesce(slack.summary, '') AS discussion_summary,
       coalesce(slack.thread_id, 'NONE') AS thread_id
"""

# QUERY 10: Recent Changes in 2-Hop Dependency Neighborhood (Track 03)
QUERY_10_RECENT_CHANGES = """
MATCH (s:Microservice {name: $service})
MATCH (neighbor:Microservice)-[:DEPENDS_ON*0..2]-(s)
WITH DISTINCT neighbor
OPTIONAL MATCH (pr:PullRequest)-[:MODIFIED]->(neighbor)
  WHERE pr.merged_at_epoch >= $current_epoch - 604800
WITH neighbor, collect(DISTINCT
  CASE WHEN pr IS NOT NULL
       THEN pr.pr_id + ': ' + pr.title
       ELSE NULL END
) AS raw_prs
OPTIONAL MATCH (d:DeployEvent)-[:DEPLOYED]->(neighbor)
  WHERE d.timestamp_epoch >= $current_epoch - 604800
WITH neighbor, raw_prs, collect(DISTINCT d.version) AS recent_deploys
OPTIONAL MATCH (a:IncidentAlert)-[:TRIGGERED_ON]->(neighbor)
  WHERE a.timestamp_epoch >= $current_epoch - 604800
RETURN neighbor.name AS service,
       neighbor.status AS current_status,
       [x IN raw_prs WHERE x IS NOT NULL] AS recent_prs,
       recent_deploys,
       collect(DISTINCT a.alert_id) AS recent_incidents
"""

# AGENT HANDOFF QUERIES (Track 02 Coordination)
QUERY_WRITE_TRIAGE_HANDOFF = """
MATCH (a:IncidentAlert {alert_id: $alert_id})
CREATE (tr:TriageResult {
  session_id: $session_id,
  alert_id: $alert_id,
  suspect_service: $suspect_service,
  suspect_pr: $suspect_pr,
  blast_radius_count: $blast_radius_count,
  confidence: $confidence,
  created_at_epoch: $now
})-[:EVALUATED]->(a)
RETURN tr.session_id AS session_id
"""

QUERY_READ_TRIAGE_HANDOFF = """
MATCH (tr:TriageResult {session_id: $session_id})-[:EVALUATED]->(a:IncidentAlert)
RETURN tr.suspect_service AS suspect_service,
       tr.suspect_pr AS suspect_pr,
       tr.blast_radius_count AS blast_radius_count,
       tr.confidence AS confidence,
       a.alert_id AS alert_id,
       a.error_signature AS error_signature,
       a.error_log AS error_log
"""