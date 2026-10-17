import os
import time
from falkordb import FalkorDB
from embeddings import get_embedding, cosine_similarity

FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", 6379))


def seed_acme_corp(db: FalkorDB, now: int):
    graph = db.select_graph("tenant_acme_corp")
    try:
        graph.delete()
    except Exception:
        pass

    print("1. Creating property & vector indexes for tenant_acme_corp...")
    index_queries = [
        "CREATE INDEX FOR (s:Microservice) ON (s.name)",
        "CREATE INDEX FOR (a:IncidentAlert) ON (a.alert_id)",
        "CREATE INDEX FOR (a:IncidentAlert) ON (a.error_signature)",
        "CREATE INDEX FOR (t:TriageResult) ON (t.session_id)",
        "CREATE INDEX FOR (p:Person) ON (p.github_handle)",
        "CREATE INDEX FOR (pr:PullRequest) ON (pr.pr_id)",
        "CREATE INDEX FOR (pr:PullRequest) ON (pr.merged_at_epoch)",
        "CREATE INDEX FOR (m:ResolutionMemory) ON (m.memory_id)",
        "CREATE INDEX FOR (c:ConversationTurn) ON (c.session_id)",
        "CREATE INDEX FOR (sess:Session) ON (sess.session_id)",
    ]
    for q in index_queries:
        try:
            graph.query(q)
        except Exception:
            pass

    try:
        graph.query(
            "CREATE VECTOR INDEX FOR (a:IncidentAlert) ON (a.error_embedding) "
            "OPTIONS {dimension: 768, similarityFunction: 'cosine'}"
        )
    except Exception as e:
        print(f"   [Note] Vector index creation note: {e}")

    print("2. Creating Microservices, Databases, Teams, and Developers...")
    graph.query(
        """
        CREATE
          (gw:Microservice {name: 'APIGateway', status: 'healthy', tier: 1, language: 'Go', team_owner: 'Platform', ui_x: 420, ui_y: 65}),
          (web:Microservice {name: 'WebStorefront', status: 'healthy', tier: 1, language: 'TypeScript', team_owner: 'Checkout', ui_x: 640, ui_y: 65}),
          (chk:Microservice {name: 'CheckoutService', status: 'down', tier: 1, language: 'Python', team_owner: 'Checkout', ui_x: 485, ui_y: 195}),
          (ord:Microservice {name: 'OrderService', status: 'degraded', tier: 2, language: 'Python', team_owner: 'Checkout', ui_x: 665, ui_y: 195}),
          (pay:Microservice {name: 'PaymentService', status: 'degraded', tier: 2, language: 'Go', team_owner: 'Checkout', ui_x: 405, ui_y: 335}),
          (inv:Microservice {name: 'InventoryService', status: 'healthy', tier: 2, language: 'Java', team_owner: 'CoreInfra', ui_x: 640, ui_y: 335}),
          (auth:Microservice {name: 'AuthTokenService', status: 'degraded', tier: 3, language: 'Go', team_owner: 'Platform', ui_x: 495, ui_y: 475}),
          (fraud:Microservice {name: 'FraudEngine', status: 'healthy', tier: 3, language: 'Python', team_owner: 'CoreInfra', ui_x: 675, ui_y: 475}),

          (pg:Database {name: 'PostgresPrimary', engine: 'postgres', host: 'pg.acme.internal', port: 5432, status: 'healthy', ui_x: 410, ui_y: 595}),
          (rd:Database {name: 'RedisCache', engine: 'redis', host: 'redis.acme.internal', port: 6379, status: 'healthy', ui_x: 635, ui_y: 595}),

          (t_plat:Team {name: 'Platform', department: 'Infrastructure', slack_channel: '#team-platform'}),
          (t_chk:Team {name: 'Checkout', department: 'Commerce', slack_channel: '#team-checkout'}),
          (t_core:Team {name: 'CoreInfra', department: 'Infrastructure', slack_channel: '#team-core'}),

          (p1:Person {name: 'Aarav Sharma', email: 'aarav@acme.io', role: 'lead', slack_handle: '@aarav', github_handle: 'aarav-s', timezone: 'IST'}),
          (p2:Person {name: 'Priya Nair', email: 'priya@acme.io', role: 'SRE', slack_handle: '@priya.sre', github_handle: 'priyanair', timezone: 'IST'}),
          (p3:Person {name: 'Marcus Vance', email: 'marcus@acme.io', role: 'backend', slack_handle: '@mvance', github_handle: 'mvance', timezone: 'UTC'}),
          (p4:Person {name: 'Elena Rostova', email: 'elena@acme.io', role: 'backend', slack_handle: '@elena', github_handle: 'erostova', timezone: 'CET'}),

          (gw)-[:DEPENDS_ON]->(chk),
          (gw)-[:DEPENDS_ON]->(ord),
          (web)-[:DEPENDS_ON]->(gw),
          (chk)-[:DEPENDS_ON]->(pay),
          (chk)-[:DEPENDS_ON]->(ord),
          (ord)-[:DEPENDS_ON]->(inv),
          (pay)-[:DEPENDS_ON]->(auth),
          (pay)-[:DEPENDS_ON]->(fraud),
          (auth)-[:CONNECTS_TO]->(rd),
          (pay)-[:CONNECTS_TO]->(pg),
          (chk)-[:CONNECTS_TO]->(rd),

          (p1)-[:MEMBER_OF]->(t_chk),
          (p2)-[:MEMBER_OF]->(t_plat),
          (p3)-[:MEMBER_OF]->(t_plat),
          (p4)-[:MEMBER_OF]->(t_core),

          (p1)-[:OWNS]->(chk),
          (p1)-[:OWNS]->(pay),
          (p2)-[:OWNS]->(auth),
          (p3)-[:OWNS]->(gw),
          (p4)-[:OWNS]->(inv)
        """
    )

    print("3. Creating Pull Requests, Deploys, ADRs, and Slack Provenance...")
    graph.query(
        """
        MATCH (auth:Microservice {name: 'AuthTokenService'}),
              (pay:Microservice {name: 'PaymentService'}),
              (chk:Microservice {name: 'CheckoutService'}),
              (p1:Person {github_handle: 'aarav-s'}),
              (p2:Person {github_handle: 'priyanair'}),
              (p3:Person {github_handle: 'mvance'})
        CREATE
          (pr402:PullRequest {
            pr_id: 'PR-402',
            title: 'Rotate JWT signing key & enforce strict RS256 header check',
            author: 'mvance',
            merged_at_epoch: $pr402_time,
            merged_at_iso: '2026-10-15T01:15:00Z',
            files_changed: 6,
            diff_summary: 'Rejected legacy HS256 tokens from PaymentService',
            risk_score: 4,
            risk_label: 'critical'
          }),
          (pr401:PullRequest {
            pr_id: 'PR-401',
            title: 'Increase payment retry timeout from 2s to 3s',
            author: 'aarav-s',
            merged_at_epoch: $pr401_time,
            merged_at_iso: '2026-10-14T20:00:00Z',
            files_changed: 2,
            diff_summary: 'Bumped HTTP timeout constant in PaymentService',
            risk_score: 2,
            risk_label: 'medium'
          }),
          (pr398:PullRequest {
            pr_id: 'PR-398',
            title: 'Add Prometheus latency histogram to CheckoutService',
            author: 'aarav-s',
            merged_at_epoch: $pr398_time,
            merged_at_iso: '2026-10-14T14:00:00Z',
            files_changed: 3,
            diff_summary: 'Non-blocking metrics instrumentation',
            risk_score: 1,
            risk_label: 'low'
          }),
          (d1:DeployEvent {
            deploy_id: 'DEP-881',
            service_name: 'AuthTokenService',
            version: 'v2.14.0',
            deployer: 'mvance',
            timestamp_epoch: $pr402_time + 300,
            timestamp_iso: '2026-10-15T01:20:00Z',
            rollback_available: true
          }),
          (adr1:ArchDecisionRecord {
            adr_id: 'ADR-019',
            title: 'Use Redis Ephemeral Sessions for Checkout Cart State',
            decision: 'Store active checkout session state in RedisCache instead of PostgresPrimary.',
            rationale: 'Reduces p99 checkout lock contention by 78% during flash sales.',
            status: 'accepted',
            decided_at_iso: '2026-08-12'
          }),
          (slack1:SlackDiscussion {
            thread_id: 'SLK-772',
            channel: '#arch-review',
            permalink: 'https://acme.slack.com/archives/C01ARCH/p1723459200',
            summary: 'Team agreed Postgres row locks were bottlenecking flash sales; approved Redis migration.',
            timestamp_epoch: $now - 500000,
            timestamp_iso: '2026-08-11T16:30:00Z'
          }),

          (pr402)-[:MODIFIED]->(auth),
          (pr401)-[:MODIFIED]->(pay),
          (pr398)-[:MODIFIED]->(chk),
          (p3)-[:AUTHORED]->(pr402),
          (p2)-[:REVIEWED]->(pr402),
          (p1)-[:AUTHORED]->(pr401),
          (p2)-[:REVIEWED]->(pr401),
          (p1)-[:AUTHORED]->(pr398),
          (d1)-[:DEPLOYED]->(auth),
          (d1)-[:DEPLOYED_BY]->(p3),
          (adr1)-[:APPLIES_TO]->(chk),
          (slack1)-[:LED_TO]->(adr1),
          (slack1)-[:DISCUSSED]->(chk)
        """,
        {
            "now": now,
            "pr402_time": now - 7200,
            "pr401_time": now - 21600,
            "pr398_time": now - 43200,
        },
    )

    print("4. Generating Embeddings & Seeding Incident Alerts, Memory & Session Turns...")
    past_auth_log = "ERROR 401 Unauthorized: JWT signature verification failed on AuthTokenService (RS256 key mismatch)"
    past_db_log = "FATAL: remaining connection slots are reserved for superuser connections on PostgresPrimary pool timeout"
    active_alert_log = "CheckoutService HTTP 502 Bad Gateway: upstream PaymentService rejected by AuthTokenService (401 JWT signature mismatch)"

    emb_past_auth = get_embedding(past_auth_log)
    emb_past_db = get_embedding(past_db_log)
    emb_active = get_embedding(active_alert_log)

    sim_same = cosine_similarity(emb_active, emb_past_auth)
    sim_diff = cosine_similarity(emb_active, emb_past_db)
    print(f"   [Embedding Check] Similarity (Active vs Past Auth Incident): {sim_same:.4f}")
    print(f"   [Embedding Check] Similarity (Active vs Past DB Incident):   {sim_diff:.4f}")

    graph.query(
        """
        MATCH (chk:Microservice {name: 'CheckoutService'}),
              (auth:Microservice {name: 'AuthTokenService'}),
              (pay:Microservice {name: 'PaymentService'}),
              (pr402:PullRequest {pr_id: 'PR-402'})
        CREATE
          (a_past1:IncidentAlert {
            alert_id: 'ALERT-710',
            severity: 'P1',
            severity_rank: 1,
            error_log: $past_auth_log,
            error_signature: 'ERR_JWT_SIG_MISMATCH_401',
            error_embedding: vecf32($emb_past_auth),
            timestamp_epoch: $now - 2592000,
            timestamp_iso: '2026-09-15T03:10:00Z',
            status: 'resolved',
            mttr_minutes: 14
          }),
          (a_past2:IncidentAlert {
            alert_id: 'ALERT-650',
            severity: 'P2',
            severity_rank: 2,
            error_log: $past_db_log,
            error_signature: 'ERR_PG_POOL_EXHAUSTED',
            error_embedding: vecf32($emb_past_db),
            timestamp_epoch: $now - 5184000,
            timestamp_iso: '2026-08-15T11:40:00Z',
            status: 'resolved',
            mttr_minutes: 28
          }),
          (a_active:IncidentAlert {
            alert_id: 'ALERT-901',
            severity: 'P1',
            severity_rank: 1,
            error_log: $active_alert_log,
            error_signature: 'ERR_JWT_SIG_MISMATCH_401',
            error_embedding: vecf32($emb_active),
            timestamp_epoch: $now,
            timestamp_iso: '2026-10-15T03:15:00Z',
            status: 'open',
            mttr_minutes: 0
          }),

          (mem1:ResolutionMemory {
            memory_id: 'MEM-301',
            solution_summary: 'Rollback AuthTokenService deploy to restore dual HS256/RS256 key validation and flush Redis JWKS cache.',
            steps_taken: 'Rolled back AuthTokenService to v2.13.9 and flushed JWKS cache.',
            success_rate: 0.98,
            applied_count: 4,
            created_at_epoch: $now - 2591000,
            created_at_iso: '2026-09-15T03:25:00Z'
          }),
          (step1:ProcedureStep {
            step_id: 'STEP-1',
            runbook_name: 'RB-JWT-ROLLBACK',
            step_order: 1,
            instruction: 'Verify offending PR modified JWT key validation on AuthTokenService.',
            auto_exec: true
          }),
          (step2:ProcedureStep {
            step_id: 'STEP-2',
            runbook_name: 'RB-JWT-ROLLBACK',
            step_order: 2,
            instruction: 'Execute rollback of AuthTokenService deployment DEP-881 to last stable version.',
            auto_exec: true
          }),
          (step3:ProcedureStep {
            step_id: 'STEP-3',
            runbook_name: 'RB-JWT-ROLLBACK',
            step_order: 3,
            instruction: 'Page service owner (@priya.sre) and Checkout lead (@aarav) with provenance link.',
            auto_exec: true
          }),

          (sess0:Session {
            session_id: 'sess-baseline-001',
            tenant_id: 'acme_corp',
            started_at_epoch: $now - 3600
          }),
          (turn1:ConversationTurn {
            turn_id: 'TURN-1',
            session_id: 'sess-baseline-001',
            agent_name: 'Triage & Code Analyst Agent',
            role: 'action',
            content: 'Isolated root cause on AuthTokenService via PR-402.',
            timestamp_epoch: $now - 3595
          }),
          (turn2:ConversationTurn {
            turn_id: 'TURN-2',
            session_id: 'sess-baseline-001',
            agent_name: 'Memory & Knowledge Agent',
            role: 'action',
            content: 'Recalled runbook MEM-301 for JWT signature mismatch.',
            timestamp_epoch: $now - 3590
          }),

          (a_past1)-[:TRIGGERED_ON]->(auth),
          (a_past2)-[:TRIGGERED_ON]->(pay),
          (a_active)-[:TRIGGERED_ON]->(chk),
          (a_active)-[:CAUSED_BY]->(pr402),
          (mem1)-[:RESOLVED]->(a_past1),
          (mem1)-[:USED_STEP]->(step1),
          (mem1)-[:USED_STEP]->(step2),
          (mem1)-[:USED_STEP]->(step3),
          (turn1)-[:IN_SESSION]->(sess0),
          (turn2)-[:IN_SESSION]->(sess0),
          (turn1)-[:REFERENCES]->(pr402),
          (turn2)-[:REFERENCES]->(mem1)
        """,
        {
            "now": now,
            "past_auth_log": past_auth_log,
            "past_db_log": past_db_log,
            "active_alert_log": active_alert_log,
            "emb_past_auth": emb_past_auth,
            "emb_past_db": emb_past_db,
            "emb_active": emb_active,
        },
    )


def seed_beta_health(db: FalkorDB, now: int):
    print("5. Seeding second isolated tenant graph (tenant_beta_health)...")
    graph = db.select_graph("tenant_beta_health")
    try:
        graph.delete()
    except Exception:
        pass

    graph.query(
        """
        CREATE
          (portal:Microservice {name: 'PatientPortal', status: 'healthy', tier: 1, language: 'TypeScript', team_owner: 'ClinicalUX', ui_x: 500, ui_y: 100}),
          (ehr:Microservice {name: 'EHRRecordsAPI', status: 'degraded', tier: 2, language: 'Java', team_owner: 'HealthCore', ui_x: 360, ui_y: 260}),
          (dicom:Microservice {name: 'ImagingService', status: 'healthy', tier: 2, language: 'C++', team_owner: 'HealthCore', ui_x: 640, ui_y: 260}),
          (hipaa:Microservice {name: 'HIPAAuditVault', status: 'healthy', tier: 3, language: 'Go', team_owner: 'SecOps', ui_x: 500, ui_y: 420}),
          (portal)-[:DEPENDS_ON]->(ehr),
          (portal)-[:DEPENDS_ON]->(dicom),
          (ehr)-[:DEPENDS_ON]->(hipaa),
          (dicom)-[:DEPENDS_ON]->(hipaa)
        """
    )


if __name__ == "__main__":
    print(f"Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}...")
    db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    now_epoch = int(time.time())
    seed_acme_corp(db, now_epoch)
    seed_beta_health(db, now_epoch)
    print("\n✅ SUCCESS! Both tenant_acme_corp and tenant_beta_health are seeded.")