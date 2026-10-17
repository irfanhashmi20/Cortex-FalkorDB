import os
import re
import hmac
import json
import time
import socket
import logging
import asyncio
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
import seed_graph
import graph_tools as gt
from agent_swarm import (
    orchestrator_node,
    triage_analyst_node,
    memory_knowledge_node,
    remediation_node,
)

logger = logging.getLogger("cortex.api")

app = FastAPI(title="CORTEX Graph API", version="4.0")

_DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500"
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_SAFE_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}/[A-Za-z0-9_.\-]{1,64}$")

_CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

_VENDOR_DIR = Path(__file__).resolve().parent.parent / "frontend" / "vendor"
_VENDOR_ASSETS = {
    "tailwindcss.js": "https://cdn.tailwindcss.com",
    "cytoscape.min.js": "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js",
}


def _ensure_vendor_asset(asset_name: str) -> Path:
    """Downloads and caches whitelisted frontend JS libraries locally so no third-party CDN is required."""
    if asset_name not in _VENDOR_ASSETS:
        raise HTTPException(status_code=404, detail="Asset not found.")
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    target_path = _VENDOR_DIR / asset_name
    if not target_path.exists() or target_path.stat().st_size == 0:
        url = _VENDOR_ASSETS[asset_name]
        req = urllib.request.Request(url, headers={"User-Agent": "CORTEX/4.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            target_path.write_bytes(resp.read())
    return target_path


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Applies OWASP-recommended security and anti-caching headers to all HTTP responses."""
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _CSP_POLICY
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class TriggerRequest(BaseModel):
    alert_id: Optional[str] = "ALERT-901"
    tenant_id: Optional[str] = "acme_corp"


class ChatRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = "acme_corp"


class IngestKnowledgeRequest(BaseModel):
    tenant_id: Optional[str] = "acme_corp"
    service_name: Optional[str] = "PaymentService"
    github_repo: Optional[str] = "FalkorDB/FalkorDB"
    pr_number: Optional[int] = None


def _verify_admin_api_key(request: Request) -> None:
    """
    Enforces X-API-Key authentication on administrative endpoints when CORTEX_ADMIN_API_KEY is set.
    If CORTEX_ADMIN_API_KEY is unset, local demo mode remains accessible.
    """
    expected_key = os.environ.get("CORTEX_ADMIN_API_KEY", "").strip()
    if not expected_key:
        return
    provided_key = (request.headers.get("X-API-Key") or "").strip()
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: valid X-API-Key header required for administrative operations.",
        )


def _validate_tenant_or_400(tenant_id: Optional[str]) -> str:
    """Validates tenant_id against ALLOWED_TENANTS and raises HTTP 400 on malicious/unknown input."""
    raw = tenant_id if tenant_id is not None else "acme_corp"
    try:
        return gt.normalize_tenant_id(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_identifier_or_400(value: Optional[str], field_name: str, default: str) -> str:
    """Validates entity identifiers (alert_id, service_name) against injection or malformed input."""
    val = (value if value is not None else default).strip()
    if not _SAFE_ID_PATTERN.match(val):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: {val!r}. Must be 1-64 alphanumeric, underscore, or hyphen characters.",
        )
    return val


def _validate_github_repo_or_400(repo: Optional[str]) -> str:
    """Validates GitHub owner/repo format to prevent SSRF or path traversal."""
    val = (repo if repo is not None else "FalkorDB/FalkorDB").strip()
    if not _SAFE_REPO_PATTERN.match(val) or ".." in val:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid github_repo: {val!r}. Expected format 'owner/repo'.",
        )
    return val


def _parse_github_iso_timestamp(iso_str: Optional[str], fallback_epoch: int) -> tuple[int, str]:
    """Parses a GitHub ISO-8601 timestamp into an exact Unix epoch integer and normalized ISO string."""
    if not iso_str or not isinstance(iso_str, str):
        return fallback_epoch, datetime.fromtimestamp(fallback_epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        cleaned = iso_str.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned).astimezone(timezone.utc)
        return int(dt.timestamp()), dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return fallback_epoch, iso_str.strip()


def _score_pr_risk(title: str, body: str, filenames: List[str], patch_preview: str) -> tuple[int, str]:
    """Computes numeric risk_score (1-4) and risk_label from PR title, body, changed file paths, and diff."""
    text = f"{title} {body} {' '.join(filenames)} {patch_preview}".lower()
    if any(k in text for k in ["auth", "jwt", "security", "breaking", "token", "crash", "critical", "cve"]):
        return 4, "critical"
    if any(k in text for k in ["memory", "timeout", "pool", "connection", "deadlock", "high", " locking"]):
        return 3, "high"
    if any(k in text for k in ["fix", "bug", "refactor", "index", "query", "perf", "cypher"]):
        return 2, "medium"
    return 1, "low"


def _fetch_pr_files_diff_metadata(github_repo: str, pr_num: int, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Calls GitHub's /repos/{owner}/{repo}/pulls/{pr_num}/files endpoint to retrieve actual
    changed file names, additions, deletions, and code diff patch snippets.
    """
    files_url = f"https://api.github.com/repos/{github_repo}/pulls/{pr_num}/files?per_page=10"
    req = urllib.request.Request(files_url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as resp:
        files_payload = json.loads(resp.read().decode("utf-8"))

    if not isinstance(files_payload, list) or not files_payload:
        return {
            "files_changed": 0,
            "changed_files_list": [],
            "additions": 0,
            "deletions": 0,
            "patch_preview": "",
            "diff_summary": "No file diff entries returned by GitHub.",
        }

    filenames: List[str] = []
    file_summaries: List[str] = []
    total_additions = 0
    total_deletions = 0
    first_patch = ""

    for f_obj in files_payload:
        if not isinstance(f_obj, dict):
            continue
        fname = str(f_obj.get("filename") or "unknown").strip()
        fstatus = str(f_obj.get("status") or "modified").strip()
        adds = int(f_obj.get("additions") or 0)
        dels = int(f_obj.get("deletions") or 0)
        total_additions += adds
        total_deletions += dels
        filenames.append(fname)
        file_summaries.append(f"{fname} ({fstatus}, +{adds}/-{dels})")
        if not first_patch and f_obj.get("patch"):
            raw_patch = str(f_obj["patch"]).replace("\r", " ").replace("\n", " ").strip()
            first_patch = raw_patch[:120]

    shown_files = ", ".join(file_summaries[:4])
    more_suffix = f" (+{len(file_summaries) - 4} more)" if len(file_summaries) > 4 else ""
    patch_suffix = f" | Patch: {first_patch}" if first_patch else ""
    diff_summary = (
        f"Changed {len(filenames)} file(s) (+{total_additions}/-{total_deletions}): "
        f"{shown_files}{more_suffix}{patch_suffix}"
    )[:300]

    return {
        "files_changed": len(filenames),
        "changed_files_list": filenames[:10],
        "additions": total_additions,
        "deletions": total_deletions,
        "patch_preview": first_patch,
        "diff_summary": diff_summary,
    }


def _build_offline_fallback_pr(github_repo: str, fallback_reason: str) -> Dict[str, Any]:
    """Returns a clearly labeled synthetic fallback record when GitHub is unreachable or rate-limited."""
    now_epoch = int(time.time())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ingestion_source": "fallback_offline",
        "fallback_reason": fallback_reason,
        "provenance_type": "offline_fallback",
        "adr_status": "synthetic_fallback",
        "github_repo": github_repo,
        "pr_id": "PR-405",
        "title": "Add circuit-breaker fallback for AuthTokenService JWT validation",
        "author_handle": "aarav-s",
        "author_name": "Aarav Sharma",
        "is_merged": False,
        "merged_at_epoch": None,
        "merged_at_iso": None,
        "event_timestamp_epoch": now_epoch,
        "event_timestamp_iso": now_iso,
        "files_changed": 2,
        "changed_files_list": ["src/auth/jwt_verifier.py", "src/payment/circuit_breaker.py"],
        "diff_summary": "Synthetic fallback diff (+34/-8): src/auth/jwt_verifier.py (modified, +22/-6), src/payment/circuit_breaker.py (modified, +12/-2)",
        "risk_score": 2,
        "risk_label": "medium",
        "html_url": f"https://github.com/{github_repo}",
        "adr_id": "ADR-024",
        "adr_title": "Synthetic Fallback Record: Dual-Key JWT Grace Period (24h)",
        "adr_decision": "Require 24h dual-key validation window on all JWT key rotations (synthetic fallback).",
        "adr_rationale": f"Generated via offline fallback ({fallback_reason}) because GitHub API was unreachable.",
        "decided_at_iso": now_iso[:10],
    }


def _fetch_live_github_pr(github_repo: str, pr_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Fetches real-time Pull Request and changed-file diff metadata from the GitHub REST API.
    - Raises HTTP 400 if pr_number <= 0.
    - Raises HTTP 404 if the requested PR or repository does not exist on GitHub.
    - Falls back gracefully ONLY on network/timeout errors or GitHub rate-limit/5xx errors.
    """
    if pr_number is not None and pr_number <= 0:
        raise HTTPException(status_code=400, detail="pr_number must be a positive integer.")

    now_epoch = int(time.time())
    if pr_number is not None:
        api_url = f"https://api.github.com/repos/{github_repo}/pulls/{int(pr_number)}"
    else:
        api_url = f"https://api.github.com/repos/{github_repo}/pulls?state=all&sort=updated&direction=desc&per_page=1"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "CORTEX-FalkorDB-Agent/4.0",
    }
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        if isinstance(payload, list):
            if not payload:
                raise HTTPException(
                    status_code=404,
                    detail=f"No pull requests found in GitHub repository '{github_repo}'.",
                )
            pr_obj = payload[0]
        else:
            pr_obj = payload

        if not isinstance(pr_obj, dict) or "number" not in pr_obj:
            raise HTTPException(
                status_code=404,
                detail=f"Pull request not found in GitHub repository '{github_repo}'.",
            )

        num = int(pr_obj["number"])
        title = str(pr_obj.get("title") or f"GitHub PR #{num}").strip()[:160]
        body_raw = str(pr_obj.get("body") or "").strip()
        user_obj = pr_obj.get("user") or {}
        author_login = str(user_obj.get("login") or "github-dev").strip()[:64]
        html_url = str(pr_obj.get("html_url") or f"https://github.com/{github_repo}/pull/{num}")

        # Store a merge timestamp only when GitHub confirms the PR was merged.
        is_merged = bool(pr_obj.get("merged_at"))
        if is_merged:
            merged_at_epoch, merged_at_iso = _parse_github_iso_timestamp(
                pr_obj.get("merged_at"), now_epoch
            )
        else:
            merged_at_epoch = None
            merged_at_iso = None

        # Preserve PR activity time separately; it is not a merge timestamp.
        event_raw_ts = pr_obj.get("updated_at") or pr_obj.get("created_at")
        event_timestamp_epoch, event_timestamp_iso = _parse_github_iso_timestamp(
            event_raw_ts, now_epoch
        )

        # Fetch actual changed files and diff metadata from /pulls/{num}/files
        try:
            diff_meta = _fetch_pr_files_diff_metadata(github_repo, num, headers)
        except Exception as files_exc:
            logger.warning("PR files endpoint fallback for PR-%s: %s", num, files_exc)
            fallback_count = int(pr_obj.get("changed_files") or 1)
            body_clean = body_raw[:180].replace("\r", " ").replace("\n", " ") if body_raw else title
            diff_meta = {
                "files_changed": fallback_count,
                "changed_files_list": [],
                "additions": int(pr_obj.get("additions") or 0),
                "deletions": int(pr_obj.get("deletions") or 0),
                "patch_preview": "",
                "diff_summary": f"PR summary ({fallback_count} file(s)): {body_clean}",
            }

        files_changed = diff_meta["files_changed"] or int(pr_obj.get("changed_files") or 1)
        risk_score, risk_label = _score_pr_risk(
            title,
            body_raw,
            diff_meta["changed_files_list"],
            diff_meta["patch_preview"],
        )

        return {
            "ingestion_source": "github_rest_api",
            "provenance_type": "github_pr_derived",
            "adr_status": "derived_from_pr",
            "github_repo": github_repo,
            "pr_id": f"PR-{num}",
            "title": title,
            "author_handle": author_login,
            "author_name": author_login,
            "is_merged": is_merged,
            "merged_at_epoch": merged_at_epoch,
            "merged_at_iso": merged_at_iso,
            "event_timestamp_epoch": event_timestamp_epoch,
            "event_timestamp_iso": event_timestamp_iso,
            "files_changed": files_changed,
            "changed_files_list": diff_meta["changed_files_list"],
            "diff_summary": diff_meta["diff_summary"],
            "risk_score": risk_score,
            "risk_label": risk_label,
            "html_url": html_url,
            "adr_id": f"ADR-PR-{num}",
            "adr_title": f"PR-Derived Technical Record ({github_repo} #{num})",
            "adr_decision": f"[Derived from PR-{num}] {title}",
            "adr_rationale": f"Source: {html_url} | {diff_meta['diff_summary'][:180]}",
            "decided_at_iso": event_timestamp_iso[:10],
        }

    except HTTPException:
        raise
    except urllib.error.HTTPError as http_exc:
        if http_exc.code == 404:
            target_desc = f"PR #{pr_number}" if pr_number is not None else "Repository or PR"
            raise HTTPException(
                status_code=404,
                detail=f"{target_desc} not found on GitHub for repository '{github_repo}'.",
            ) from http_exc
        if http_exc.code in (400, 422):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GitHub API request for '{github_repo}' (HTTP {http_exc.code}).",
            ) from http_exc
        if http_exc.code in (403, 429) or http_exc.code >= 500:
            logger.warning("GitHub API rate-limited/unavailable (HTTP %s), using fallback.", http_exc.code)
            return _build_offline_fallback_pr(github_repo, f"github_http_{http_exc.code}")
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected GitHub API status: HTTP {http_exc.code}.",
        ) from http_exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as net_exc:
        logger.warning("GitHub network unreachable (%s), using offline fallback.", net_exc)
        return _build_offline_fallback_pr(github_repo, f"network_{type(net_exc).__name__}")


@app.api_route("/", methods=["GET", "HEAD"])
def serve_dashboard():
    """Serves the 3-Panel SRE Command Center UI directly."""
    ui_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return FileResponse(ui_path, media_type="text/html; charset=utf-8")


@app.api_route("/vendor/{asset_name}", methods=["GET", "HEAD"])
def serve_vendor_asset(asset_name: str):
    """Serves locally cached frontend JS bundles from same-origin to eliminate third-party CDN risks."""
    try:
        asset_path = _ensure_vendor_asset(asset_name)
        return FileResponse(asset_path, media_type="application/javascript; charset=utf-8")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Vendor asset load failed (%s): %s", asset_name, exc)
        raise HTTPException(status_code=503, detail="Asset temporarily unavailable.") from exc


@app.api_route("/robots.txt", methods=["GET", "HEAD"], response_class=PlainTextResponse)
def serve_robots():
    """Prevents crawlers from indexing API routes."""
    return "User-agent: *\nDisallow: /api/\n"


@app.api_route("/sitemap.xml", methods=["GET", "HEAD"])
def serve_sitemap():
    """Returns a minimal valid XML sitemap with strict anti-cache headers."""
    xml_content = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    return Response(content=xml_content, media_type="application/xml; charset=utf-8")


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    """Verifies live FalkorDB connectivity and returns controlled 503 if unreachable."""
    try:
        db = gt.get_db()
        conn = getattr(db, "connection", None)
        if conn is not None and hasattr(conn, "ping"):
            conn.ping()
        return {"status": "healthy", "engine": "FalkorDB v6.0", "app": "CORTEX"}
    except Exception as exc:
        logger.error("Health check failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Graph database service is temporarily unavailable.",
        ) from exc


@app.post("/api/reset", dependencies=[Depends(_verify_admin_api_key)])
def reset_demo_graph():
    """Re-seeds both tenant graphs so you can re-run the P1 demo anytime."""
    try:
        db = gt.get_db()
        now_epoch = int(time.time())
        seed_graph.seed_acme_corp(db, now_epoch)
        seed_graph.seed_beta_health(db, now_epoch)
        return {"status": "reset_complete", "timestamp": now_epoch}
    except Exception as exc:
        logger.error("Failed to reset demo graph: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to reset demo graph due to database unavailability.",
        ) from exc


@app.post("/api/knowledge/ingest", dependencies=[Depends(_verify_admin_api_key)])
def ingest_live_knowledge(req: IngestKnowledgeRequest):
    """
    Track 03: Fetches real-time Pull Request and changed-file diff metadata from GitHub REST API
    (returning 404 for nonexistent PRs and falling back only on network/rate-limit errors) and merges
    the PullRequest, Person author, and provenance-tagged ArchDecisionRecord into FalkorDB.
    """
    clean_tenant = _validate_tenant_or_400(req.tenant_id)
    service_name = _validate_identifier_or_400(req.service_name, "service_name", "PaymentService")
    github_repo = _validate_github_repo_or_400(req.github_repo)

    pr_meta = _fetch_live_github_pr(github_repo, req.pr_number)

    try:
        graph = gt.get_graph(clean_tenant)
        res = graph.query(
            """
            MATCH (svc:Microservice {name: $service_name})
            MERGE (author:Person {github_handle: $author_handle})
            ON CREATE SET
              author.name = $author_name,
              author.role = 'backend',
              author.slack_handle = '@' + $author_handle,
              author.timezone = 'UTC'
            MERGE (pr:PullRequest {pr_id: $pr_id})
            SET
              pr.title = $title,
              pr.author = $author_handle,
              pr.is_merged = $is_merged,
              pr.merged_at_epoch = $merged_at_epoch,
              pr.merged_at_iso = $merged_at_iso,
              pr.event_timestamp_epoch = $event_timestamp_epoch,
              pr.event_timestamp_iso = $event_timestamp_iso,
              pr.files_changed = $files_changed,
              pr.changed_files_list = $changed_files_list,
              pr.diff_summary = $diff_summary,
              pr.risk_score = $risk_score,
              pr.risk_label = $risk_label,
              pr.permalink = $html_url
            MERGE (adr:ArchDecisionRecord {adr_id: $adr_id})
            SET
              adr.title = $adr_title,
              adr.decision = $adr_decision,
              adr.rationale = $adr_rationale,
              adr.status = $adr_status,
              adr.provenance_type = $provenance_type,
              adr.decided_at_iso = $decided_at_iso
            MERGE (pr)-[:MODIFIED]->(svc)
            MERGE (author)-[:AUTHORED]->(pr)
            MERGE (adr)-[:APPLIES_TO]->(svc)
            RETURN svc.name, author.name, pr.pr_id, adr.adr_id, adr.status, pr.merged_at_epoch
            """,
            {
                "service_name": service_name,
                "author_handle": pr_meta["author_handle"],
                "author_name": pr_meta["author_name"],
                "pr_id": pr_meta["pr_id"],
                "title": pr_meta["title"],
                "is_merged": pr_meta["is_merged"],
                "merged_at_epoch": pr_meta["merged_at_epoch"],
                "merged_at_iso": pr_meta["merged_at_iso"],
                "event_timestamp_epoch": pr_meta["event_timestamp_epoch"],
                "event_timestamp_iso": pr_meta["event_timestamp_iso"],
                "files_changed": pr_meta["files_changed"],
                "changed_files_list": pr_meta["changed_files_list"],
                "diff_summary": pr_meta["diff_summary"],
                "risk_score": pr_meta["risk_score"],
                "risk_label": pr_meta["risk_label"],
                "html_url": pr_meta["html_url"],
                "adr_id": pr_meta["adr_id"],
                "adr_title": pr_meta["adr_title"],
                "adr_decision": pr_meta["adr_decision"],
                "adr_rationale": pr_meta["adr_rationale"],
                "adr_status": pr_meta["adr_status"],
                "provenance_type": pr_meta["provenance_type"],
                "decided_at_iso": pr_meta["decided_at_iso"],
            },
        )
        if not res.result_set:
            raise HTTPException(
                status_code=404,
                detail=f"Service {service_name!r} not found in tenant_{clean_tenant}.",
            )
        row = res.result_set[0]
        return {
            "status": "ingested",
            "ingestion_source": pr_meta["ingestion_source"],
            "provenance_type": pr_meta["provenance_type"],
            "adr_status": row[4],
            "github_repo": github_repo,
            "tenant_id": clean_tenant,
            "pr_id": row[2],
            "adr_id": row[3],
            "author": row[1],
            "title": pr_meta["title"],
            "is_merged": pr_meta["is_merged"],
            "merged_at_epoch": row[5],
            "merged_at_iso": pr_meta["merged_at_iso"],
            "event_timestamp_epoch": pr_meta["event_timestamp_epoch"],
            "event_timestamp_iso": pr_meta["event_timestamp_iso"],
            "files_changed": pr_meta["files_changed"],
            "changed_files_list": pr_meta["changed_files_list"],
            "diff_summary": pr_meta["diff_summary"],
            "permalink": pr_meta["html_url"],
            "new_nodes": [row[2], row[3], row[0], row[1]],
            "message": (
                f"Synced {row[2]} ('{pr_meta['title']}') by {row[1]} from {github_repo} "
                f"[source={pr_meta['ingestion_source']}, adr_status={row[4]}] -> linked to {row[0]}."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Graph ingestion failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Graph ingestion temporarily unavailable.",
        ) from exc


@app.get("/api/graph")
def get_graph_topology(tenant_id: str = Query("acme_corp")):
    """Returns nodes, edges, and deterministic coordinates for Cytoscape.js."""
    clean_tenant = _validate_tenant_or_400(tenant_id)
    try:
        graph = gt.get_graph(clean_tenant)
        res = graph.query("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m")
    except Exception as exc:
        logger.error("Graph query failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Graph topology service is temporarily unavailable.",
        ) from exc

    nodes_map = {}
    edges_map = {}

    def _extract_node(node_obj):
        nid = str(node_obj.id)
        labels = list(node_obj.labels) if hasattr(node_obj, "labels") and node_obj.labels else ["Node"]
        cat = labels[0]
        raw_props = dict(node_obj.properties) if hasattr(node_obj, "properties") else {}
        props = {k: v for k, v in raw_props.items() if k != "error_embedding"}

        label_name = (
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
            or f"{cat}-{nid}"
        )

        return {
            "id": nid,
            "label": str(label_name),
            "category": cat,
            "status": props.get("status", "healthy"),
            "tier": props.get("tier", 0),
            "properties": props,
            "x": props.get("ui_x"),
            "y": props.get("ui_y"),
        }

    for row in res.result_set:
        n, r, m = row[0], row[1], row[2]
        if n and str(n.id) not in nodes_map:
            nodes_map[str(n.id)] = _extract_node(n)
        if m and str(m.id) not in nodes_map:
            nodes_map[str(m.id)] = _extract_node(m)
        if r:
            rel_type = getattr(r, "relation", "CONNECTED")
            src = str(getattr(r, "src_node", getattr(r, "source", "")))
            dst = str(getattr(r, "dest_node", getattr(r, "destination", "")))
            if src and dst:
                eid = f"{src}-{rel_type}-{dst}"
                edges_map[eid] = {
                    "id": eid,
                    "source": src,
                    "target": dst,
                    "label": rel_type,
                }

    return {
        "tenant_id": clean_tenant,
        "nodes": list(nodes_map.values()),
        "edges": list(edges_map.values()),
    }


@app.post("/api/incident/trigger")
def trigger_incident_stream(req: TriggerRequest):
    """Streams the 4-Agent LangGraph Incident Swarm with strict input validation and error recovery."""
    clean_tenant = _validate_tenant_or_400(req.tenant_id)
    clean_alert_id = _validate_identifier_or_400(req.alert_id, "alert_id", "ALERT-901")
    session_id = f"sess-{int(time.time() * 1000)}"

    initial_state = {
        "session_id": session_id,
        "tenant_id": clean_tenant,
        "alert_id": clean_alert_id,
        "query_text": f"Investigate alert {clean_alert_id}",
        "mode": "incident",
        "parallel": True,
        "use_pooling": True,
        "use_cache": True,
        "custom_memory_id": "",
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

    async def event_generator():
        state = initial_state
        try:
            state = orchestrator_node(state)
            yield f"data: {json.dumps({'step': 'orchestrator', 'state': state})}\n\n"
            await asyncio.sleep(0)

            state = triage_analyst_node(state)
            yield f"data: {json.dumps({'step': 'triage', 'state': state})}\n\n"
            await asyncio.sleep(0)

            state = memory_knowledge_node(state)
            yield f"data: {json.dumps({'step': 'memory', 'state': state})}\n\n"
            await asyncio.sleep(0)

            state = remediation_node(state)
            yield f"data: {json.dumps({'step': 'remediation', 'state': state})}\n\n"
            await asyncio.sleep(0)

            yield f"data: {json.dumps({'step': 'complete', 'state': state})}\n\n"
        except Exception as exc:
            logger.error("Incident stream failed: %s: %s", type(exc).__name__, exc)
            state["final_answer"] = "⚠️ Graceful Failure Recovery: Graph engine temporarily unavailable."
            yield f"data: {json.dumps({'step': 'error', 'error': 'Graph engine temporarily unavailable.', 'state': state})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat")
def trigger_chat_stream(req: ChatRequest):
    """Streams the Company Brain Q&A flow with strict tenant and query validation."""
    clean_tenant = _validate_tenant_or_400(req.tenant_id)
    query_str = (req.query or "").strip()
    if not query_str or len(query_str) > 500:
        raise HTTPException(
            status_code=400,
            detail="Query must be between 1 and 500 characters.",
        )

    session_id = f"chat-{int(time.time() * 1000)}"

    initial_state = {
        "session_id": session_id,
        "tenant_id": clean_tenant,
        "alert_id": "",
        "query_text": query_str,
        "mode": "brain",
        "parallel": True,
        "use_pooling": True,
        "use_cache": True,
        "custom_memory_id": "",
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

    async def event_generator():
        state = initial_state
        try:
            state = orchestrator_node(state)
            yield f"data: {json.dumps({'step': 'orchestrator', 'state': state})}\n\n"
            await asyncio.sleep(0)

            state = memory_knowledge_node(state)
            yield f"data: {json.dumps({'step': 'memory', 'state': state})}\n\n"
            await asyncio.sleep(0)

            state = remediation_node(state)
            yield f"data: {json.dumps({'step': 'complete', 'state': state})}\n\n"
        except Exception as exc:
            logger.error("Chat stream failed: %s: %s", type(exc).__name__, exc)
            state["final_answer"] = "⚠️ Graceful Failure Recovery: Graph engine temporarily unavailable."
            yield f"data: {json.dumps({'step': 'error', 'error': 'Graph engine temporarily unavailable.', 'state': state})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)