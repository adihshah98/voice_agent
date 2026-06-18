"""Google OAuth + JWT auth for multi-tenant API.

Flow:
  GET /auth/google          → redirect to Google consent screen
  GET /auth/google/callback → exchange code → upsert User → issue JWT → redirect to frontend
  GET /auth/me              → decode JWT → return user + org info

JWT payload: { sub: user_id, org_id, role, email, exp }

Dev mode: JWT_SECRET="" → _require_auth() returns a fake superuser principal so the
app works without credentials set up.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import logfire
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlmodel import Session, select

from voice_agent.config import settings
from voice_agent.state import DEFAULT_ORG_ID, OrgMember, Organization, User

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Scopes: email + profile (no drive, calendar, etc.)
GOOGLE_SCOPES = "openid email profile"


class AuthPrincipal(BaseModel):
    user_id: str
    org_id: str
    role: str
    email: str
    name: Optional[str] = None


# --- JWT helpers --------------------------------------------------------------


def _issue_jwt(principal: AuthPrincipal) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_expire_days)
    payload = {
        "sub": principal.user_id,
        "org_id": principal.org_id,
        "role": principal.role,
        "email": principal.email,
        "name": principal.name,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_jwt(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


# --- FastAPI dependency -------------------------------------------------------


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    # Also accept from cookie (set by callback redirect)
    return request.cookies.get("access_token")


async def require_auth(request: Request) -> AuthPrincipal:
    """FastAPI dependency: decode JWT → AuthPrincipal.

    Dev mode (JWT_SECRET=""): returns a fake admin principal so the app works
    without credentials configured.
    """
    if not settings.jwt_secret:
        # Dev mode — single implicit org
        return AuthPrincipal(
            user_id="dev-user",
            org_id=DEFAULT_ORG_ID,
            role="admin",
            email="dev@localhost",
            name="Dev User",
        )

    token = _extract_bearer(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        payload = _decode_jwt(token)
    except JWTError as e:
        logfire.warning("jwt_invalid", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return AuthPrincipal(
        user_id=payload["sub"],
        org_id=payload["org_id"],
        role=payload["role"],
        email=payload["email"],
        name=payload.get("name"),
    )


def require_admin(principal: AuthPrincipal = Depends(require_auth)) -> AuthPrincipal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return principal


# --- OAuth helpers ------------------------------------------------------------


def _google_client() -> AsyncOAuth2Client:
    return AsyncOAuth2Client(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scope=GOOGLE_SCOPES,
    )


def _callback_url(request: Request) -> str:
    base = settings.base_url or settings.webhook_url or str(request.base_url).rstrip("/")
    return f"{base}/auth/google/callback"


# --- User upsert + org membership --------------------------------------------


def _upsert_user_and_get_principal(session: Session, userinfo: dict[str, Any]) -> AuthPrincipal:
    """Create or update User from Google userinfo; find their org membership."""
    google_sub = userinfo["sub"]
    email = userinfo["email"]
    name = userinfo.get("name")
    avatar_url = userinfo.get("picture")

    user = session.exec(select(User).where(User.google_sub == google_sub)).first()
    if user is None:
        # Check if email already exists (e.g. manually provisioned)
        user = session.exec(select(User).where(User.email == email)).first()

    if user is None:
        user = User(email=email, name=name, avatar_url=avatar_url, google_sub=google_sub)
        session.add(user)
        session.flush()
        logfire.info("user_created", user_id=user.id, email=email)
    else:
        user.name = name
        user.avatar_url = avatar_url
        if user.google_sub is None:
            user.google_sub = google_sub
        session.add(user)
        session.flush()

    # Find org membership — use first org found (users may be in one org for now)
    membership = session.exec(
        select(OrgMember).where(OrgMember.user_id == user.id)
    ).first()

    if membership is None:
        # Bootstrap: if this email matches ADMIN_BOOTSTRAP_EMAIL, auto-grant admin
        # on the default org. Only fires once (no membership yet).
        bootstrap_email = settings.admin_bootstrap_email.strip().lower()
        if bootstrap_email and email.strip().lower() == bootstrap_email:
            now = datetime.now(timezone.utc)
            membership = OrgMember(
                org_id=DEFAULT_ORG_ID,
                user_id=user.id,
                role="admin",
                invited_at=now,
                joined_at=now,
            )
            session.add(membership)
            session.flush()
            logfire.info("admin_bootstrapped", user_id=user.id, email=email)
        else:
            raise HTTPException(
                status_code=403,
                detail="Your account has not been granted access to any organization. Contact your admin.",
            )

    if membership.joined_at is None:
        membership.joined_at = datetime.now(timezone.utc)
        session.add(membership)

    return AuthPrincipal(
        user_id=user.id,
        org_id=membership.org_id,
        role=membership.role,
        email=user.email,
        name=user.name,
    )
