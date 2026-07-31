"""
app/security/auth.py — User model, in-memory user store, and authentication logic.

For production, replace the in-memory UserStore with a database-backed
implementation (SQLite / PostgreSQL) following the same interface.

Production startup requirements
--------------------------------
The following environment variables MUST be set to non-default values before
the service will start in ``AI_SERVICE_ENV=production`` mode:

  JWT_SECRET_KEY                — strong random HS256 signing secret
  DEFAULT_ADMIN_PASSWORD        — replaces the hardcoded "Admin@123!"
  DEFAULT_RESEARCHER_PASSWORD   — replaces "Research@123!"
  DEFAULT_OPERATOR_PASSWORD     — replaces "Operator@123!"
  DEFAULT_VIEWER_PASSWORD       — replaces "Viewer@123!"

If any of the above still hold their placeholder/default values the service
will raise RuntimeError at startup and refuse to serve requests.

Classes
-------
UserInDB        — Full user record including hashed password.
UserPublic      — Safe user representation (no password hash).
UserStore       — Thread-safe in-memory user registry.

Functions
---------
authenticate_user(username, password) -> UserInDB | None
    Verify credentials, apply lockout policy, return user or None.

get_user_store() -> UserStore
    Return the singleton store instance.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.security.password import hash_password, verify_password
from app.security.roles import Role


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class UserInDB:
    """Full user record stored in the backend."""
    user_id:        str
    username:       str
    email:          str
    hashed_password: str
    role:           Role
    is_active:      bool  = True
    is_locked:      bool  = False

    # Lockout tracking
    failed_login_count: int              = 0
    locked_until:       Optional[datetime] = None

    # Password reset
    reset_token:          Optional[str]      = None
    reset_token_expires:  Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_login: Optional[datetime] = None

    def is_currently_locked(self) -> bool:
        """Return True if the account is within its lockout window."""
        if not self.is_locked:
            return False
        if self.locked_until and datetime.now(timezone.utc) >= self.locked_until:
            # Lockout window expired — auto-unlock
            self.is_locked = False
            self.failed_login_count = 0
            self.locked_until = None
            return False
        return True

    def to_public(self) -> "UserPublic":
        return UserPublic(
            user_id=self.user_id,
            username=self.username,
            email=self.email,
            role=self.role,
            is_active=self.is_active,
            is_locked=self.is_currently_locked(),
            created_at=self.created_at,
            last_login=self.last_login,
        )


@dataclass
class UserPublic:
    """User representation safe to return in API responses."""
    user_id:    str
    username:   str
    email:      str
    role:       Role
    is_active:  bool
    is_locked:  bool
    created_at: datetime
    last_login: Optional[datetime]

    def to_dict(self) -> dict:
        return {
            "user_id":    self.user_id,
            "username":   self.username,
            "email":      self.email,
            "role":       self.role.value,
            "is_active":  self.is_active,
            "is_locked":  self.is_locked,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


# ─── In-memory user store ─────────────────────────────────────────────────────

class UserStore:
    """
    Thread-safe in-memory user registry.

    Seeded on first access with a default admin account whose credentials
    come from environment variables (or hard-coded defaults for development).
    Replace with a real database adapter in production.
    """

    def __init__(self) -> None:
        self._lock:  threading.Lock             = threading.Lock()
        self._users: dict[str, UserInDB]        = {}  # username → UserInDB
        self._by_id: dict[str, UserInDB]        = {}  # user_id  → UserInDB
        self._seed_defaults()

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_defaults(self) -> None:
        """
        Create the built-in accounts on first startup.

        In development the passwords fall back to the well-known placeholder
        values defined in ``config.py``.  In production every password MUST
        be overridden via environment variables
        (``DEFAULT_ADMIN_PASSWORD``, ``DEFAULT_RESEARCHER_PASSWORD``,
        ``DEFAULT_OPERATOR_PASSWORD``, ``DEFAULT_VIEWER_PASSWORD``).

        If the service starts in production mode with any password still set to
        the committed placeholder, startup is aborted immediately — it is far
        safer to crash loudly than to run with known-weak credentials.
        """
        # ── Map each account to its configured password ───────────────────────
        default_users = [
            # (username, email, password_from_settings, role)
            ("admin",      "admin@braintumor.local",      settings.default_admin_password,      Role.ADMIN),
            ("researcher", "researcher@braintumor.local", settings.default_researcher_password, Role.RESEARCHER),
            ("operator",   "operator@braintumor.local",   settings.default_operator_password,   Role.OPERATOR),
            ("viewer",     "viewer@braintumor.local",     settings.default_viewer_password,      Role.VIEWER),
        ]

        # ── Production guard ──────────────────────────────────────────────────
        # The placeholder values that ship in source code / .env.example.
        _KNOWN_DEFAULTS = {
            "admin":      "Admin@123!",
            "researcher": "Research@123!",
            "operator":   "Operator@123!",
            "viewer":     "Viewer@123!",
        }
        if settings.ai_service_env == "production":
            weak = [
                username
                for username, _, password, _ in default_users
                if password == _KNOWN_DEFAULTS.get(username)
            ]
            if weak:
                raise RuntimeError(
                    "PRODUCTION STARTUP BLOCKED: default seed passwords have not been "
                    f"changed for the following accounts: {weak}. "
                    "Set DEFAULT_ADMIN_PASSWORD, DEFAULT_RESEARCHER_PASSWORD, "
                    "DEFAULT_OPERATOR_PASSWORD, and DEFAULT_VIEWER_PASSWORD to "
                    "strong unique secrets in your production environment before "
                    "starting the service."
                )

        for username, email, password, role in default_users:
            self._add_user_internal(username, email, password, role)

        if settings.ai_service_env == "production":
            logger.info(
                f"UserStore seeded with {len(default_users)} accounts "
                "(production passwords verified)."
            )
        else:
            logger.warning(
                f"UserStore seeded with {len(default_users)} default accounts. "
                "These accounts use development passwords — "
                "NEVER use this configuration in production."
            )

    def _add_user_internal(
        self,
        username: str,
        email: str,
        plain_password: str,
        role: Role,
    ) -> UserInDB:
        user = UserInDB(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=hash_password(plain_password),
            role=role,
        )
        self._users[username] = user
        self._by_id[user.user_id] = user
        return user

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_by_username(self, username: str) -> Optional[UserInDB]:
        with self._lock:
            return self._users.get(username)

    def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        with self._lock:
            return self._by_id.get(user_id)

    def list_users(self) -> list[UserPublic]:
        with self._lock:
            return [u.to_public() for u in self._users.values()]

    # ── Write ─────────────────────────────────────────────────────────────────

    def create_user(
        self,
        username: str,
        email: str,
        plain_password: str,
        role: Role = Role.VIEWER,
    ) -> UserInDB:
        with self._lock:
            if username in self._users:
                raise ValueError(f"Username '{username}' is already taken.")
            user = self._add_user_internal(username, email, plain_password, role)
            logger.info(f"User created: username={username} role={role.value}")
            return user

    def update_password(self, user_id: str, new_plain_password: str) -> None:
        with self._lock:
            user = self._by_id.get(user_id)
            if user is None:
                raise ValueError(f"User '{user_id}' not found.")
            user.hashed_password = hash_password(new_plain_password)
            user.reset_token         = None
            user.reset_token_expires = None
            user.failed_login_count  = 0
            user.is_locked           = False
            user.locked_until        = None
            logger.info(f"Password updated for user_id={user_id}")

    def record_failed_login(self, username: str) -> None:
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return
            user.failed_login_count += 1
            if user.failed_login_count >= settings.max_failed_logins:
                user.is_locked    = True
                user.locked_until = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=settings.lockout_minutes)
                )
                logger.warning(
                    f"Account locked: username={username} "
                    f"until={user.locked_until.isoformat()}"
                )

    def record_successful_login(self, username: str) -> None:
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return
            user.failed_login_count = 0
            user.is_locked          = False
            user.locked_until       = None
            user.last_login         = datetime.now(timezone.utc)

    def set_reset_token(self, username: str, token: str, ttl_minutes: int = 30) -> None:
        with self._lock:
            user = self._users.get(username)
            if user is None:
                raise ValueError(f"User '{username}' not found.")
            user.reset_token         = token
            user.reset_token_expires = (
                datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
            )

    def deactivate_user(self, user_id: str) -> None:
        with self._lock:
            user = self._by_id.get(user_id)
            if user:
                user.is_active = False

    def activate_user(self, user_id: str) -> None:
        with self._lock:
            user = self._by_id.get(user_id)
            if user:
                user.is_active = True

    def unlock_user(self, user_id: str) -> None:
        with self._lock:
            user = self._by_id.get(user_id)
            if user:
                user.is_locked          = False
                user.failed_login_count = 0
                user.locked_until       = None


# ─── Singleton ────────────────────────────────────────────────────────────────

_store: Optional[UserStore] = None
_store_lock = threading.Lock()


def get_user_store() -> UserStore:
    """Return (and lazily create) the global UserStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = UserStore()
    return _store


# ─── Authentication logic ─────────────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Verify *username* and *password* against the store.

    Applies the lockout policy:
      - If account is locked and the window has not expired → return None.
      - If credentials are wrong  → increment failure counter.
      - If credentials are correct → reset failure counter, record login time.

    Returns the user on success, None on failure.
    """
    store = get_user_store()
    user  = store.get_by_username(username)

    # Unknown username — don't reveal this information
    if user is None:
        logger.debug(f"authenticate_user: unknown username='{username}'")
        return None

    # Active / locked checks
    if not user.is_active:
        logger.warning(f"Login attempt on inactive account: username={username}")
        return None

    if user.is_currently_locked():
        logger.warning(
            f"Login attempt on locked account: username={username} "
            f"locked_until={user.locked_until}"
        )
        return None

    # Password verification
    if not verify_password(password, user.hashed_password):
        store.record_failed_login(username)
        logger.warning(
            f"Failed login for username={username} "
            f"attempts={user.failed_login_count}"
        )
        return None

    store.record_successful_login(username)
    logger.info(f"Successful login: username={username} role={user.role.value}")
    return user
