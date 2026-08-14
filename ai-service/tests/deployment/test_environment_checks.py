"""
tests/deployment/test_environment_checks.py
─────────────────────────────────────────────
Validates environment configuration templates, deployment scripts,
and production readiness checks without requiring a running server.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AI_ENV_EXAMPLE = REPO_ROOT / "ai-service" / ".env.example"
AI_ENV_PROD    = REPO_ROOT / "ai-service" / ".env.production"
SCRIPTS_DIR    = REPO_ROOT / "scripts"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def env_vars(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, stripping inline comments and whitespace."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            # Strip inline comments (e.g. "development          # comment")
            val = val.split("#")[0].strip()
            result[key.strip()] = val
    return result


# ═════════════════════════════════════════════════════════════════════════════
# .env.example completeness
# ═════════════════════════════════════════════════════════════════════════════

class TestEnvExample:
    """Validates ai-service/.env.example has all required variables."""

    REQUIRED_VARS = [
        "AI_SERVICE_HOST",
        "AI_SERVICE_PORT",
        "AI_SERVICE_ENV",
        "AI_SERVICE_DEBUG",
        "ALLOWED_ORIGINS",
        "ACTIVE_MODEL",
        "IMAGE_SIZE",
        "IMAGE_CHANNELS",
        "CLASS_NAMES",
        "SAVED_MODELS_DIR",
        "DATASET_RAW_DIR",
        "DATASET_PROCESSED_DIR",
        "GRADCAM_OUTPUT_DIR",
        "LOG_LEVEL",
        "LOG_DIR",
    ]

    @pytest.fixture(autouse=True)
    def load(self):
        assert AI_ENV_EXAMPLE.exists(), "ai-service/.env.example not found"
        self._vars = env_vars(AI_ENV_EXAMPLE)

    def test_all_required_vars_present(self):
        missing = [v for v in self.REQUIRED_VARS if v not in self._vars]
        assert not missing, f"Missing variables in .env.example: {missing}"

    def test_default_env_is_development(self):
        assert self._vars.get("AI_SERVICE_ENV", "").lower() == "development", \
            ".env.example should default to development environment"

    def test_default_debug_is_true(self):
        assert self._vars.get("AI_SERVICE_DEBUG", "").lower() == "true", \
            ".env.example should default AI_SERVICE_DEBUG to true (development)"

    def test_default_active_model_is_valid(self):
        valid = {"cnn", "vgg16", "resnet50", "efficientnet", "mambavision"}
        model = self._vars.get("ACTIVE_MODEL", "").lower()
        assert model in valid, f"ACTIVE_MODEL default '{model}' not in {valid}"

    def test_class_names_has_four_classes(self):
        classes = self._vars.get("CLASS_NAMES", "").split(",")
        assert len(classes) == 4, f"Expected 4 CLASS_NAMES, got {len(classes)}: {classes}"

    def test_default_image_size_is_numeric(self):
        size = self._vars.get("IMAGE_SIZE", "")
        assert size.isdigit(), f"IMAGE_SIZE must be a number, got '{size}'"


# ═════════════════════════════════════════════════════════════════════════════
# Production env template
# ═════════════════════════════════════════════════════════════════════════════

class TestProductionEnvTemplate:
    """Validates ai-service/.env.production template."""

    @pytest.fixture(autouse=True)
    def load(self):
        assert AI_ENV_PROD.exists(), "ai-service/.env.production not found"
        self._content = AI_ENV_PROD.read_text(encoding="utf-8")
        self._vars = env_vars(AI_ENV_PROD)

    def test_env_is_production(self):
        assert self._vars.get("AI_SERVICE_ENV", "").lower() == "production"

    def test_debug_is_false(self):
        assert self._vars.get("AI_SERVICE_DEBUG", "").lower() == "false", \
            "Production template must have AI_SERVICE_DEBUG=false"

    def test_jwt_secret_is_placeholder(self):
        """Production template must have a placeholder, not a real secret."""
        jwt = self._vars.get("JWT_SECRET_KEY", "")
        assert "REPLACE" in jwt or len(jwt) == 0, \
            "Production template JWT_SECRET_KEY must be a placeholder"

    def test_cors_not_wildcard(self):
        origins = self._vars.get("ALLOWED_ORIGINS", "")
        assert "*" not in origins, \
            "Production template must not use wildcard ALLOWED_ORIGINS"

    def test_auth_mode_is_authenticated(self):
        mode = self._vars.get("PREDICTION_AUTH_MODE", "public")
        assert mode == "authenticated", \
            "Production template should require authentication for predictions"

    def test_bcrypt_rounds_minimum(self):
        rounds = int(self._vars.get("BCRYPT_ROUNDS", "12"))
        assert rounds >= 12, f"Production bcrypt_rounds must be >= 12, got {rounds}"

    def test_log_level_not_debug(self):
        level = self._vars.get("LOG_LEVEL", "INFO").upper()
        assert level != "DEBUG", \
            "Production log level should not be DEBUG"

    def test_rate_limits_lower_than_dev(self):
        """Production prediction rate limit should be tighter than dev default (60)."""
        limit = int(self._vars.get("RATE_LIMIT_PREDICTION", "60"))
        assert limit <= 60, f"RATE_LIMIT_PREDICTION should be <= 60, got {limit}"

    def test_has_security_section_comment(self):
        assert "CHANGE THESE" in self._content or "REPLACE" in self._content, \
            "Production template must clearly mark secrets that need to be changed"


# ═════════════════════════════════════════════════════════════════════════════
# Deployment scripts
# ═════════════════════════════════════════════════════════════════════════════

class TestDeploymentScripts:
    """Validates deploy.sh and deploy.ps1 structure."""

    def test_deploy_sh_exists(self):
        assert (SCRIPTS_DIR / "deploy.sh").exists()

    def test_deploy_ps1_exists(self):
        assert (SCRIPTS_DIR / "deploy.ps1").exists()

    def test_backup_sh_exists(self):
        assert (SCRIPTS_DIR / "backup.sh").exists()

    def test_restore_sh_exists(self):
        assert (SCRIPTS_DIR / "restore.sh").exists()

    def test_validate_env_sh_exists(self):
        assert (SCRIPTS_DIR / "validate-env.sh").exists()

    def test_validate_env_ps1_exists(self):
        assert (SCRIPTS_DIR / "validate-env.ps1").exists()

    def test_bump_version_sh_exists(self):
        assert (SCRIPTS_DIR / "bump-version.sh").exists()

    def test_deploy_sh_has_set_e(self):
        content = (SCRIPTS_DIR / "deploy.sh").read_text(encoding="utf-8")
        assert "set -euo pipefail" in content or "set -e" in content, \
            "deploy.sh must use 'set -e' for error handling"

    def test_deploy_sh_has_rollback(self):
        content = (SCRIPTS_DIR / "deploy.sh").read_text(encoding="utf-8")
        assert "rollback" in content.lower(), "deploy.sh must support rollback"

    def test_deploy_sh_validates_env(self):
        content = (SCRIPTS_DIR / "deploy.sh").read_text(encoding="utf-8")
        assert "validate" in content.lower() or "JWT_SECRET_KEY" in content, \
            "deploy.sh must validate environment before deploying"

    def test_deploy_sh_smoke_tests(self):
        content = (SCRIPTS_DIR / "deploy.sh").read_text(encoding="utf-8")
        assert "smoke" in content.lower() or "health" in content.lower(), \
            "deploy.sh must run smoke tests after deployment"

    def test_deploy_sh_backup_step(self):
        content = (SCRIPTS_DIR / "deploy.sh").read_text(encoding="utf-8")
        assert "backup" in content.lower(), "deploy.sh must create a backup before deploying"

    def test_deploy_ps1_has_param_block(self):
        content = (SCRIPTS_DIR / "deploy.ps1").read_text(encoding="utf-8")
        assert "param(" in content or "[CmdletBinding()]" in content, \
            "deploy.ps1 must use PowerShell parameter declarations"

    def test_backup_sh_prunes_old_backups(self):
        content = (SCRIPTS_DIR / "backup.sh").read_text(encoding="utf-8")
        assert "Pruning" in content or "tail" in content, \
            "backup.sh must prune old backups to manage disk space"

    def test_restore_sh_has_safety_prompt(self):
        content = (SCRIPTS_DIR / "restore.sh").read_text(encoding="utf-8")
        assert "WARNING" in content or "Continue?" in content or "confirm" in content, \
            "restore.sh must prompt user for confirmation before overwriting data"

    def test_validate_env_checks_jwt(self):
        content = (SCRIPTS_DIR / "validate-env.sh").read_text(encoding="utf-8")
        assert "JWT_SECRET_KEY" in content, \
            "validate-env.sh must check JWT_SECRET_KEY strength"

    def test_validate_env_checks_production_flags(self):
        content = (SCRIPTS_DIR / "validate-env.sh").read_text(encoding="utf-8")
        assert "production" in content.lower(), \
            "validate-env.sh must apply stricter checks for production environment"


# ═════════════════════════════════════════════════════════════════════════════
# Project configuration files
# ═════════════════════════════════════════════════════════════════════════════

class TestProjectConfig:
    """Validates project-level configuration files."""

    def test_precommit_config_exists(self):
        assert (REPO_ROOT / ".pre-commit-config.yaml").exists()

    def test_precommit_has_ruff(self):
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "ruff" in content

    def test_precommit_has_black(self):
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "black" in content

    def test_precommit_has_prettier(self):
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "prettier" in content

    def test_precommit_has_secret_detection(self):
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "detect-secrets" in content or "detect_private_key" in content

    def test_gitignore_excludes_secrets(self):
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "docker/secrets/*.txt" in content or "secrets/*.txt" in content, \
            ".gitignore must exclude Docker secret files"

    def test_gitignore_excludes_env_files(self):
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in content

    def test_gitignore_excludes_backups(self):
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".backups/" in content

    def test_changelog_exists(self):
        assert (REPO_ROOT / "CHANGELOG.md").exists()

    def test_changelog_has_unreleased_section(self):
        content = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [Unreleased]" in content or "## Unreleased" in content

    def test_pyproject_has_ruff_config(self):
        content = (REPO_ROOT / "ai-service" / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.ruff]" in content

    def test_pyproject_has_black_config(self):
        content = (REPO_ROOT / "ai-service" / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.black]" in content

    def test_pyproject_has_isort_config(self):
        content = (REPO_ROOT / "ai-service" / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.isort]" in content

    def test_pyproject_has_coverage_config(self):
        content = (REPO_ROOT / "ai-service" / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.coverage" in content

    def test_frontend_has_eslintrc(self):
        assert (REPO_ROOT / "frontend" / ".eslintrc.cjs").exists()

    def test_frontend_has_prettierrc(self):
        assert (REPO_ROOT / "frontend" / ".prettierrc").exists()

    def test_frontend_scripts_include_format(self):
        pkg = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        scripts = pkg.get("scripts", {})
        assert "format" in scripts or "format:check" in scripts, \
            "frontend package.json must have a format script"

    def test_frontend_scripts_include_type_check(self):
        pkg = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        scripts = pkg.get("scripts", {})
        assert "type-check" in scripts or "build" in scripts, \
            "frontend must have a type-check or build (which includes tsc) script"


# ═════════════════════════════════════════════════════════════════════════════
# Version consistency
# ═════════════════════════════════════════════════════════════════════════════

class TestVersionConsistency:
    """Ensures version numbers are consistent across manifests."""

    def _get_pyproject_version(self) -> str:
        content = (REPO_ROOT / "ai-service" / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        return m.group(1) if m else ""

    def _get_frontend_version(self) -> str:
        pkg = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        return pkg.get("version", "")

    def test_pyproject_version_is_semver(self):
        version = self._get_pyproject_version()
        assert re.match(r"^\d+\.\d+\.\d+", version), \
            f"pyproject.toml version '{version}' is not semver"

    def test_frontend_version_is_semver(self):
        version = self._get_frontend_version()
        assert re.match(r"^\d+\.\d+\.\d+", version), \
            f"frontend package.json version '{version}' is not semver"
