# Knox Security Findings — Zero_DTE_Options_Strategy

**Audit Date:** 2026-03-07
**Auditor:** Knox (Security Agent)

---

## [CRITICAL] Hardcoded API Keys in .env Files
- **Files:** `.env`, `.env.development`, `.env.uat`, `.env.production`
- **Risk:** Polygon, Twelve Data, and Tradier API keys are stored in plaintext across multiple env files. If any of these files are committed to version control or exposed, all keys are compromised. Production uses the same keys as dev — no key isolation.
- **Fix:** Rotate all API keys immediately. Use a secrets manager (AWS Secrets Manager, 1Password CLI, or HashiCorp Vault). Ensure `.env*` files are in `.gitignore`. Use distinct keys per environment.
- **Status:** OPEN

## [CRITICAL] Theta Data Credentials in Plaintext (creds.txt)
- **Files:** `creds.txt`, `theta_terminal/creds.txt`
- **Risk:** Email and password for Theta Data account stored in plaintext. The `theta_terminal/creds.txt` copy was world-readable (644 permissions).
- **Fix:** Permissions on `theta_terminal/creds.txt` fixed to 600 by Knox. Credentials should still be rotated and moved to a secrets manager or at minimum loaded from environment variables.
- **Status:** PARTIAL — permissions fixed, credential rotation still needed

## [CRITICAL] Tradier Trading Account Credentials Exposed
- **Files:** `.env.development` (lines 15-16), `.env.uat` (lines 16-17)
- **Risk:** Tradier account ID and paper trading token exposed. If these files leak, the trading account could be accessed.
- **Fix:** Rotate the Tradier paper token. Move to secrets manager. Never store account IDs alongside auth tokens in the same file.
- **Status:** OPEN

## [HIGH] SQL Injection in setup_duckdb.py
- **File:** `scripts/setup_duckdb.py`
- **Lines:** 61-63, 69
- **Risk:** Table names and file paths are injected via f-strings into SQL queries (`DROP TABLE`, `CREATE TABLE`, `SELECT`). Currently table names come from a hardcoded dict, but this pattern is dangerous if inputs ever change.
- **Fix:** Use parameterized queries or validate/sanitize table names against an allowlist before string interpolation.
- **Status:** FIXED — Mason added regex validation for table names, quoted identifiers, and parameterized the file path in read_parquet()

## [HIGH] Pickle Deserialization Without Validation
- **Files:** `scripts/evaluate.py` (line 60), `scripts/score_live.py` (line 152), `scripts/shadow_trade_log.py` (lines 259, 270)
- **Risk:** `pickle.load()` can execute arbitrary code. If model files are tampered with, this enables remote code execution.
- **Fix:** Add HMAC verification before loading pickle files, or migrate to safer serialization (joblib with security settings, ONNX for models, or JSON for metadata).
- **Status:** OPEN

## [MEDIUM] Wrong Project Path in watchdog.sh
- **File:** `watchdog.sh` (line 7)
- **Risk:** References `$HOME/ironworks/Zero_DTE_Options_Strategy` instead of `$HOME/ironworks/Zero_DTE_Options_Strategy`. Script will fail silently or operate on wrong directory.
- **Fix:** Update path to `$HOME/ironworks/Zero_DTE_Options_Strategy`.
- **Status:** FIXED — Mason updated path to `$HOME/ironworks/Zero_DTE_Options_Strategy`

## [MEDIUM] Wrong User and Path in status_check.sh
- **File:** `scripts/status_check.sh` (lines 32, 45)
- **Risk:** Hardcoded path references `/Users/matto/ironworks/...` — wrong username and wrong project directory.
- **Fix:** Use `$HOME` or `$(dirname "$0")` relative paths.
- **Status:** FIXED — Mason replaced hardcoded paths with `$HOME/ironworks/...` and `os.path.expanduser()`

## [MEDIUM] No Version Pinning in requirements.txt
- **File:** `requirements.txt`
- **Risk:** Without version pinning, `pip install` could pull in a compromised or breaking version of any dependency.
- **Fix:** Pin all dependency versions (e.g., `requests==2.31.0`). Use `pip freeze` to capture current working versions.
- **Status:** FIXED — Mason pinned all 8 dependencies to current installed versions

---

## Positive Findings
- Python scripts load credentials from environment, not hardcoded in code
- `execute_trade.py` defaults to dry-run mode, requires `--execute` flag
- Production trading requires explicit "CONFIRM" input
- API calls use HTTPS
- Main `.env` and `creds.txt` have correct 600 permissions
- Environment separation (dev/uat/prod) is in place
