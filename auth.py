import hashlib
import hmac
import json
import os
import secrets

USER_DB = "users.json"

PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: str = None) -> str:
    """Return 'salt$hash' using PBKDF2-HMAC-SHA256 (stdlib only, no new dependency)."""
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return f"{salt}${derived.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored 'salt$hash' value.

    Falls back to a plain-equality check (and flags for migration) for any
    account still holding an old plaintext password, so existing accounts
    aren't locked out.
    """
    if "$" not in stored:
        # Legacy plaintext record — verify directly, caller should re-hash on success.
        return hmac.compare_digest(password, stored)

    salt, _ = stored.split("$", 1)
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, stored)


def load_users():
    if not os.path.exists(USER_DB):
        return {}
    try:
        with open(USER_DB, "r") as f:
            data = json.load(f)
            return data if data else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)


def signup(username, password, profile):
    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": _hash_password(password),
        "profile": profile
    }

    save_users(users)
    return True


def login(username, password):
    users = load_users()

    if username not in users:
        return None

    stored = users[username]["password"]
    if not _verify_password(password, stored):
        return None

    # Transparently upgrade any legacy plaintext password to a hashed one
    # the first time that account successfully logs in.
    if "$" not in stored:
        users[username]["password"] = _hash_password(password)
        save_users(users)

    return users[username]["profile"]


def update_profile(username, profile):
    users = load_users()

    if username not in users:
        return False

    users[username]["profile"] = profile
    save_users(users)
    return True