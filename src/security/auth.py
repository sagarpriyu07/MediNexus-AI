"""
Authentication and Password Management for MediNexus AI.
"""

import os
import hashlib
import binascii
from typing import Optional, Dict, Any
from config.settings import DEMO_USERS, DEMO_PASSWORD_DEFAULT

try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (or PBKDF2 fallback with salt)."""
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    else:
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return f"pbkdf2:{binascii.hexlify(salt).decode('ascii')}:{binascii.hexlify(key).decode('ascii')}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plain password against a stored hash."""
    if not hashed or not password:
        return False
    if hashed.startswith("pbkdf2:"):
        try:
            _, salt_hex, key_hex = hashed.split(":")
            salt = binascii.unhexlify(salt_hex)
            expected_key = binascii.unhexlify(key_hex)
            computed_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
            return computed_key == expected_key
        except Exception:
            return False
    elif BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    else:
        return False


# In-memory user store with hashed passwords initialized on first load
_USER_STORE: Dict[str, Dict[str, Any]] = {}


def get_user_store() -> Dict[str, Dict[str, Any]]:
    """Retrieve initialized user store with hashed credentials."""
    global _USER_STORE
    if not _USER_STORE:
        for username, user_info in DEMO_USERS.items():
            _USER_STORE[username] = {
                "username": username,
                "name": user_info["name"],
                "email": user_info["email"],
                "role": user_info["role"],
                "department": user_info["department"],
                "password_hash": hash_password(user_info.get("plain_pwd", DEMO_PASSWORD_DEFAULT)),
            }
    return _USER_STORE


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate username and password against the secure user store."""
    users = get_user_store()
    user = users.get(username.strip().lower())
    if not user:
        return None
    if verify_password(password, user["password_hash"]):
        return {
            "username": user["username"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "department": user["department"],
        }
    return None
