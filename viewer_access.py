"""Small, local HTTP viewer authentication using only the Python standard library.

This protects access on a trusted LAN; HTTP does not encrypt the code, cookie,
or video in transit. Do not forward this service to the Internet.
"""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

COOKIE_NAME = "rover_viewer"
COOKIE_TTL_SECONDS = 14 * 24 * 60 * 60
PBKDF2_ITERATIONS = 310_000
MAX_PEERS = 256
ATTEMPTS_PER_WINDOW = 5
ATTEMPT_WINDOW_SECONDS = 60
BLOCK_SECONDS = 5 * 60
CODE_FILENAME = "viewer-access-code.txt"
_URLSAFE = re.compile(r"^[A-Za-z0-9_-]+$")
_CODE = re.compile(r"^[A-HJ-NP-Z2-9]{6}(?:-[A-HJ-NP-Z2-9]{6}){2}$")


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str, length: int) -> bytes:
    if not isinstance(value, str) or len(value) > 128 or not _URLSAFE.fullmatch(value):
        raise ValueError("Invalid authentication configuration or token")
    decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    if len(decoded) != length or _encode(decoded) != value:
        raise ValueError("Invalid authentication configuration or token")
    return decoded


def _private_regular(path: Path) -> None:
    if path.is_symlink():
        raise PermissionError("Access files must not be symbolic links")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PermissionError("Access files must be regular files")
    if os.name == "posix":
        if info.st_mode & 0o077:
            raise PermissionError("Access files must be private to their owner (mode 0600 or 0400)")
        if info.st_uid != os.getuid():
            raise PermissionError("Access files must belong to the current user")


def _write_private(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial newly-created file must not masquerade as valid state.
        path.unlink(missing_ok=True)
        raise


def _initialize_path(path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.name == CODE_FILENAME:
        raise ValueError("The JSON configuration and access-code paths must differ")
    source = Path(__file__).resolve().parent
    if path.resolve().is_relative_to(source):
        raise ValueError("Store access files outside the application source directory")
    return path


def initialize(path) -> bool:
    """Create private access state once, returning True only on first creation.

    The sole plaintext code is in sibling ``viewer-access-code.txt``. Existing
    valid JSON is never rewritten, and repeating this command does not rotate
    the code or cookie-signing key. The function never prints secret values.
    """
    path = _initialize_path(path)
    if path.exists() or path.is_symlink():
        ViewerAccess(path)
        return False
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock = path.with_name(path.name + ".init.lock")
    _write_private(lock, "initialization in progress\n")
    try:
        # Recheck after acquiring the exclusive creation lock.
        if path.exists() or path.is_symlink():
            ViewerAccess(path)
            return False
        code_path = path.with_name(CODE_FILENAME)
        if code_path.exists() or code_path.is_symlink():
            # Resume safely after an interrupted first initialization without
            # replacing an already-issued code. Arbitrary weak codes are denied.
            _private_regular(code_path)
            if code_path.stat().st_size > 128:
                raise ValueError("Invalid access-code file")
            code = code_path.read_text(encoding="utf-8").strip()
            if not _CODE.fullmatch(code):
                raise ValueError("Invalid access-code file")
        else:
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
            code = "-".join("".join(secrets.choice(alphabet) for _ in range(6)) for _ in range(3))
            _write_private(code_path, code + "\n")
        salt = secrets.token_bytes(16)
        config = {
            "version": 1,
            "pbkdf2_iterations": PBKDF2_ITERATIONS,
            "salt": _encode(salt),
            "password_hash": _encode(hashlib.pbkdf2_hmac("sha256", code.encode("ascii"), salt, PBKDF2_ITERATIONS)),
            "cookie_key": _encode(secrets.token_bytes(32)),
        }
        _write_private(path, json.dumps(config, indent=2) + "\n")
        return True
    finally:
        lock.unlink(missing_ok=True)


class ViewerAccess:
    """Load existing private state; construction never initializes or mutates it.

    ``clock`` is an optional callable returning seconds, intended for tests.
    ``check_password`` returns False for both invalid codes and rate limiting.
    ``retry_after(peer)`` distinguishes an active limit for HTTP 429 handling.
    """

    def __init__(self, path, *, clock=None):
        path = Path(path).expanduser().absolute()
        _private_regular(path)
        if path.stat().st_size > 16_384:
            raise ValueError("Authentication configuration is too large")
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or type(config.get("version")) is not int or config.get("version") != 1:
            raise ValueError("Unsupported authentication configuration")
        self._iterations = config.get("pbkdf2_iterations")
        if type(self._iterations) is not int or not 100_000 <= self._iterations <= 2_000_000:
            raise ValueError("Invalid PBKDF2 iteration count")
        self._salt = _decode(config.get("salt"), 16)
        self._hash = _decode(config.get("password_hash"), 32)
        self._cookie_key = _decode(config.get("cookie_key"), 32)
        self._clock = clock if clock is not None else time.time
        self._rate_clock = clock if clock is not None else time.monotonic
        self._peers = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _peer_key(peer) -> str:
        return str(peer)[:128]

    def retry_after(self, peer) -> int:
        with self._lock:
            entry = self._peers.get(self._peer_key(peer))
            if entry is None:
                return 0
            return max(0, math.ceil(entry[2] - self._rate_clock()))

    def check_password(self, code, peer) -> bool:
        key, now = self._peer_key(peer), self._rate_clock()
        with self._lock:
            entry = self._peers.get(key)
            if entry is not None and entry[2] > now:
                self._peers.move_to_end(key)
                return False
            if entry is None or now - entry[0] >= ATTEMPT_WINDOW_SECONDS:
                entry = [now, 0, 0.0]
                self._peers[key] = entry
            entry[1] += 1
            if entry[1] >= ATTEMPTS_PER_WINDOW:
                entry[2] = now + BLOCK_SECONDS
            self._peers.move_to_end(key)
            while len(self._peers) > MAX_PEERS:
                self._peers.popitem(last=False)
        # Bound all input handling before the expensive password derivation.
        valid = isinstance(code, str) and len(code) <= 128
        try:
            encoded = code.strip().encode("utf-8") if valid else b""
        except UnicodeError:
            encoded, valid = b"", False
        derived = hashlib.pbkdf2_hmac("sha256", encoded, self._salt, self._iterations)
        accepted = hmac.compare_digest(derived, self._hash) and valid
        if accepted:
            with self._lock:
                self._peers.pop(key, None)
        return accepted

    def issue_cookie(self) -> str:
        """Return a complete Set-Cookie value for a new 14-day session."""
        issued = int(self._clock())
        expires = issued + COOKIE_TTL_SECONDS
        body = f"v1.{issued}.{expires}.{secrets.token_urlsafe(16)}"
        signature = _encode(hmac.new(self._cookie_key, body.encode("ascii"), hashlib.sha256).digest())
        return (f"{COOKIE_NAME}={body}.{signature}; Path=/; Max-Age={COOKIE_TTL_SECONDS}; "
                "HttpOnly; SameSite=Strict")

    def authenticated(self, cookie_header) -> bool:
        if not isinstance(cookie_header, str) or len(cookie_header) > 8192:
            return False
        try:
            parsed = SimpleCookie()
            parsed.load(cookie_header)
            morsel = parsed.get(COOKIE_NAME)
            if morsel is None or len(morsel.value) > 512:
                return False
            version, issued_text, expires_text, nonce, signature = morsel.value.split(".")
            if version != "v1" or not issued_text.isdigit() or not expires_text.isdigit():
                return False
            issued, expires = int(issued_text), int(expires_text)
            if str(issued) != issued_text or str(expires) != expires_text:
                return False
            _decode(nonce, 16)
            supplied = _decode(signature, 32)
            body = ".".join((version, issued_text, expires_text, nonce))
            expected = hmac.new(self._cookie_key, body.encode("ascii"), hashlib.sha256).digest()
            if not hmac.compare_digest(expected, supplied):
                return False
            now = self._clock()
            return issued <= now + 60 and expires > now and expires - issued == COOKIE_TTL_SECONDS
        except (ValueError, TypeError, CookieError, UnicodeError):
            return False

    def login_page(self, error=False) -> bytes:
        """Local, self-contained login form; no submitted values are reflected."""
        message = ('<p class="error" role="alert">No se pudo entrar. Revisa el código '
                   'o espera unos minutos para volver a intentarlo.</p>') if error else ""
        return ('''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visor del rover</title>
<style>body{margin:0;background:#eef3ef;color:#173626;font:17px system-ui,sans-serif;display:grid;min-height:100vh;place-items:center}main{box-sizing:border-box;width:min(92vw,420px);background:white;padding:30px;border-radius:18px;box-shadow:0 8px 35px #16372818}h1{margin-top:0;font-size:27px}p{line-height:1.5}label{display:block;font-weight:600;margin-bottom:8px}input,button{box-sizing:border-box;width:100%;padding:13px;font:inherit;border-radius:9px}input{border:1px solid #81998a}button{border:0;background:#216642;color:white;font-weight:600;margin-top:16px;cursor:pointer}.error{color:#8b2525}</style>
<main><h1>Visor del rover</h1><p>Introduce el código de acceso de la Raspberry. Este navegador recordará el acceso durante 14 días.</p>'''
                + message + '''<form method="post" action="/login"><label for="code">Código de acceso</label><input id="code" name="code" type="password" autocomplete="current-password" spellcheck="false" maxlength="128" required autofocus><button type="submit">Entrar</button></form></main></html>''').encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", metavar="JSON_PATH", required=True,
                        help="Create private access configuration outside the application directory")
    args = parser.parse_args()
    try:
        created = initialize(args.init)
    except (OSError, ValueError):
        # Never echo config contents, passwords, or exception data containing
        # private material. CLI errors retain a clear nonzero exit status.
        parser.exit(1, "Unable to initialize private access files. Check the path, ownership, and permissions.\n")
    print("Access initialized." if created else "Existing access preserved.")
    print("The access code is stored in the private sibling viewer-access-code.txt file.")


if __name__ == "__main__":
    main()
