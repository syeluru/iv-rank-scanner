"""
Secure model serialization with HMAC-SHA256 verification.

Prevents arbitrary code execution from tampered pickle files by verifying
an HMAC signature before deserializing.

Signing key resolution (in order):
  1. MODEL_SIGNING_KEY environment variable
  2. models/.signing_key file (auto-generated if missing)
"""

import hashlib
import hmac
import os
import pickle
import secrets
import stat
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
KEY_FILE = MODELS_DIR / ".signing_key"


def _get_signing_key() -> bytes:
    """Get or create the HMAC signing key."""
    env_key = os.environ.get("MODEL_SIGNING_KEY")
    if env_key:
        return env_key.encode()

    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()

    # Generate a new key and restrict permissions
    key = secrets.token_bytes(32)
    KEY_FILE.write_bytes(key)
    os.chmod(KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 600
    print(f"  [model_io] Generated signing key at {KEY_FILE}")
    return key


def _compute_hmac(data: bytes) -> bytes:
    """Compute HMAC-SHA256 of data."""
    return hmac.new(_get_signing_key(), data, hashlib.sha256).digest()


def save_model(model_data: dict, model_path: Path) -> None:
    """Pickle model_data to model_path and write a .sig file alongside it."""
    model_path = Path(model_path)
    data = pickle.dumps(model_data)

    model_path.write_bytes(data)

    sig_path = model_path.with_suffix(".sig")
    sig_path.write_bytes(_compute_hmac(data))
    os.chmod(sig_path, stat.S_IRUSR | stat.S_IWUSR)  # 600


def load_model(model_path: Path) -> dict:
    """Load and verify a signed pickle model file.

    Raises ValueError if the signature is missing or invalid.
    """
    model_path = Path(model_path)
    sig_path = model_path.with_suffix(".sig")

    data = model_path.read_bytes()

    if not sig_path.exists():
        raise ValueError(
            f"No signature file found at {sig_path}. "
            f"Re-train the model to generate a signed copy, "
            f"or run: python3 -m scripts.model_io --sign {model_path}"
        )

    expected_sig = sig_path.read_bytes()
    actual_sig = _compute_hmac(data)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError(
            f"HMAC verification FAILED for {model_path}. "
            f"The model file may have been tampered with. "
            f"Re-train to generate a fresh signed model."
        )

    return pickle.loads(data)


def sign_existing(model_path: Path) -> None:
    """Sign an existing (trusted) pickle file in place."""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"  [model_io] File not found: {model_path}")
        return

    data = model_path.read_bytes()
    sig_path = model_path.with_suffix(".sig")
    sig_path.write_bytes(_compute_hmac(data))
    os.chmod(sig_path, stat.S_IRUSR | stat.S_IWUSR)
    print(f"  [model_io] Signed {model_path} -> {sig_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sign existing model files")
    parser.add_argument("--sign", nargs="+", type=Path,
                        help="Path(s) to .pkl files to sign")
    parser.add_argument("--sign-all", action="store_true",
                        help="Sign all .pkl files in models/")
    args = parser.parse_args()

    if args.sign_all:
        for pkl in MODELS_DIR.glob("*.pkl"):
            sign_existing(pkl)
    elif args.sign:
        for p in args.sign:
            sign_existing(p)
    else:
        parser.print_help()
