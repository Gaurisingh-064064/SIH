from __future__ import annotations
import hashlib
import io
import itertools
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from app.blockchain import blockchain


# Keep NLP lazy: importing the NLP module may load spaCy/model data and block startup.
def extract_entities(text: str):
    from .nlp import extract_entities as _extract_entities

    return _extract_entities(text)


from .security import hash_link, sha256_json, utc_now
from .supabase_client import supabase

app = FastAPI(
    title="NyayaNet — AI-Powered Criminal Network Analysis API",
    version="1.1.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_check():
    print("\n==============================")
    print("NYAYANET BACKEND STARTING")
    print("==============================")
    print("Relationship model path:", RELATIONSHIP_MODEL_PATH)
    print("Model exists:", RELATIONSHIP_MODEL_PATH.exists())
    print("==============================\n")


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT/"backend"/ "ml"

RELATIONSHIP_MODEL_PATH = MODEL_DIR / "relationship_model.joblib"
# ANOMALY_MODEL_PATH = MODEL_DIR / "suspicious_pattern_model.joblib"

# Set DEBUG_RELATIONSHIP_MODEL=true in the environment to print raw feature
# values / predict_proba output for every relationship scored. Off by
# default so normal runs don't flood the logs.
DEBUG_RELATIONSHIP_MODEL = (
    os.getenv("DEBUG_RELATIONSHIP_MODEL", "false").lower() == "true"
)


def load_model(path: Path):
    """Load a model only when inference actually needs it.

    Keeping joblib/model deserialization out of module import prevents the
    FastAPI process from blocking before Application startup complete.
    """
    try:
        import joblib

        if not path.exists():
            print(f"Model file not found: {path}")
            return None
        return joblib.load(path)
    except Exception as exc:
        print(f"Model load failed for {path}: {exc}")
        return None


RELATIONSHIP_MODEL = None
# ANOMALY_MODEL = None


def verify_relationship_model() -> bool:
    """Lazily load the relationship model and log what it expects.

    Does not force eager loading at import time on its own; only runs when
    called (e.g. from the startup hook below), and any failure is logged
    rather than raised.
    """
    global RELATIONSHIP_MODEL

    if RELATIONSHIP_MODEL is None:
        RELATIONSHIP_MODEL = load_model(RELATIONSHIP_MODEL_PATH)

    if RELATIONSHIP_MODEL is None:
        print("WARNING: Relationship model could not be loaded.")
        return False

    try:
        feature_count = getattr(RELATIONSHIP_MODEL, "n_features_in_", None)
        print("Relationship model loaded successfully.")
        print(f"Model expects {feature_count} features.")
        return True
    except Exception as exc:
        print("Model verification failed:", exc)
        return False


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class InvestigationCreate(BaseModel):
    title: str
    description: Optional[str] = None


class SourceInput(BaseModel):
    source_type: str = ""
    title: Optional[str] = None
    content: str = ""
    language: str = "en"


class InvestigationAnalyzeRequest(BaseModel):
    sources: List[SourceInput] = Field(default_factory=list)


class DocumentIn(BaseModel):
    investigation_id: str
    source_type: str
    title: str
    content: str
    language: str = "en"


class LinkIn(BaseModel):
    investigation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: List[str] = Field(default_factory=list)


class TipIn(BaseModel):
    investigation_id: str
    text: str


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------


def require_user(authorization: Optional[str]) -> str:
    if supabase is None:
        raise HTTPException(500, "Supabase not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Access token required")

    token = authorization.split(" ", 1)[1]
    try:
        user = supabase.auth.get_user(token).user
    except Exception as exc:
        print("Token validation failed:", exc)
        raise HTTPException(401, "Invalid token")

    profile = (
        supabase.table("profiles").select("is_authorized").eq("id", user.id).execute()
    )
    if not profile.data or not profile.data[0]["is_authorized"]:
        raise HTTPException(403, "User not authorized")
    return user.id


# Verify that the authenticated investigator owns the requested investigation.
# The backend uses the Supabase service key, so database RLS is bypassed for
# these server-side queries; ownership must therefore also be enforced here.
def require_investigation_owner(investigation_id: str, user_id: str):
    investigation = (
        supabase.table("investigations")
        .select("id, created_by")
        .eq("id", investigation_id)
        .maybe_single()
        .execute()
    )

    if not investigation.data:
        raise HTTPException(404, "Investigation not found")

    created_by = investigation.data.get("created_by")
    if created_by != user_id:
        # Return the same response an absent case would use so another
        # investigator cannot discover whether a case ID exists.
        raise HTTPException(404, "Investigation not found")

    return investigation.data


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def clean_phone(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    phone = re.sub(r"[\s-]", "", str(value))
    if phone.startswith("+91"):
        phone = phone[3:]
    return phone


def generate_evidence_hash(content: str) -> str:
    """
    Generate a SHA-256 hash for evidence/source content.

    The original evidence is NOT stored in the blockchain.
    Only this cryptographic fingerprint will be recorded.
    """

    if content is None:
        content = ""

    normalized_content = str(content).encode("utf-8")

    return hashlib.sha256(normalized_content).hexdigest()


def generate_evidence_id(
    investigation_id: str,
    source_type: str,
    content: str,
) -> str:
    """
    Generate a deterministic unique evidence identifier.
    """

    evidence_input = f"{investigation_id}|" f"{source_type}|" f"{content}"

    evidence_hash = hashlib.sha256(evidence_input.encode("utf-8")).hexdigest()

    return f"EV-{evidence_hash[:12].upper()}"


# -----------------------------------------------------------------------------
# BLOCKCHAIN EVIDENCE LEDGER
# -----------------------------------------------------------------------------

# Temporary in-memory fallback blockchain storage.
# Supabase is used when the investigation_blockchain table is available.
BLOCKCHAIN_EVIDENCE_LEDGER: Dict[str, List[Dict[str, Any]]] = {}


def calculate_block_hash(block_data: Dict[str, Any]) -> str:
    """
    Generate SHA-256 hash for a blockchain block.
    """

    encoded_data = json.dumps(
        block_data,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded_data).hexdigest()


def get_investigation_chain(
    investigation_id: str,
) -> List[Dict[str, Any]]:
    """
    Get the blockchain ledger for an investigation.

    First tries Supabase.
    Falls back to in-memory storage if the database table
    is unavailable.
    """

    try:
        result = (
            supabase.table("investigation_blockchain")
            .select("*")
            .eq(
                "investigation_id",
                investigation_id,
            )
            .order("block_index")
            .execute()
        )

        chain = result.data or []

        if chain:
            return chain

    except Exception as exc:
        print(
            "Blockchain database load failed:",
            exc,
        )

    # Fallback to memory

    if investigation_id not in BLOCKCHAIN_EVIDENCE_LEDGER:
        BLOCKCHAIN_EVIDENCE_LEDGER[investigation_id] = []

    return BLOCKCHAIN_EVIDENCE_LEDGER[investigation_id]


def create_evidence_block(
    investigation_id: str,
    source_type: str,
    title: str,
    content: str,
) -> Dict[str, Any]:
    """
    Create an immutable blockchain block for evidence.

    Original evidence content is NOT stored
    in the blockchain.

    Only metadata and SHA-256 fingerprint
    are stored.
    """

    # -------------------------------------------------
    # GENERATE EVIDENCE HASH
    # -------------------------------------------------

    evidence_hash = generate_evidence_hash(content)

    evidence_id = generate_evidence_id(
        investigation_id,
        source_type,
        content,
    )

    # -------------------------------------------------
    # GET EXISTING BLOCKCHAIN
    # -------------------------------------------------

    chain = get_investigation_chain(investigation_id)

    # -------------------------------------------------
    # PREVENT DUPLICATE BLOCKS
    # -------------------------------------------------

    for existing_block in chain:

        if existing_block.get("evidence_id") == evidence_id:
            return existing_block

    # -------------------------------------------------
    # DETERMINE BLOCK INDEX
    # -------------------------------------------------

    if len(chain) == 0:

        block_index = 0

        previous_hash = "0" * 64

    else:

        last_block = chain[-1]

        block_index = (
            int(
                last_block.get(
                    "block_index",
                    last_block.get(
                        "index",
                        len(chain) - 1,
                    ),
                )
            )
            + 1
        )

        previous_hash = last_block.get("block_hash") or "0" * 64

    # -------------------------------------------------
    # CREATE BLOCK DATA
    # -------------------------------------------------

    timestamp = utc_now()

    block = {
        "block_index": block_index,
        "timestamp": timestamp,
        "investigation_id": investigation_id,
        "evidence_id": evidence_id,
        "source_type": (source_type or "").upper(),
        "title": (title or "Evidence"),
        "evidence_hash": evidence_hash,
        "previous_hash": previous_hash,
    }

    # -------------------------------------------------
    # GENERATE BLOCK HASH
    # -------------------------------------------------

    block_hash = calculate_block_hash(block)

    block["block_hash"] = block_hash

    # -------------------------------------------------
    # SAVE BLOCK TO SUPABASE
    # -------------------------------------------------

    database_saved = False

    try:

        blockchain_payload = {
            "investigation_id": investigation_id,
            "block_index": block_index,
            "evidence_id": evidence_id,
            "source_type": (source_type or "").upper(),
            "title": (title or "Evidence"),
            "evidence_hash": evidence_hash,
            "previous_hash": previous_hash,
            "block_hash": block_hash,
            "timestamp": timestamp,
        }

        (
            supabase.table("investigation_blockchain")
            .upsert(
                blockchain_payload,
                on_conflict=("investigation_id," "evidence_id"),
            )
            .execute()
        )

        database_saved = True

        print("Blockchain block saved to Supabase")

    except Exception as exc:

        print(
            "Blockchain persistence failed:",
            exc,
        )

    # -------------------------------------------------
    # FALLBACK MEMORY STORAGE
    # -------------------------------------------------

    if not database_saved:

        if investigation_id not in BLOCKCHAIN_EVIDENCE_LEDGER:
            BLOCKCHAIN_EVIDENCE_LEDGER[investigation_id] = []

        memory_chain = BLOCKCHAIN_EVIDENCE_LEDGER[investigation_id]

        duplicate_found = False

        for existing_block in memory_chain:

            if existing_block.get("evidence_id") == evidence_id:
                duplicate_found = True
                break

        if not duplicate_found:

            memory_chain.append(block)

    # -------------------------------------------------
    # LOG BLOCK CREATION
    # -------------------------------------------------

    print("\n==============================")

    print("BLOCKCHAIN EVIDENCE ADDED")

    print("==============================")

    print(
        "Investigation ID:",
        investigation_id,
    )

    print(
        "Block Index:",
        block_index,
    )

    print(
        "Evidence ID:",
        evidence_id,
    )

    print(
        "Evidence Hash:",
        evidence_hash,
    )

    print(
        "Previous Hash:",
        previous_hash,
    )

    print(
        "Block Hash:",
        block_hash,
    )

    print(
        "Database Saved:",
        database_saved,
    )

    print("==============================\n")

    # IMPORTANT

    return block


def verify_blockchain_integrity(
    investigation_id: str,
) -> Dict[str, Any]:
    """
    Verify the complete blockchain evidence chain.

    Checks:
    1. Previous hash linkage
    2. Block hash integrity
    3. Block order
    """

    try:
        response = (
            supabase.table("investigation_blockchain")
            .select("*")
            .eq(
                "investigation_id",
                investigation_id,
            )
            .order("block_index")
            .execute()
        )

        blocks = response.data or []

    except Exception as exc:

        print(
            "Blockchain verification database error:",
            exc,
        )

        return {
            "valid": False,
            "message": "Unable to load blockchain records",
            "total_blocks": 0,
            "verified_blocks": 0,
            "invalid_blocks": [],
        }

    if not blocks:

        return {
            "valid": True,
            "message": "No blockchain evidence records found",
            "total_blocks": 0,
            "verified_blocks": 0,
            "invalid_blocks": [],
        }
        # ------------------------------------------
    # LOAD CURRENT INVESTIGATION SOURCES
    # ------------------------------------------

    try:

        sources_response = (
            supabase.table("investigation_sources")
            .select("source_type, title, content")
            .eq(
                "investigation_id",
                investigation_id,
            )
            .execute()
        )

        current_sources = sources_response.data or []

    except Exception as exc:

        print(
            "Evidence source load failed:",
            exc,
        )

        current_sources = []

    # ------------------------------------------
    # CREATE SOURCE LOOKUP
    # ------------------------------------------

    sources_by_type: Dict[str, List[Dict[str, Any]]] = {}

    for source in current_sources:

        source_type = (
            source.get(
                "source_type",
                "",
            )
            .strip()
            .upper()
        )

        if source_type:

            sources_by_type.setdefault(source_type, []).append(source)

    invalid_blocks = []

    verified_blocks = 0

    expected_previous_hash = "0" * 64

    expected_block_index = 0

    for block in blocks:

        block_index = int(
            block.get(
                "block_index",
                0,
            )
        )

        evidence_id = block.get(
            "evidence_id",
            "",
        )

        evidence_hash = block.get(
            "evidence_hash",
            "",
        )

        previous_hash = block.get(
            "previous_hash",
            "",
        )

        stored_block_hash = block.get(
            "block_hash",
            "",
        )

        timestamp = block.get(
            "timestamp",
            "",
        )

        source_type = block.get(
            "source_type",
            "",
        )

        title = block.get(
            "title",
            "Evidence",
        )

        errors = []
        # ------------------------------------------
        # 3. VERIFY ACTUAL EVIDENCE CONTENT
        # ------------------------------------------

        normalized_source_type = source_type.strip().upper()

        current_source_candidates = sources_by_type.get(normalized_source_type, [])

        if not current_source_candidates:

            errors.append("Original evidence source is missing")

        else:

            # A source_type can have more than one evidence document (e.g. a
            # re-uploaded/edited source, or two documents of the same
            # category). Only flag tampering if NONE of the current sources
            # of this type still hash to what was recorded on-chain for this
            # block — matching against just the most-recently-loaded source
            # produced false "modified" flags for untouched evidence.
            match_found = any(
                generate_evidence_hash(candidate.get("content", "") or "")
                == evidence_hash
                for candidate in current_source_candidates
            )

            if not match_found:

                errors.append(
                    "Evidence content has been modified "
                    "after blockchain registration"
                )

        # ------------------------------------------
        # 1. VERIFY BLOCK INDEX
        # ------------------------------------------

        if block_index != expected_block_index:

            errors.append("Block index sequence is invalid")

        # ------------------------------------------
        # 2. VERIFY PREVIOUS HASH
        # ------------------------------------------

        if previous_hash != expected_previous_hash:

            errors.append("Previous hash does not match " "the preceding block")

        # ------------------------------------------
        # 3. RECREATE EXACT ORIGINAL BLOCK
        # ------------------------------------------

        original_block_data = {
            "block_index": block_index,
            "timestamp": timestamp,
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "source_type": (source_type or "").upper(),
            "title": (title or "Evidence"),
            "evidence_hash": evidence_hash,
            "previous_hash": previous_hash,
        }

        # ------------------------------------------
        # 4. RECALCULATE BLOCK HASH
        # ------------------------------------------

        recalculated_block_hash = calculate_block_hash(original_block_data)

        if recalculated_block_hash != stored_block_hash:

            errors.append("Block hash integrity check failed")

        # ------------------------------------------
        # BLOCK RESULT
        # ------------------------------------------

        if errors:

            invalid_blocks.append(
                {
                    "block_index": block_index,
                    "evidence_id": evidence_id,
                    "source_type": source_type,
                    "title": title,
                    "errors": errors,
                }
            )

        else:

            verified_blocks += 1

        # ------------------------------------------
        # PREPARE NEXT BLOCK CHECK
        # ------------------------------------------

        expected_previous_hash = stored_block_hash

        expected_block_index += 1

        # ------------------------------------------
        # FINAL RESULT
        # ------------------------------------------

        tampered_blocks = []

    for block in invalid_blocks:

        tampering_errors = [
            error
            for error in block.get("errors", [])
            if ("modified" in error.lower() or "missing" in error.lower())
        ]

        if tampering_errors:

            tampered_blocks.append(
                {
                    "block_index": block.get("block_index"),
                    "evidence_id": block.get("evidence_id"),
                    "source_type": block.get("source_type"),
                    "title": block.get("title"),
                    "errors": tampering_errors,
                }
            )

    return {
        "valid": (len(invalid_blocks) == 0),
        "message": (
            "Blockchain and evidence integrity " "verified successfully"
            if len(invalid_blocks) == 0
            else "Blockchain or evidence integrity " "violation detected"
        ),
        "total_blocks": len(blocks),
        "verified_blocks": verified_blocks,
        "invalid_blocks": invalid_blocks,
        "tampering_detected": (len(tampered_blocks) > 0),
        "tampered_blocks": tampered_blocks,
    }


def verify_evidence_integrity(
    investigation_id: str,
    evidence_id: str,
    content: str,
) -> Dict[str, Any]:
    """
    Verify evidence content against the blockchain.

    Also verifies the complete blockchain chain.
    """

    # -------------------------------------------------
    # GET BLOCKCHAIN
    # -------------------------------------------------

    chain = get_investigation_chain(investigation_id)

    evidence_block = None

    # -------------------------------------------------
    # FIND EVIDENCE BLOCK
    # -------------------------------------------------

    for block in chain:

        if block.get("evidence_id") == evidence_id:

            evidence_block = block

            break

    # -------------------------------------------------
    # BLOCK NOT FOUND
    # -------------------------------------------------

    if evidence_block is None:

        return {
            "found": False,
            "valid": False,
            "tampered": None,
            "chain_valid": False,
            "message": ("Evidence not found " "in blockchain."),
        }

    # -------------------------------------------------
    # VERIFY EVIDENCE HASH
    # -------------------------------------------------

    current_hash = generate_evidence_hash(content)

    stored_hash = evidence_block.get("evidence_hash")

    evidence_valid = current_hash == stored_hash

    # -------------------------------------------------
    # VERIFY COMPLETE BLOCKCHAIN
    # -------------------------------------------------

    chain_valid = True

    previous_hash = "0" * 64

    for block in chain:

        block_index = block.get(
            "block_index",
            block.get("index"),
        )

        # Recreate EXACT original block data
        # used while generating the block hash

        block_data = {
            "block_index": block_index,
            "timestamp": block.get("timestamp"),
            "investigation_id": block.get("investigation_id"),
            "evidence_id": block.get("evidence_id"),
            "source_type": block.get("source_type"),
            "title": block.get("title"),
            "evidence_hash": block.get("evidence_hash"),
            "previous_hash": block.get("previous_hash"),
        }

        expected_hash = calculate_block_hash(block_data)

        # Verify previous hash connection

        if block.get("previous_hash") != previous_hash:

            chain_valid = False

            break

        # Verify block hash

        if block.get("block_hash") != expected_hash:

            chain_valid = False

            break

        previous_hash = block.get("block_hash")

    # -------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------

    final_valid = evidence_valid and chain_valid

    return {
        "found": True,
        "valid": final_valid,
        "tampered": (not evidence_valid),
        "chain_valid": chain_valid,
        "evidence_id": evidence_id,
        "stored_evidence_hash": stored_hash,
        "current_evidence_hash": current_hash,
        "block_hash": evidence_block.get("block_hash"),
        "previous_hash": evidence_block.get("previous_hash"),
        "message": (
            "Evidence integrity verified successfully."
            if final_valid
            else ("Evidence integrity check failed. " "Possible tampering detected.")
        ),
    }


INVALID_PERSON_TERMS = {
    "unknown",
    "person",
    "persons",
    "observed",
    "suspect",
    "accused",
    "witness",
    "complainant",
    "victim",
    "officer",
    "police",
    "report",
    "database",
    "history",
    "records",
    "transaction",
    "transactions",
    "surveillance",
    "intelligence",
    "financial",
    "social media",
    "warehouse",
    "zone",
    "location",
    "area",
    "city",
    "district",
    "station",
}


def extract_valid_people(person_names: List[str]) -> List[str]:
    """
    Filter a raw PERSON entity list down to plausible human names —
    drop short fragments, generic/role words, and anything with no
    alphabetic characters. Mirrors the person-filtering rules used
    when building profiles in analyze_sources().
    """
    valid = []
    seen = set()
    for person_name in person_names or []:
        key = normalize_text(person_name)
        if not key or len(key) < 3:
            continue
        if key in INVALID_PERSON_TERMS:
            continue
        if not any(char.isalpha() for char in key):
            continue
        if key in seen:
            continue
        seen.add(key)
        valid.append(person_name)
    return valid


INVALID_PERSON_TERMS = {
    "unknown",
    "person",
    "persons",
    "observed",
    "suspect",
    "accused",
    "witness",
    "complainant",
    "victim",
    "officer",
    "police",
    "report",
    "database",
    "history",
    "records",
    "transaction",
    "transactions",
    "surveillance",
    "intelligence",
    "financial",
    "social media",
    "warehouse",
    "zone",
    "location",
    "area",
    "city",
    "district",
    "station",
}


def extract_valid_people(person_names: List[str]) -> List[str]:
    """
    Filter a raw PERSON entity list down to plausible human names —
    drop short fragments, generic/role words, and anything with no
    alphabetic characters. Mirrors the person-filtering rules used
    when building profiles in analyze_sources().
    """
    valid = []
    seen = set()
    for person_name in person_names or []:
        key = normalize_text(person_name)
        if not key or len(key) < 3:
            continue
        if key in INVALID_PERSON_TERMS:
            continue
        if not any(char.isalpha() for char in key):
            continue
        if key in seen:
            continue
        seen.add(key)
        valid.append(person_name)
    return valid


def normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def extract_text_from_pdf(file_obj) -> str:
    """Extract text content from an uploaded text-based PDF."""
    try:
        reader = PdfReader(file_obj)
        extracted_text: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_text.append(page_text.strip())

        text = "\n".join(extracted_text).strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found in this PDF. "
                    "Scanned/image-only PDFs need OCR support."
                ),
            )
        return text
    except HTTPException:
        raise
    except Exception as exc:
        print("PDF extraction failed:", exc)
        raise HTTPException(
            status_code=400,
            detail="Unable to read text from the PDF file.",
        )


def extract_uploaded_source_text(filename: str, content_type: str, raw: bytes) -> str:
    """Return extracted text for a supported TXT or PDF upload."""
    suffix = Path(filename or "").suffix.lower()

    if suffix == ".pdf" or content_type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(raw))

    if suffix in {".txt", ".text", ".csv", ".log"} or content_type.startswith("text/"):
        for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
            try:
                text = raw.decode(encoding).strip()
                if text:
                    return text
            except UnicodeDecodeError:
                continue
        raise HTTPException(400, "Unable to decode the uploaded text file.")

    raise HTTPException(
        status_code=400,
        detail="Only TXT and PDF files are supported.",
    )


def get_next_person_id() -> str:
    if supabase is None:
        return "P0001"
    response = (
        supabase.table("persons")
        .select("person_id")
        .order("person_id", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return "P0001"
    last = response.data[0].get("person_id", "")
    match = re.search(r"(\d+)$", last)
    return f"P{int(match.group(1)) + 1:04d}" if match else "P0001"


def first_money_values(text: str) -> List[float]:
    values: List[float] = []
    patterns = [
        r"(?:INR|Rs\.?|₹)\s*([0-9,]+(?:\.\d+)?)",
        r"amount\s*[:=]\s*([0-9,]+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            try:
                values.append(float(match.group(1).replace(",", "")))
            except ValueError:
                continue
    return values


def count_numeric(patterns: List[str], text: str) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return 0


def source_activity(source_type: str, text: str) -> Dict[str, float]:
    source = source_type.upper()

    calls = count_numeric(
        [
            r"(?:phone\s+)?calls?\s*[:=]\s*(\d+)",
            r"(\d+)\s+(?:phone\s+)?calls?",
        ],
        text,
    )
    transactions = count_numeric(
        [
            r"transactions?\s*[:=]\s*(\d+)",
            r"(\d+)\s+transactions?",
        ],
        text,
    )
    meetings = count_numeric(
        [
            r"meetings?\s*[:=]\s*(\d+)",
            r"(\d+)\s+meetings?",
        ],
        text,
    )

    if source == "CDR" and calls == 0:
        # Count explicit CDR rows when the source has caller/receiver records.
        calls = len(re.findall(r"caller\s*:", text, flags=re.I))
        if calls == 0 and text.strip():
            calls = 1

    if source == "FINANCIAL" and transactions == 0:
        transactions = len(re.findall(r"transaction\s*\d+", text, flags=re.I))
        if transactions == 0 and text.strip():
            transactions = 1

    if source == "SURVEILLANCE" and meetings == 0:
        meetings = len(
            re.findall(r"(?:\d{1,2}:\d{2}|meeting|observed meeting)", text, flags=re.I)
        )
        meetings = min(meetings, 20)
        if meetings == 0 and text.strip():
            meetings = 1

    money = first_money_values(text)
    total_amount = sum(money) if source == "FINANCIAL" else max(money) if money else 0.0

    duration_values = [
        int(x)
        for x in re.findall(
            r"duration\s*[:=]\s*(\d+)\s*(?:seconds?|sec)?", text, flags=re.I
        )
    ]
    total_duration = sum(duration_values)

    return {
        "calls": float(calls),
        "duration": float(total_duration),
        "transactions": float(transactions),
        "amount": float(total_amount),
        "meetings": float(meetings),
    }


# -----------------------------------------------------------------------------
# ML feature generation
# -----------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:

    try:

        if value is None or value == "":
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def relationship_features(record: Dict[str, Any]) -> List[float]:
    """
    Create the exact 6 features used by the current
    Random Forest relationship model.
    """

    calls = float(record.get("phone_call_count") or record.get("calls") or 0)

    duration = float(
        record.get("total_call_duration_sec") or record.get("duration") or 0
    )

    transactions = float(
        record.get("transaction_count") or record.get("transactions") or 0
    )

    amount = float(record.get("total_transaction_amount") or record.get("amount") or 0)

    meetings = float(record.get("meeting_count") or record.get("meetings") or 0)

    source_diversity = float(record.get("source_diversity") or 0)

    return [
        math.log1p(calls),
        math.log1p(duration),
        math.log1p(transactions),
        math.log1p(amount),
        math.log1p(meetings),
        source_diversity,
    ]


def predict_relationship(record: Dict[str, Any]) -> float:

    global RELATIONSHIP_MODEL

    values = relationship_features(record)

    model_score = None

    if RELATIONSHIP_MODEL is None:
        RELATIONSHIP_MODEL = load_model(RELATIONSHIP_MODEL_PATH)

    if RELATIONSHIP_MODEL is not None:
        try:
            feature_names = [
                "log_calls",
                "log_call_duration",
                "log_transactions",
                "log_transaction_amount",
                "log_meetings",
                "source_diversity",
            ]

            X = pd.DataFrame(
                [values],
                columns=feature_names,
            )

            model_score = float(RELATIONSHIP_MODEL.predict_proba(X)[0][1])

        except Exception as exc:
            print(
                "Relationship model inference failed:",
                exc,
            )

            model_score = None

    # ==============================================
    # EXPLAINABLE EVIDENCE SCORE
    # ==============================================

    calls = float(record.get("phone_call_count") or record.get("calls") or 0)

    duration = float(
        record.get("total_call_duration_sec") or record.get("duration") or 0
    )

    transactions = float(
        record.get("transaction_count") or record.get("transactions") or 0
    )

    amount = float(record.get("total_transaction_amount") or record.get("amount") or 0)

    meetings = float(record.get("meeting_count") or record.get("meetings") or 0)

    source_diversity = float(record.get("source_diversity") or 0)

    co = float(record.get("co_occurrences") or 0)

    evidence = 0.0

    # Communication evidence
    evidence += min(calls / 20.0, 1.0) * 20.0

    # Duration evidence
    evidence += min(duration / 5000.0, 1.0) * 12.0

    # Financial evidence
    evidence += min(transactions / 5.0, 1.0) * 18.0

    # Transaction amount evidence
    evidence += min(amount / 250000.0, 1.0) * 15.0

    # Meetings
    evidence += min(meetings / 4.0, 1.0) * 15.0

    # Multiple source evidence
    evidence += min(source_diversity / 4.0, 1.0) * 10.0

    # Co-occurrences
    evidence += min(co, 4.0) * 2.5

    # Shared identifiers
    if record.get("shared_phone"):
        evidence += 8.0

    if record.get("shared_vehicle"):
        evidence += 5.0

    if record.get("shared_org"):
        evidence += 4.0

    if record.get("shared_location"):
        evidence += 4.0

    evidence_score = max(0.0, min(100.0, evidence)) / 100.0

    # ==============================================
    # FINAL SCORE
    # ==============================================

    if model_score is None:

        final_score = evidence_score

    else:

        final_score = 0.75 * evidence_score + 0.25 * model_score

    return float(max(0.0, min(1.0, final_score)))


def get_risk_level(confidence: float) -> str:
    """
    Bucket a 0..1 relationship confidence score into an investigator-facing
    risk tier, using the SIH-recommended cutoffs:

        0.00 - 0.34  -> LOW
        0.35 - 0.54  -> MEDIUM
        0.55 - 0.74  -> SUSPICIOUS
        0.75 - 1.00  -> HIGH

    `confidence` is expected on the 0..1 scale that predict_relationship()
    already returns (not 0..100).
    """
    value = float(confidence or 0.0)

    if value >= 0.75:
        return "HIGH"
    elif value >= 0.55:
        return "SUSPICIOUS"
    elif value >= 0.35:
        return "MEDIUM"

    return "LOW"


def anomaly_result(record: Dict[str, Any]) -> Dict[str, Any]:

    reasons: List[str] = []

    calls = float(record.get("phone_call_count") or record.get("calls") or 0)

    transactions = float(
        record.get("transaction_count") or record.get("transactions") or 0
    )

    amount = float(record.get("total_transaction_amount") or record.get("amount") or 0)

    meetings = float(record.get("meeting_count") or record.get("meetings") or 0)

    source_diversity = float(record.get("source_diversity") or 0)

    if calls >= 8:
        reasons.append("High communication frequency")

    if transactions >= 3:
        reasons.append("Repeated financial activity")

    if amount >= 100000:
        reasons.append("High aggregate transaction value")

    if meetings >= 2:
        reasons.append("Repeated meetings")

    if source_diversity >= 3:
        reasons.append("Evidence spans multiple intelligence sources")

    if any(
        record.get(key)
        for key in [
            "shared_phone",
            "shared_vehicle",
            "shared_org",
            "shared_location",
        ]
    ):
        reasons.append("Shared identifying or contextual attribute")

    return {
        # Previously required >=2 rule-based reasons before flagging a
        # relationship as suspicious. That silently hid single strong
        # signals (e.g. a relationship the ML model already scores as
        # HIGH/SUSPICIOUS risk on confidence alone) from this panel. One
        # concrete reason is enough for an investigator-facing flag —
        # the reasons list itself still shows exactly what triggered it.
        "is_anomaly": len(reasons) >= 1,
        "anomaly_score": min(
            len(reasons) / 5.0,
            1.0,
        ),
        "reasons": reasons,
    }


def relationship_reason(record: Dict[str, Any]) -> str:
    reasons: List[str] = []
    if record.get("shared_phone"):
        reasons.append("shared phone evidence")
    if record.get("shared_vehicle"):
        reasons.append("shared vehicle evidence")
    if record.get("shared_org"):
        reasons.append("shared organization evidence")
    if record.get("shared_location"):
        reasons.append("shared location evidence")
    if record.get("co_occurrences"):
        reasons.append(
            f"co-occurrence in {int(record['co_occurrences'])} source record(s)"
        )
    if record.get("calls"):
        duration = int(record.get("duration") or 0)
        if duration:
            reasons.append(
                f"{int(record['calls'])} call signal(s) totaling {duration} seconds"
            )
        else:
            reasons.append(f"{int(record['calls'])} call signal(s)")
    if record.get("transactions"):
        amount = float(record.get("amount") or 0)
        reasons.append(
            f"{int(record['transactions'])} transaction signal(s) totaling ₹{amount:,.2f}"
            if amount
            else f"{int(record['transactions'])} transaction signal(s)"
        )
    if record.get("meetings"):
        reasons.append(f"{int(record['meetings'])} meeting signal(s)")
    if record.get("source_diversity"):
        reasons.append(
            f"evidence across {int(record['source_diversity'])} source type(s)"
        )

    return (
        "Candidate link generated from "
        + ", ".join(reasons or ["shared investigative context"])
        + "."
    )


# -----------------------------------------------------------------------------
# Investigator assistance insights
# -----------------------------------------------------------------------------


def build_investigator_insights(
    relationships: List[Dict[str, Any]],
    influential_persons: List[Dict[str, Any]],
    suspicious_patterns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create concise, evidence-based analytical insights for investigators."""

    insights: Dict[str, Any] = {
        "most_influential_person": None,
        "strongest_relationship": None,
        "highest_risk_relationship": None,
        "suspicious_signals_count": len(suspicious_patterns or []),
        "top_financial_relationship": None,
        "top_communication_relationship": None,
        "recommendations": [],
    }

    relationships = relationships or []
    influential_persons = influential_persons or []
    suspicious_patterns = suspicious_patterns or []

    if influential_persons:
        insights["most_influential_person"] = max(
            influential_persons,
            key=lambda item: float(
                item.get("influence")
                or item.get("score")
                or item.get("centrality")
                or item.get("connection_count")
                or 0
            ),
        )

    if relationships:
        insights["strongest_relationship"] = max(
            relationships,
            key=lambda item: float(item.get("model_confidence") or 0),
        )

        risk_order = {
            "CRITICAL": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
        }

        insights["highest_risk_relationship"] = max(
            relationships,
            key=lambda item: (
                risk_order.get(str(item.get("risk_level") or "").upper(), 0),
                float(item.get("model_confidence") or 0),
                float(item.get("anomaly_score") or 0),
            ),
        )

        top_financial = max(
            relationships,
            key=lambda item: float(
                item.get("amount") or item.get("total_transaction_amount") or 0
            ),
        )
        if (
            float(
                top_financial.get("amount")
                or top_financial.get("total_transaction_amount")
                or 0
            )
            > 0
        ):
            insights["top_financial_relationship"] = top_financial

        top_communication = max(
            relationships,
            key=lambda item: float(
                item.get("calls") or item.get("phone_call_count") or 0
            ),
        )
        if (
            float(
                top_communication.get("calls")
                or top_communication.get("phone_call_count")
                or 0
            )
            > 0
        ):
            insights["top_communication_relationship"] = top_communication

    recommendations: List[Dict[str, Any]] = []

    strongest = insights["strongest_relationship"]
    if strongest:
        recommendations.append(
            {
                "priority": "HIGH",
                "type": "RELATIONSHIP",
                "message": (
                    "Review the strongest detected relationship between "
                    f"{strongest.get('person_a_name') or strongest.get('person_a_key') or 'Unknown'} "
                    "and "
                    f"{strongest.get('person_b_name') or strongest.get('person_b_key') or 'Unknown'}."
                ),
            }
        )

    financial = insights["top_financial_relationship"]
    if financial:
        amount = float(
            financial.get("amount") or financial.get("total_transaction_amount") or 0
        )
        recommendations.append(
            {
                "priority": "HIGH" if amount >= 100000 else "MEDIUM",
                "type": "FINANCIAL",
                "message": (
                    "Verify financial activity between "
                    f"{financial.get('person_a_name') or financial.get('person_a_key') or 'Unknown'} "
                    "and "
                    f"{financial.get('person_b_name') or financial.get('person_b_key') or 'Unknown'} "
                    f"with total value ₹{amount:,.0f}."
                ),
            }
        )

    communication = insights["top_communication_relationship"]
    if communication:
        calls = int(
            communication.get("calls") or communication.get("phone_call_count") or 0
        )
        recommendations.append(
            {
                "priority": "HIGH" if calls >= 8 else "MEDIUM",
                "type": "COMMUNICATION",
                "message": (
                    "Review communication records between "
                    f"{communication.get('person_a_name') or communication.get('person_a_key') or 'Unknown'} "
                    "and "
                    f"{communication.get('person_b_name') or communication.get('person_b_key') or 'Unknown'} "
                    f"showing {calls} recorded call(s)."
                ),
            }
        )

    if suspicious_patterns:
        recommendations.append(
            {
                "priority": "HIGH",
                "type": "SUSPICIOUS_ACTIVITY",
                "message": (
                    f"Investigate {len(suspicious_patterns)} detected suspicious "
                    "activity signal(s) and review their supporting evidence."
                ),
            }
        )

    influential = insights["most_influential_person"]
    if influential:
        person_name = (
            influential.get("name")
            or influential.get("person_name")
            or influential.get("person")
            or "the most connected individual"
        )
        recommendations.append(
            {
                "priority": "MEDIUM",
                "type": "NETWORK",
                "message": (
                    f"Prioritize network analysis around {person_name} due to "
                    "their influence within the investigation."
                ),
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "LOW",
                "type": "GENERAL",
                "message": (
                    "No high-priority analytical lead is currently available. "
                    "Add more structured source evidence for stronger insights."
                ),
            }
        )

    insights["recommendations"] = recommendations
    return insights


# -----------------------------------------------------------------------------
# Live-investigation graph construction
# -----------------------------------------------------------------------------


PERSON_FALSE_POSITIVE = {
    "age",
    "status",
    "record",
    "record 1",
    "record 2",
    "record 3",
    "record 4",
    "record 5",
    "record 6",
    "record 7",
    "record 8",
    "cyber crime",
    "cyber crime unit",
    "criminal history",
    "criminal history database",
    "police",
    "police report",
    "police reports",
    "investigation",
    "investigation report",
    "investigators",
    "investigator",
    "financial fraud",
    "fraud",
    "fraud facilitation",
    "to account",
    "from account",
    "to person",
    "from person",
    "caller",
    "receiver",
    "account",
    "amount",
    "date",
    "time",
    "location",
    "vehicle",
    "organization",
    "company",
    "profile",
    "public profile",
    "source",
    "status",
    "completed",
    "under investigation",
    "under review",
    "none",
}

RELATIONSHIP_CUES = (
    "met",
    "meet",
    "meeting",
    "meetings",
    "called",
    "call",
    "contacted",
    "contact",
    "communication",
    "communicated",
    "spoke",
    "talked",
    "messaged",
    "message",
    "interaction",
    "interacted",
    "together",
    "partner",
    "friend",
    "family",
    "brother",
    "sister",
    "father",
    "mother",
    "spouse",
    "relative",
    "colleague",
    "associate",
    "associated",
    "transferred",
    "transfer",
    "transaction",
    "paid",
    "payment",
    "sent",
    "received",
    "observed",
    "seen",
    "travelled",
    "traveling",
    "travelling",
    "shared",
    "linked",
    "connected",
    "arrived",
    "departed",
)


def clean_person_name(
    value: str,
) -> str | None:

    value = re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip(" ,.;:-")

    if not value:
        return None

    normalized = normalize_text(value)

    # ==========================================
    # KNOWN FALSE POSITIVES
    # ==========================================

    if normalized in PERSON_FALSE_POSITIVE:
        return None

    # ==========================================
    # SPLIT NAME
    # ==========================================

    tokens = value.split()

    # Genuine names should generally
    # contain 2 to 4 words.

    if len(tokens) < 2 or len(tokens) > 4:
        return None

    # ==========================================
    # NUMBERS NOT ALLOWED
    # ==========================================

    if any(char.isdigit() for char in value):
        return None

    # ==========================================
    # FALSE POSITIVE TOKENS
    # ==========================================

    blocked_words = {
        # Investigation fields
        "account",
        "amount",
        "status",
        "crime",
        "record",
        "case",
        "reference",
        "date",
        "time",
        "duration",
        # Locations
        "location",
        "sector",
        "road",
        "street",
        "district",
        "state",
        "city",
        "village",
        "area",
        "zone",
        # Organizations
        "police",
        "department",
        "station",
        "bank",
        "company",
        "corporation",
        "agency",
        "bureau",
        "unit",
        # Documents
        "report",
        "complaint",
        "records",
        "database",
        "history",
        "transaction",
        # Documents / source headings
        "fir",
        "cdr",
        "financial",
        "surveillance",
        "social",
        "media",
        "call",
        "detail",
        "details",
        "statement",
        "statements",
        "extract",
        "log",
        "logs",
        # Generic
        "profile",
        "source",
        "evidence",
        "intelligence",
        "investigation",
        "criminal",
    }

    # ==========================================
    # REJECT BLOCKED WORDS
    # ==========================================

    for token in tokens:

        cleaned_token = token.lower().strip(":,.;-")

        if cleaned_token in blocked_words:
            return None

        if cleaned_token in PERSON_FALSE_POSITIVE:
            return None

    # ==========================================
    # EACH TOKEN SHOULD LOOK LIKE A NAME
    # ==========================================

    for token in tokens:

        cleaned_token = token.strip(":,.;-")

        # Must contain letters

        if not re.search(
            r"[A-Za-zÀ-ÿ]",
            cleaned_token,
        ):
            return None

        # Reject ALL CAPS headings

        if len(cleaned_token) > 2 and cleaned_token.isupper():
            return None

    # ==========================================
    # REJECT COMMON NON-PERSON PATTERNS
    # ==========================================

    non_person_patterns = [
        r"\bpolice\b",
        r"\bpolice station\b",
        r"\bsector\s*\d+\b",
        r"\bnoida\b",
        r"\bdelhi\b",
        r"\bghaziabad\b",
        r"\bwarehouse\b",
        r"\bzone\b",
        r"\bbank\b",
        r"\bdepartment\b",
        r"\bdatabase\b",
        r"\breport\b",
        r"\bcomplaint\b",
        r"\bhistory\b",
    ]

    for pattern in non_person_patterns:

        if re.search(
            pattern,
            normalized,
            flags=re.I,
        ):
            return None

    # ==========================================
    # VALID PERSON
    # ==========================================

    return value


def extract_people_from_source_text(
    text: str,
    person_profiles: Dict[str, Dict[str, Any]],
) -> list[str]:
    """
    Resolve only already-known people from this investigation. This is used
    after entity extraction so the graph cannot accidentally promote arbitrary
    entities into people.
    """
    found = []
    for key, profile in person_profiles.items():
        name = profile["name"]
        if re.search(
            rf"\b{re.escape(name)}\b",
            text,
            flags=re.I,
        ):
            found.append(key)
    return sorted(
        set(found),
        key=lambda key: text.lower().find(person_profiles[key]["name"].lower()),
    )


def make_person_profile(
    name: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    profile = {
        "temp_id": "",
        "name": name,
        "age": None,
        "location": None,
        "phone_num": None,
        "vehicle_num": None,
        "org": None,
        "bank_account": None,
        "crime_recorded": None,
        "fir_language": None,
        "source_types": set(),
    }

    def sentence_contexts(text: str) -> list[str]:
        return [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|\n+", text)
            if item.strip()
            and re.search(
                rf"\b{re.escape(name)}\b",
                item,
                flags=re.I,
            )
        ]

    for source in sources:
        entities = source["entities"]
        text = source["content"]
        source_type = source["source_type"].upper()
        profile["source_types"].add(source_type)

        for context in sentence_contexts(text):
            if not profile["phone_num"]:
                match = re.search(
                    r"(?<!\d)((?:\+91[- ]?)?[6-9]\d{9})(?!\d)",
                    context,
                )
                if match:
                    profile["phone_num"] = clean_phone(match.group(1))

            if not profile["vehicle_num"]:
                match = re.search(
                    r"\b((?:DL|HR|UP|PB)[- ]?\d{1,2}[- ]?[A-Z]{1,3}[- ]?\d{3,4})\b",
                    context,
                    flags=re.I,
                )
                if match:
                    profile["vehicle_num"] = match.group(1)

            if not profile["age"]:
                match = re.search(
                    r"\bage\s*[:=]?\s*(\d{1,3})",
                    context,
                    flags=re.I,
                )
                if match:
                    profile["age"] = int(match.group(1))

            if not profile["org"]:
                match = re.search(
                    r"(?:associated with|works at|works for|organization|company)\s*"
                    r"[:=]?\s*([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,6})",
                    context,
                    flags=re.I,
                )
                if match:
                    candidate = match.group(1).strip(" ,.;:")
                    if len(candidate) > 2:
                        profile["org"] = candidate

            if not profile["location"]:
                match = re.search(
                    r"(?:near|at|in|from|location)\s*[:=]?\s*"
                    r"([A-Z][A-Za-z0-9 ,.'-]{2,60})",
                    context,
                    flags=re.I,
                )
                if match:
                    candidate = match.group(1).strip(" ,.;:")
                    if len(candidate) > 2:
                        profile["location"] = candidate

            if not profile["bank_account"]:
                match = re.search(
                    r"\b(?:account|bank account)\s*[:=]?\s*"
                    r"(X{2,}\d{2,}|(?:XX)?\d{6,18})\b",
                    context,
                    flags=re.I,
                )
                if match:
                    profile["bank_account"] = match.group(1)

        # Safe fallback if the source contains exactly one valid person.
        valid_people = extract_valid_people(entities.get("PERSON", []))
        if len(valid_people) == 1 and valid_people[0].lower() == name.lower():
            if not profile["phone_num"] and entities.get("PHONE"):
                profile["phone_num"] = clean_phone(entities["PHONE"][0])
            if not profile["vehicle_num"] and entities.get("VEHICLE"):
                profile["vehicle_num"] = entities["VEHICLE"][0]
            if not profile["org"] and entities.get("ORG"):
                profile["org"] = entities["ORG"][0]
            if not profile["location"] and entities.get("GPE"):
                profile["location"] = entities["GPE"][0]
            if not profile["bank_account"] and entities.get("BANK"):
                profile["bank_account"] = entities["BANK"][0]

        if source_type in {"CRIMINAL_HISTORY", "FIR"} and not profile["crime_recorded"]:
            match = re.search(
                rf"\b{re.escape(name)}\b[^.\n]{{0,180}}?"
                r"(?:recorded\s+categories|crime|charges?|case\s+references?)"
                r"\s*[:=]?\s*([^\.\n]+)",
                text,
                flags=re.I,
            )
            if match:
                profile["crime_recorded"] = match.group(1).strip()

        if source_type == "FIR" and not profile["fir_language"]:
            profile["fir_language"] = source.get("language", "en")

    return profile


def candidate_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def build_live_candidates(
    sources: List[Dict[str, Any]],
    person_profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build ONLY evidence-backed person-to-person relationships.

    There is no all-to-all combination across a complete document.
    """
    candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_candidate(a: str, b: str) -> Dict[str, Any]:
        key = candidate_key(a, b)
        return candidates.setdefault(
            key,
            {
                "person_a_key": key[0],
                "person_b_key": key[1],
                "phone_call_count": 0,
                "total_call_duration_sec": 0,
                "transaction_count": 0,
                "total_transaction_amount": 0,
                "meeting_count": 0,
                "co_occurrences": 0,
                "source_types": set(),
                "shared_phone": 0,
                "shared_vehicle": 0,
                "shared_org": 0,
                "shared_location": 0,
                # NEW: Store the actual evidence behind this relationship
                "evidence": [],
            },
        )

    def add_pair(
        a_key: str,
        b_key: str,
        source_type: str,
        calls: int = 0,
        duration: int = 0,
        transactions: int = 0,
        amount: float = 0.0,
        meetings: int = 0,
        evidence_text: str = "",
    ):
        if not a_key or not b_key or a_key == b_key:
            return

        record = get_candidate(a_key, b_key)
        record["co_occurrences"] += 1
        record["source_types"].add(source_type)
        record["phone_call_count"] += calls
        record["total_call_duration_sec"] += duration
        record["transaction_count"] += transactions
        record["total_transaction_amount"] += amount
        record["meeting_count"] += meetings
        # Store evidence so investigators can see why this relationship exists
        record["evidence"].append(
            {
                "source_type": source_type,
                "description": evidence_text
                or f"Relationship detected from {source_type}",
                "calls": calls,
                "duration": duration,
                "transactions": transactions,
                "amount": amount,
                "meetings": meetings,
            }
        )

    # --------------------------------------------------------------
    # Narrative sources: pair names ONLY when they are explicitly
    # connected in the same sentence/record.
    # --------------------------------------------------------------
    narrative_sources = {
        "FIR",
        "POLICE_REPORT",
        "SURVEILLANCE",
        "SOCIAL_MEDIA",
        "CRIMINAL_HISTORY",
    }

    for source in sources:
        source_type = source["source_type"].upper()
        content = source["content"]

        if source_type in narrative_sources:
            records = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+|\n+", content)
                if item.strip()
            ]

            for record_text in records:
                mentioned = extract_people_from_source_text(
                    record_text,
                    person_profiles,
                )

                if len(mentioned) < 2:
                    continue

                # Only relationship-bearing records generate links.
                lower = record_text.lower()
                if not any(cue in lower for cue in RELATIONSHIP_CUES):
                    continue

                if len(mentioned) == 2:
                    a_key, b_key = mentioned
                    add_pair(
                        a_key,
                        b_key,
                        source_type,
                        meetings=(
                            1
                            if source_type == "SURVEILLANCE"
                            and any(
                                cue in lower
                                for cue in (
                                    "meet",
                                    "meeting",
                                    "observed",
                                    "seen together",
                                )
                            )
                            else 0
                        ),
                    )
                else:
                    # With >2 people in one sentence, do NOT make all pairs.
                    # Link only the two people closest around the relationship
                    # cue. This avoids N choose 2 explosions.
                    cue_positions = [
                        lower.find(cue)
                        for cue in RELATIONSHIP_CUES
                        if lower.find(cue) >= 0
                    ]
                    cue_pos = (
                        min(cue_positions) if cue_positions else len(record_text) // 2
                    )

                    before = [
                        key
                        for key in mentioned
                        if record_text.lower().find(
                            person_profiles[key]["name"].lower()
                        )
                        < cue_pos
                    ]
                    after = [
                        key
                        for key in mentioned
                        if record_text.lower().find(
                            person_profiles[key]["name"].lower()
                        )
                        > cue_pos
                    ]

                    if before and after:
                        add_pair(
                            before[-1],
                            after[0],
                            source_type,
                            meetings=(1 if source_type == "SURVEILLANCE" else 0),
                        )

        # ----------------------------------------------------------
        # CDR: ONLY explicit caller -> receiver phone pair.
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # CDR: Parse explicit caller -> receiver phone pairs.
        # Supports multiple common input formats.
        # ----------------------------------------------------------
        elif source_type == "CDR":

            people_by_phone = {
                clean_phone(profile.get("phone_num")): key
                for key, profile in person_profiles.items()
                if profile.get("phone_num")
            }

            print("\n========== CDR PARSER DEBUG ==========")
            print("Known phones:", people_by_phone)
            print("CDR Content:")
            print(content)
            print("======================================\n")

            # Split using Record, Call, Entry, or Transaction-style blocks.
            blocks = [
                item.strip()
                for item in re.split(
                    r"(?=(?:Record|Call|Entry)\s*\d+)",
                    content,
                    flags=re.I,
                )
                if item.strip()
            ]

            # If no numbered blocks were found, treat every non-empty line
            # as a possible CDR record.
            if len(blocks) <= 1:
                blocks = [line.strip() for line in content.splitlines() if line.strip()]

            for block in blocks:

                print("\n----- CDR BLOCK -----")
                print(block)

                # Extract all Indian phone numbers from the block.
                phone_matches = re.findall(
                    r"(?:\+91[-\s]?)?[6-9]\d{9}",
                    block,
                )

                if len(phone_matches) < 2:
                    continue

                caller = clean_phone(phone_matches[0])
                receiver = clean_phone(phone_matches[1])

                a_key = people_by_phone.get(caller)
                b_key = people_by_phone.get(receiver)

                print("Caller phone  :", caller)
                print("Receiver phone:", receiver)
                print("Person A      :", a_key)
                print("Person B      :", b_key)

                # Both phone numbers must belong to known investigation persons.
                if not a_key or not b_key or a_key == b_key:
                    print("Skipping unknown phone pair.")
                    continue

                # Try multiple duration formats.
                duration_match = re.search(
                    r"(?:Duration|Call Duration|Duration Sec|Seconds?)"
                    r"\s*[:=-]?\s*(\d+)",
                    block,
                    flags=re.I,
                )

                duration = int(duration_match.group(1)) if duration_match else 0

                add_pair(
                    a_key,
                    b_key,
                    "CDR",
                    calls=1,
                    duration=duration,
                )

                print("CDR relationship added successfully.")

        # ----------------------------------------------------------
        # Financial: ONLY explicit From Person -> To Person.
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # FINANCIAL: Parse explicit sender -> receiver transactions.
        # Supports multiple common input formats.
        # ----------------------------------------------------------
        elif source_type == "FINANCIAL":

            print("\n========== FINANCIAL PARSER DEBUG ==========")
            print("Financial Content:")
            print(content)
            print("============================================\n")

            # Split numbered transactions
            transactions = [
                item.strip()
                for item in re.split(
                    r"(?=(?:Transaction|Txn|Record)\s*\d+)",
                    content,
                    flags=re.I,
                )
                if item.strip()
            ]

            # If no numbered blocks exist, try paragraphs
            if len(transactions) <= 1:
                transactions = [
                    item.strip()
                    for item in re.split(r"\n\s*\n", content)
                    if item.strip()
                ]

            for chunk in transactions:

                print("\n----- FINANCIAL BLOCK -----")
                print(chunk)

                # Sender
                from_match = re.search(
                    r"(?:From Person|From|Sender|Payer)\s*:\s*([^\n]+)",
                    chunk,
                    flags=re.I,
                )

                # Receiver
                to_match = re.search(
                    r"(?:To Person|To|Receiver|Recipient|Payee)\s*:\s*([^\n]+)",
                    chunk,
                    flags=re.I,
                )

                # Amount
                amount_match = re.search(
                    r"(?:Amount|Value|Transaction Amount)"
                    r"\s*:\s*(?:INR|Rs\.?|₹)?\s*"
                    r"([0-9,]+(?:\.\d+)?)",
                    chunk,
                    flags=re.I,
                )

                if not from_match or not to_match:
                    print("Skipping: sender or receiver not found.")
                    continue

                # Normalize names
                a_key = normalize_text(from_match.group(1))
                b_key = normalize_text(to_match.group(1))

                print("From:", a_key)
                print("To  :", b_key)

                if a_key not in person_profiles:
                    print(f"Unknown sender: {a_key}")
                    continue

                if b_key not in person_profiles:
                    print(f"Unknown receiver: {b_key}")
                    continue

                if a_key == b_key:
                    print("Skipping same person transaction.")
                    continue

                amount = (
                    float(amount_match.group(1).replace(",", ""))
                    if amount_match
                    else 0.0
                )

                print("Amount:", amount)

                add_pair(
                    a_key,
                    b_key,
                    "FINANCIAL",
                    transactions=1,
                    amount=amount,
                )

                print("Financial relationship added successfully.")

    results: List[Dict[str, Any]] = []

    for record in candidates.values():
        record["source_diversity"] = len(record.pop("source_types"))
        record["calls"] = record["phone_call_count"]
        record["duration"] = record["total_call_duration_sec"]
        record["transactions"] = record["transaction_count"]
        record["amount"] = record["total_transaction_amount"]
        record["meetings"] = record["meeting_count"]

        record["model_confidence"] = predict_relationship(record)
        record["risk_level"] = get_risk_level(record["model_confidence"])

        # Human-readable score basis for the investigator.
        score_factors = []

        if int(record.get("calls") or 0) > 0:
            score_factors.append(f"{int(record['calls'])} phone call(s)")

        if int(record.get("transactions") or 0) > 0:
            score_factors.append(
                f"{int(record['transactions'])} financial transaction(s)"
            )

        if float(record.get("amount") or 0) > 0:
            score_factors.append(
                f"₹{float(record['amount']):,.0f} total transaction value"
            )

        if int(record.get("meetings") or 0) > 0:
            score_factors.append(
                f"{int(record['meetings'])} meeting/observation record(s)"
            )

        if int(record.get("source_diversity") or 0) > 1:
            score_factors.append(
                f"{int(record['source_diversity'])} independent source type(s)"
            )

        if record.get("shared_phone"):
            score_factors.append("shared phone identifier")

        if record.get("shared_vehicle"):
            score_factors.append("shared vehicle identifier")

        if record.get("shared_org"):
            score_factors.append("shared organization")

        if record.get("shared_location"):
            score_factors.append("shared location")

        record["score_basis"] = score_factors
        record["reason"] = relationship_reason(record)

        calls = int(record.get("calls") or 0)
        txns = int(record.get("transactions") or 0)
        meetings = int(record.get("meetings") or 0)

        if calls >= 3 and txns >= 1:
            record["relationship_type"] = "Communication & Financial Association"
        elif meetings >= 1 and txns >= 1:
            record["relationship_type"] = "Meeting & Financial Association"
        elif meetings >= 2:
            record["relationship_type"] = "Repeated Meeting Association"
        elif txns >= 1:
            record["relationship_type"] = "Financial Association"
        elif calls >= 1:
            record["relationship_type"] = "Communication Association"
        elif record.get("source_diversity", 0) >= 2:
            record["relationship_type"] = "Multi-source Association"
        else:
            record["relationship_type"] = "Evidence-linked Association"

        anomaly = anomaly_result(record)
        record["suspicious"] = anomaly["is_anomaly"]
        record["anomaly_score"] = anomaly["anomaly_score"]
        record["suspicious_reasons"] = anomaly["reasons"]

        results.append(record)

    results.sort(
        key=lambda item: item["model_confidence"],
        reverse=True,
    )
    return results


def build_live_graph(
    investigation_id: str,
    person_profiles: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    connected_keys = set()

    for record in candidates:
        connected_keys.add(record["person_a_key"])
        connected_keys.add(record["person_b_key"])

    investigation_prefix = re.sub(
        r"[^A-Za-z0-9]",
        "",
        investigation_id,
    )[:12].upper()

    key_to_id: Dict[str, str] = {}

    for index, key in enumerate(
        sorted(connected_keys),
        start=1,
    ):
        key_to_id[key] = f"LIVE-{investigation_prefix}-{index:04d}"

    nodes = []

    for key in sorted(connected_keys):
        profile = person_profiles[key]

        nodes.append(
            {
                "id": key_to_id[key],
                "name": profile["name"],
                "type": "PERSON",
                "is_center": False,
                "age": profile.get("age"),
                "location": profile.get("location"),
                "phone_num": profile.get("phone_num"),
                "vehicle_num": profile.get("vehicle_num"),
                "org": profile.get("org"),
                "bank_account": profile.get("bank_account"),
                "crime_recorded": profile.get("crime_recorded"),
                "fir_language": profile.get("fir_language"),
                "source_types": sorted(profile.get("source_types") or []),
            }
        )

    links = []

    for record in candidates:
        source = key_to_id.get(record["person_a_key"])
        target = key_to_id.get(record["person_b_key"])

        if not source or not target:
            continue

        links.append(
            {
                "source": source,
                "target": target,
                "relationship_type": record["relationship_type"],
                "relationship_description": (
                    f"{record['reason']} This is an analytical lead "
                    "generated only from evidence supplied for this "
                    "investigation."
                ),
                "confidence": float(record.get("model_confidence") or 0.0),
                "risk_level": record.get("risk_level")
                or get_risk_level(record.get("model_confidence")),
                "reason": record["reason"],
                "score_basis": record.get("score_basis", []),
                "calls": record["phone_call_count"],
                "total_call_duration_sec": record["total_call_duration_sec"],
                "transactions": record["transaction_count"],
                "meetings": record["meeting_count"],
                "total_transaction_amount": record["total_transaction_amount"],
                "suspicious": record["suspicious"],
                "anomaly_score": record["anomaly_score"],
                "suspicious_reasons": record["suspicious_reasons"],
            }
        )

    return {
        "nodes": nodes,
        "links": links,
    }


def live_graph_analytics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    import networkx as nx

    graph = nx.Graph()
    names = {}
    for node in graph_data["nodes"]:
        graph.add_node(node["id"])
        names[node["id"]] = node["name"]
    for link in graph_data["links"]:
        graph.add_edge(
            link["source"],
            link["target"],
            weight=float(link.get("confidence") or 0.0),
        )

    if not graph.nodes:
        return {"influential_persons": [], "community_count": 0}

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    pagerank = (
        nx.pagerank(graph, weight="weight")
        if len(graph) > 1
        else {next(iter(graph.nodes)): 1.0}
    )

    influential = []
    for pid in graph.nodes:
        score = (
            0.35 * degree.get(pid, 0)
            + 0.35 * betweenness.get(pid, 0)
            + 0.30 * pagerank.get(pid, 0)
        )
        influential.append(
            {
                "person_id": pid,
                "name": names.get(pid, pid),
                "influence_score": round(float(score), 4),
                "degree_centrality": round(float(degree.get(pid, 0)), 4),
                "betweenness_centrality": round(float(betweenness.get(pid, 0)), 4),
                "pagerank": round(float(pagerank.get(pid, 0)), 4),
            }
        )

    influential.sort(key=lambda item: item["influence_score"], reverse=True)
    return {
        "influential_persons": influential[:10],
        "community_count": nx.number_connected_components(graph),
    }


# -----------------------------------------------------------------------------
# Persistence helpers — current investigation only
# -----------------------------------------------------------------------------


def persist_people(
    investigation_id: str,
    graph_data: Dict[str, Any],
    source_documents: List[Dict[str, Any]],
) -> None:
    """
    Persist only people generated from the current investigation's submitted
    evidence. source_type is stored inside each item's document object.
    """
    if supabase is None:
        return

    document_ids: Dict[str, str] = {}

    for item in source_documents:
        document = item.get("document") or {}
        source_type = document.get("source_type")
        document_id = document.get("id")

        if source_type and document_id:
            document_ids[source_type] = document_id

    for node in graph_data.get("nodes", []):
        payload = {
            "person_id": node["id"],
            "investigation_id": investigation_id,
            "name": node["name"],
            "age": node.get("age"),
            "location": node.get("location"),
            "phone_num": node.get("phone_num"),
            "vehicle_num": node.get("vehicle_num"),
            "org": node.get("org"),
            "bank_account": node.get("bank_account"),
            "crime_recorded": (node.get("crime_recorded") or "Source-linked subject"),
            "fir_language": node.get("fir_language"),
            "source_document_id": document_ids.get("FIR"),
        }

        payload = {key: value for key, value in payload.items() if value is not None}

        try:
            existing = (
                supabase.table("persons")
                .select("id")
                .eq("investigation_id", investigation_id)
                .eq("person_id", node["id"])
                .limit(1)
                .execute()
            )

            if existing.data:
                (
                    supabase.table("persons")
                    .update(payload)
                    .eq("id", existing.data[0]["id"])
                    .execute()
                )
            else:
                supabase.table("persons").insert(payload).execute()

        except Exception as exc:
            print(
                f"Person persistence failed for "
                f"{node.get('name', 'Unknown')}: {exc}"
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unable to save extracted person "
                    f"{node.get('name', 'Unknown')}: {exc}"
                ),
            )


def persist_relationships(
    investigation_id: str,
    graph_data: Dict[str, Any],
) -> None:
    """
    Persist generated relationships for this investigation only.
    Retries without optional anomaly fields for an older schema.
    """
    if supabase is None:
        return

    try:
        (
            supabase.table("person_relationships")
            .delete()
            .eq("investigation_id", investigation_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to reset investigation relationships: {exc}",
        )

    node_names = {node["id"]: node["name"] for node in graph_data.get("nodes", [])}

    investigation_token = re.sub(
        r"[^A-Za-z0-9]",
        "",
        investigation_id,
    )[:12].upper()

    for index, link in enumerate(
        graph_data.get("links", []),
        start=1,
    ):
        payload = {
            "relationship_id": f"LIVE-{investigation_token}-{index:04d}",
            "investigation_id": investigation_id,
            "person_a_id": link["source"],
            "person_a_name": node_names.get(
                link["source"],
                link["source"],
            ),
            "person_b_id": link["target"],
            "person_b_name": node_names.get(
                link["target"],
                link["target"],
            ),
            "phone_call_count": link.get("calls", 0),
            "total_call_duration_sec": link.get(
                "total_call_duration_sec",
                0,
            ),
            "transaction_count": link.get(
                "transactions",
                0,
            ),
            "total_transaction_amount": link.get(
                "total_transaction_amount",
                0,
            ),
            "meeting_count": link.get(
                "meetings",
                0,
            ),
            "relationship_label": 1,
            "ground_truth_confidence": None,
            "model_confidence": float(link.get("confidence") or 0.0),
            "relationship_type": (
                link.get("relationship_type") or "Evidence-linked Association"
            ),
            "relationship_description": link.get("relationship_description"),
            "reason": link.get("reason"),
            "suspicious": bool(link.get("suspicious")),
            "anomaly_score": link.get("anomaly_score"),
        }

        try:
            (supabase.table("person_relationships").insert(payload).execute())
        except Exception as first_error:
            error_text = str(first_error).lower()

            missing_optional_column = (
                "suspicious" in error_text
                or "anomaly_score" in error_text
                or ("column" in error_text and "does not exist" in error_text)
            )

            if not missing_optional_column:
                raise HTTPException(
                    status_code=500,
                    detail=("Unable to save generated relationship: " f"{first_error}"),
                )

            legacy_payload = dict(payload)
            legacy_payload.pop("suspicious", None)
            legacy_payload.pop("anomaly_score", None)

            try:
                (
                    supabase.table("person_relationships")
                    .insert(legacy_payload)
                    .execute()
                )
            except Exception as second_error:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Unable to save generated relationship. "
                        f"Initial error: {first_error}; "
                        f"Retry error: {second_error}"
                    ),
                )


# -----------------------------------------------------------------------------
# Core endpoints
# -----------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase_configured": supabase is not None,
        "relationship_model_loaded": RELATIONSHIP_MODEL is not None,
        # "anomaly_model_loaded": ANOMALY_MODEL is not None,
        "model_loading": "lazy_on_first_analysis",
        "analysis_mode": "live-submitted-evidence",
    }


@app.get("/api/investigations/{investigation_id}/blockchain/verify")
def verify_investigation_blockchain(
    investigation_id: str,
):
    result = verify_blockchain_integrity(investigation_id)

    return result


@app.get("/api/investigations")
def list_investigations(
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    result = (
        supabase.table("investigations")
        .select("*")
        .eq("created_by", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


@app.post("/api/investigations")
def create_investigation(
    body: InvestigationCreate,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    result = (
        supabase.table("investigations")
        .insert(
            {
                "title": body.title.strip(),
                "description": body.description,
                "created_by": user_id,
                "status": "active",
            }
        )
        .execute()
    )
    if not result.data:
        raise HTTPException(500, "Unable to create investigation")
    return result.data[0]


@app.post("/api/investigations/{investigation_id}/close")
def close_investigation(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    result = (
        supabase.table("investigations")
        .update({"status": "closed", "closed_at": utc_now()})
        .eq("id", investigation_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Investigation not found")
    return result.data[0]


@app.get("/api/investigations/{investigation_id}/workspace")
def get_investigation_workspace(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    """Return the complete persisted workspace for the authenticated owner.

    Keeping investigation metadata, saved source text, and persisted graph/
    analysis in one authenticated response avoids login-time races between
    separate source and analysis requests.
    """
    user_id = require_user(authorization)
    investigation = require_investigation_owner(investigation_id, user_id)

    source_result = (
        supabase.table("investigation_sources")
        .select(
            "id, investigation_id, source_type, title, content, language, created_at, updated_at"
        )
        .eq("investigation_id", investigation_id)
        .order("source_type")
        .execute()
    )
    source_rows = []
    for row in source_result.data or []:
        source_rows.append(
            {
                **row,
                "content": (
                    "" if row.get("content") is None else str(row.get("content"))
                ),
            }
        )

    persons = (
        supabase.table("persons")
        .select("*")
        .eq("investigation_id", investigation_id)
        .execute()
        .data
        or []
    )

    relationships = (
        supabase.table("person_relationships")
        .select("*")
        .eq("investigation_id", investigation_id)
        .execute()
        .data
        or []
    )

    graph_data = {
        "nodes": [
            {
                "id": p["person_id"],
                "name": p["name"],
                "type": "PERSON",
                "is_center": False,
                "age": p.get("age"),
                "location": p.get("location"),
                "phone_num": p.get("phone_num"),
                "vehicle_num": p.get("vehicle_num"),
                "org": p.get("org"),
                "bank_account": p.get("bank_account"),
                "crime_recorded": p.get("crime_recorded"),
                "fir_language": p.get("fir_language"),
            }
            for p in persons
        ],
        "links": [
            {
                "source": r["person_a_id"],
                "target": r["person_b_id"],
                "relationship_type": r.get("relationship_type")
                or "Evidence-linked Association",
                "relationship_description": r.get("relationship_description"),
                "confidence": r.get("model_confidence"),
                "risk_level": r.get("risk_level")
                or get_risk_level(r.get("model_confidence")),
                "suspicious": r.get("suspicious", False),
                "anomaly_score": r.get("anomaly_score"),
            }
            for r in relationships
        ],
    }

    latest_runs = (
        supabase.table("analysis_runs")
        .select(
            "id, created_at, sources_processed, entities_extracted, candidate_links, suspicious_links, summary"
        )
        .eq("investigation_id", investigation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    latest = latest_runs.data[0] if latest_runs.data else None
    saved_summary = latest.get("summary", {}) if latest else {}

    analytics = live_graph_analytics(graph_data)
    return {
        "investigation": investigation,
        "investigation_id": investigation_id,
        "sources": source_rows,
        "analysis_run": latest,
        "analysis_mode": "persisted_current_investigation",
        "graph": graph_data,
        **analytics,
        "entity_counts": saved_summary.get("entity_counts", {}),
        "candidate_relationships": saved_summary.get("top_relationships", []),
        "suspicious_patterns": saved_summary.get("suspicious_patterns", []),
        "influential_persons": saved_summary.get(
            "influential_persons", analytics.get("influential_persons", [])
        ),
        "investigator_insights": saved_summary.get(
            "investigator_insights",
            build_investigator_insights(
                saved_summary.get("top_relationships", []),
                analytics.get("influential_persons", []),
                saved_summary.get("suspicious_patterns", []),
            ),
        ),
        "community_count": saved_summary.get(
            "community_count", analytics.get("community_count", 0)
        ),
        "summary_text": saved_summary.get("summary_text", ""),
        "source_snapshot_hash": saved_summary.get("source_snapshot_hash"),
        "source_count": len(source_result.data or []),
    }


@app.get("/api/investigations/{investigation_id}/sources")
def get_investigation_sources(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)

    require_investigation_owner(investigation_id, user_id)

    result = (
        supabase.table("investigation_sources")
        .select("*")
        .eq("investigation_id", investigation_id)
        .order("source_type")
        .execute()
    )

    # Return a stable object shape for the frontend.
    return {
        "investigation_id": investigation_id,
        "sources": result.data or [],
    }


@app.post("/api/investigations/{investigation_id}/sources/upload")
async def upload_investigation_source(
    investigation_id: str,
    source_type: str = Form(...),
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: str = Form("en"),
    authorization: Optional[str] = Header(None),
):
    """Upload a TXT or PDF intelligence source and save extracted text.

    The original file is not required for analysis because NyayaNet stores the
    extracted text in the same investigation_sources table used by the editor.
    """
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)

    normalized_type = source_type.strip().upper()
    if not normalized_type:
        raise HTTPException(400, "source_type is required")

    allowed_source_types = {
        "FIR",
        "POLICE_REPORT",
        "CDR",
        "FINANCIAL",
        "SURVEILLANCE",
        "SOCIAL_MEDIA",
        "CRIMINAL_HISTORY",
    }
    if normalized_type not in allowed_source_types:
        raise HTTPException(400, "Invalid source_type")

    if not file.filename:
        raise HTTPException(400, "Please select a TXT or PDF file")

    raw = await file.read()
    max_size = 15 * 1024 * 1024
    if not raw:
        raise HTTPException(400, "Uploaded file is empty")
    if len(raw) > max_size:
        raise HTTPException(413, "File is too large. Maximum allowed size is 15 MB.")

    extracted_text = extract_uploaded_source_text(
        file.filename,
        file.content_type or "",
        raw,
    )

    source_title = (
        title.strip() if title and title.strip() else Path(file.filename).stem
    )
    payload = {
        "investigation_id": investigation_id,
        "source_type": normalized_type,
        "title": source_title,
        "content": extracted_text,
        "language": language or "en",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }

    try:
        result = (
            supabase.table("investigation_sources")
            .upsert(
                payload,
                on_conflict="investigation_id,source_type",
            )
            .execute()
        )
    except Exception as exc:
        print("Source upload persistence failed:", exc)
        raise HTTPException(500, "Unable to save uploaded source")

    saved = result.data[0] if result.data else payload
    return {
        "message": "Source uploaded and text extracted successfully",
        "investigation_id": investigation_id,
        "source_type": normalized_type,
        "title": source_title,
        "filename": file.filename,
        "file_type": Path(file.filename).suffix.lower(),
        "characters_extracted": len(extracted_text),
        "source": saved,
    }


@app.put("/api/investigations/{investigation_id}/sources")
def save_investigation_sources(
    investigation_id: str,
    body: Dict[str, Any],
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)

    require_investigation_owner(
        investigation_id,
        user_id,
    )

    sources = normalize_source_payload(body.get("sources"))

    if not sources:

        return {
            "investigation_id": investigation_id,
            "sources": [],
            "saved_count": 0,
        }

    # ==========================================================
    # LOAD EXISTING SOURCES
    # ==========================================================

    try:

        existing_response = (
            supabase.table("investigation_sources")
            .select("*")
            .eq(
                "investigation_id",
                investigation_id,
            )
            .execute()
        )

        existing_sources = existing_response.data or []

    except Exception as exc:

        print(
            "Unable to load existing sources:",
            exc,
        )

        existing_sources = []

    # ==========================================================
    # CREATE LOOKUP
    # ==========================================================

    existing_by_type = {}

    for item in existing_sources:

        source_type = (
            str(
                item.get(
                    "source_type",
                    "",
                )
            )
            .strip()
            .upper()
        )

        if source_type:

            existing_by_type[source_type] = item

    # ==========================================================
    # PREPARE UPSERT PAYLOAD
    # ==========================================================

    payload = []

    blockchain_created = []

    for source in sources:

        source_type = source.source_type.strip().upper()

        if not source_type:
            continue

        content = source.content or ""

        existing = existing_by_type.get(source_type)

        # ======================================================
        # EMPTY SOURCE
        # ======================================================

        if not content.strip():

            # Do not create blockchain evidence
            # for empty source rows.

            if existing:

                payload.append(
                    {
                        "investigation_id": investigation_id,
                        "source_type": source_type,
                        "title": existing.get("title") or source_type.title(),
                        "content": "",
                        "language": source.language or existing.get("language") or "en",
                        "evidence_id": existing.get("evidence_id"),
                        "evidence_hash": existing.get("evidence_hash"),
                        "created_at": existing.get("created_at") or utc_now(),
                        "updated_at": utc_now(),
                    }
                )

            continue

        # ======================================================
        # CALCULATE CURRENT HASH
        # ======================================================

        current_hash = generate_evidence_hash(content)

        # ======================================================
        # NEW SOURCE
        # ======================================================

        if not existing:

            block = create_evidence_block(
                investigation_id=investigation_id,
                source_type=source_type,
                title=source.title or source_type.title(),
                content=content,
            )

            blockchain_created.append(source_type)

            payload.append(
                {
                    "investigation_id": investigation_id,
                    "source_type": source_type,
                    "title": source.title or source_type.title(),
                    "content": content,
                    "language": source.language or "en",
                    "evidence_id": block["evidence_id"],
                    "evidence_hash": block["evidence_hash"],
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )

            continue

        # ======================================================
        # EXISTING SOURCE
        # ======================================================

        existing_hash = existing.get("evidence_hash")

        # ======================================================
        # CONTENT DID NOT CHANGE
        # ======================================================

        if existing_hash and current_hash == existing_hash:

            payload.append(
                {
                    "investigation_id": investigation_id,
                    "source_type": source_type,
                    "title": existing.get("title")
                    or source.title
                    or source_type.title(),
                    "content": content,
                    "language": source.language or existing.get("language") or "en",
                    # IMPORTANT:
                    # KEEP SAME BLOCKCHAIN EVIDENCE
                    "evidence_id": existing.get("evidence_id"),
                    "evidence_hash": existing_hash,
                    "created_at": existing.get("created_at") or utc_now(),
                    "updated_at": utc_now(),
                }
            )

            continue

        # ======================================================
        # CONTENT CHANGED
        # ======================================================

        block = create_evidence_block(
            investigation_id=investigation_id,
            source_type=source_type,
            title=existing.get("title") or source.title or source_type.title(),
            content=content,
        )

        blockchain_created.append(source_type)

        payload.append(
            {
                "investigation_id": investigation_id,
                "source_type": source_type,
                "title": existing.get("title") or source.title or source_type.title(),
                "content": content,
                "language": source.language or existing.get("language") or "en",
                "evidence_id": block["evidence_id"],
                "evidence_hash": block["evidence_hash"],
                "created_at": existing.get("created_at") or utc_now(),
                "updated_at": utc_now(),
            }
        )

    # ==========================================================
    # NOTHING TO SAVE
    # ==========================================================

    if not payload:

        return {
            "investigation_id": investigation_id,
            "sources": [],
            "saved_count": 0,
            "blockchain_created": [],
        }

    # ==========================================================
    # SAVE SOURCES
    # ==========================================================

    try:

        result = (
            supabase.table("investigation_sources")
            .upsert(
                payload,
                on_conflict="investigation_id,source_type",
            )
            .execute()
        )

    except Exception as exc:

        print(
            "Source persistence failed:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=("Unable to save " "investigation sources"),
        )

    # ==========================================================
    # RETURN
    # ==========================================================

    return {
        "investigation_id": investigation_id,
        "sources": result.data or [],
        "saved_count": len(result.data or []),
        "blockchain_created": blockchain_created,
    }


_ALLCAPS_RUN_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3}\b")


def normalize_case_for_ner(text: str) -> str:
    """spaCy's NER leans heavily on capitalization to spot proper nouns.
    Case-file text often renders names in ALL CAPS (e.g. "ARJUN MALHOTRA")
    inside otherwise normal-case sentences, and spaCy typically fails to
    tag those as PERSON. Title-case any multi-word ALL-CAPS run so NER
    sees an ordinary proper noun, while leaving short single-word
    acronyms (FIR, CDR, PDF...) alone.
    """

    def _fix(match):
        word = match.group(0)
        if len(word.split()) == 1 and len(word) <= 4:
            return word
        return word.title()

    return _ALLCAPS_RUN_RE.sub(_fix, text)


def extract_entities(text: str):
    from .nlp import extract_entities as _extract_entities

    return _extract_entities(normalize_case_for_ner(text))


def normalize_source_payload(raw_sources: Any) -> List[SourceInput]:
    """
    Normalize source payloads so a harmless empty/null field from the React
    form does not turn the entire analysis request into a FastAPI 422.
    Invalid/empty source rows are ignored; at least one non-empty source is
    still required by analyze_sources().
    """
    if raw_sources is None:
        return []

    if isinstance(raw_sources, dict):
        raw_sources = raw_sources.get("sources", [])

    if not isinstance(raw_sources, list):
        return []

    normalized: List[SourceInput] = []

    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue

        source_type = (
            str(raw.get("source_type") or raw.get("type") or "").strip().upper()
        )

        content = raw.get("content")
        if content is None:
            content = ""

        title = raw.get("title")
        language = raw.get("language") or "en"

        normalized.append(
            SourceInput(
                source_type=source_type,
                title=str(title) if title is not None else None,
                content=str(content),
                language=str(language),
            )
        )

    return normalized


def source_snapshot_hash(
    sources: List[Any],
) -> str:
    normalized = []

    for source in sources:
        normalized.append(
            {
                "source_type": str(source.source_type).strip().upper(),
                "title": source.title or "",
                "content": source.content or "",
                "language": source.language or "en",
            }
        )

    normalized.sort(key=lambda item: item["source_type"])
    return sha256_json(normalized)


@app.post("/api/investigations/{investigation_id}/analyze-sources")
def analyze_sources(
    investigation_id: str,
    body: Dict[str, Any],
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)

    sources = normalize_source_payload(body.get("sources"))

    if not sources:
        raise HTTPException(400, "Provide at least one intelligence source")

    if not any(source.content.strip() for source in sources):
        raise HTTPException(400, "Provide at least one non-empty intelligence source")

    # Persist raw inputs before running the analysis pipeline so the
    # investigator's work survives errors, sign-out, and re-login.
    save_investigation_sources(
        investigation_id,
        {
            "sources": [source.model_dump() for source in sources],
        },
        authorization,
    )

    current_source_hash = source_snapshot_hash(sources)

    context_documents: List[Dict[str, Any]] = []
    persisted_documents: List[Dict[str, Any]] = []
    entity_counts: Counter[str] = Counter()
    raw_person_names = {}

    # ------------------------------------------------------------------
    # IMPORTANT: analysis is performed exclusively from this request's
    # submitted source corpus.
    # ------------------------------------------------------------------

    invalid_person_terms = {
        "unknown",
        "person",
        "persons",
        "observed",
        "suspect",
        "accused",
        "witness",
        "complainant",
        "victim",
        "officer",
        "police",
        "report",
        "database",
        "history",
        "records",
        "transaction",
        "transactions",
        "surveillance",
        "intelligence",
        "financial",
        "social media",
        "warehouse",
        "zone",
        "location",
        "area",
        "city",
        "district",
        "station",
    }

    for index, source in enumerate(sources):

        content = source.content.strip()

        if not content:
            continue

        # Extract entities from the current source
        entities = extract_entities(content)

        # Extract locations and organizations so they can be
        # removed from PERSON candidates
        gpe_values = {normalize_text(value) for value in entities.get("GPE", [])}

        location_values = {normalize_text(value) for value in entities.get("LOC", [])}

        org_values = {normalize_text(value) for value in entities.get("ORG", [])}

        # ----------------------------------------------------------
        # FILTER VALID PERSONS
        # ----------------------------------------------------------

        for person_name in entities.get("PERSON", []):

            # Route through the shared clean_person_name() validator —
            # it already rejects digits, field-label junk ("phone",
            # "vehicle", "extract", "reference", ...), ALL-CAPS headings,
            # and known non-person patterns ("noida", "database", "report",
            # etc). The lighter-weight checks below (length, the local
            # invalid_person_terms set, and the location/org overlap
            # rejection) still run on top of it.
            cleaned_name = clean_person_name(person_name)

            if cleaned_name is None:
                continue

            key = normalize_text(cleaned_name)

            if not key:
                continue

            if len(key) < 3:
                continue

            if key in invalid_person_terms:
                continue

            # Reject locations and organizations incorrectly
            # detected as persons
            if any(
                entity
                and (
                    key == entity
                    or re.search(
                        rf"\b{re.escape(entity)}\b",
                        key,
                    )
                )
                for entity in (gpe_values | location_values | org_values)
            ):
                continue

            raw_person_names[key] = cleaned_name

        # ----------------------------------------------------------
        # COUNT ALL ENTITY TYPES
        # ----------------------------------------------------------

        for label, values in entities.items():
            entity_counts[label] += len(values)

        # ----------------------------------------------------------
        # CREATE DOCUMENT RECORD
        # ----------------------------------------------------------

        title = source.title or f"{source.source_type.title()} {index + 1}"

        content_hash = sha256_json(
            {
                "content": content,
                "source_type": source.source_type,
                "title": title,
            }
        )

        row = {
            "investigation_id": investigation_id,
            "source_type": source.source_type.upper(),
            "title": title,
            "content": content,
            "language": source.language,
            "content_hash": content_hash,
            "extracted_entities": entities,
        }

        inserted = supabase.table("documents").insert(row).execute()

        document = inserted.data[0] if inserted.data else None

        context_documents.append(
            {
                "source_type": source.source_type.upper(),
                "title": title,
                "content": content,
                "language": source.language,
                "entities": entities,
                "document_id": (document["id"] if document else None),
            }
        )

        persisted_documents.append(
            {
                "document": document,
                "entities": entities,
            }
        )

    # Build all live person profiles from this submitted corpus only.
    person_profiles: Dict[str, Dict[str, Any]] = {}
    for key, display_name in raw_person_names.items():
        person_profiles[key] = make_person_profile(display_name, context_documents)

    candidates = build_live_candidates(context_documents, person_profiles)
    graph_data = build_live_graph(investigation_id, person_profiles, candidates)
    analytics = live_graph_analytics(graph_data)

    suspicious_patterns = [
        {
            "person_a_id": c["person_a_key"],
            "person_b_id": c["person_b_key"],
            "confidence": c["model_confidence"],
            "risk_level": c["risk_level"],
            "reasons": c["suspicious_reasons"],
            "anomaly_score": c["anomaly_score"],
        }
        for c in candidates
        if c["suspicious"]
    ]

    investigator_insights = build_investigator_insights(
        candidates,
        analytics.get("influential_persons", []),
        suspicious_patterns,
    )

    top_relationships = [
        {
            "person_a_id": c["person_a_key"],
            "person_b_id": c["person_b_key"],
            "confidence": c["model_confidence"],
            "risk_level": c["risk_level"],
            "reason": c["reason"],
            "relationship_type": c["relationship_type"],
            "source_diversity": c["source_diversity"],
            "calls": c["phone_call_count"],
            "transactions": c["transaction_count"],
            "meetings": c["meeting_count"],
            "total_transaction_amount": c["total_transaction_amount"],
        }
        for c in candidates[:10]
    ]

    # Persist only the records generated by this current investigation.
    persist_people(investigation_id, graph_data, persisted_documents)
    persist_relationships(investigation_id, graph_data)

    audit_payload = {
        "sources_processed": len(context_documents),
        "entities": dict(entity_counts),
        "candidate_links": len(candidates),
        "suspicious_links": len(suspicious_patterns),
        "analysis_mode": "submitted_evidence_only",
    }

    try:
        supabase.table("analysis_runs").insert(
            {
                "investigation_id": investigation_id,
                "actor_id": user_id,
                "sources_processed": len(context_documents),
                "entities_extracted": int(sum(entity_counts.values())),
                "candidate_links": len(candidates),
                "suspicious_links": len(suspicious_patterns),
                "summary": {
                    "entity_counts": dict(entity_counts),
                    "top_relationships": top_relationships,
                    "influential_persons": analytics.get("influential_persons", []),
                    "suspicious_patterns": suspicious_patterns[:20],
                    "investigator_insights": investigator_insights,
                    "community_count": analytics.get("community_count", 0),
                    "graph": graph_data,
                    "summary_text": (
                        f"Analysis completed from {len(context_documents)} "
                        f"submitted intelligence source(s), identifying "
                        f"{int(sum(entity_counts.values()))} extracted "
                        f"entities and {len(candidates)} evidence-backed "
                        f"candidate relationship(s)."
                    ),
                    "source_snapshot_hash": current_source_hash,
                    "analysis_mode": "submitted_evidence_only",
                },
            }
        ).execute()
    except Exception as exc:
        print("Analysis-run persistence warning:", exc)

    try:
        previous = ""
        prior = (
            supabase.table("audit_log")
            .select("event_hash")
            .eq("investigation_id", investigation_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if prior.data:
            previous = prior.data[0]["event_hash"]
        event_hash = sha256_json({"payload": audit_payload, "previous_hash": previous})
        supabase.table("audit_log").insert(
            {
                "actor_id": user_id,
                "investigation_id": investigation_id,
                "action": "investigation_analysis",
                "object_type": "analysis_run",
                "object_id": investigation_id,
                "payload": audit_payload,
                "previous_hash": previous,
                "event_hash": event_hash,
            }
        ).execute()
    except Exception as exc:
        print("Audit persistence warning:", exc)

    # Return the graph generated from THIS request, so the frontend does not
    # need to query the 600-person training dataset or another investigation.
    return {
        "investigation_id": investigation_id,
        "analysis_mode": "submitted_evidence_only",
        "sources": [
            {
                "source_type": d["source_type"],
                "title": d["title"],
                "entity_count": sum(len(v) for v in d["entities"].values()),
                "document_id": d["document_id"],
            }
            for d in context_documents
        ],
        "documents": persisted_documents,
        "entity_counts": dict(entity_counts),
        "candidate_relationships": top_relationships,
        "suspicious_patterns": suspicious_patterns[:20],
        "influential_persons": analytics.get("influential_persons", []),
        "community_count": analytics.get("community_count", 0),
        "graph": graph_data,
        "message": "Analysis completed from submitted investigation evidence only. Scores are analytical leads, not proof of criminality.",
    }


@app.get("/api/investigations/{investigation_id}/analysis")
def investigation_analysis(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)

    persons = (
        supabase.table("persons")
        .select("*")
        .eq("investigation_id", investigation_id)
        .execute()
        .data
        or []
    )

    relationships = (
        supabase.table("person_relationships")
        .select("*")
        .eq("investigation_id", investigation_id)
        .execute()
        .data
        or []
    )

    graph_data = {
        "nodes": [
            {
                "id": p["person_id"],
                "name": p["name"],
                "type": "PERSON",
                "is_center": False,
                "age": p.get("age"),
                "location": p.get("location"),
                "phone_num": p.get("phone_num"),
                "vehicle_num": p.get("vehicle_num"),
                "org": p.get("org"),
                "bank_account": p.get("bank_account"),
                "crime_recorded": p.get("crime_recorded"),
                "fir_language": p.get("fir_language"),
            }
            for p in persons
        ],
        "links": [
            {
                "source": r["person_a_id"],
                "target": r["person_b_id"],
                "relationship_type": (
                    r.get("relationship_type") or "Evidence-linked Association"
                ),
                "relationship_description": r.get("relationship_description"),
                "confidence": r.get("model_confidence"),
                "risk_level": r.get("risk_level")
                or get_risk_level(r.get("model_confidence")),
                "reason": r.get("reason"),
                "score_basis": r.get("score_basis") or [],
                "calls": r.get("phone_call_count", 0),
                "transactions": r.get("transaction_count", 0),
                "meetings": r.get("meeting_count", 0),
                "total_transaction_amount": r.get(
                    "total_transaction_amount",
                    0,
                ),
                "suspicious": r.get("suspicious", False),
                "anomaly_score": r.get("anomaly_score"),
            }
            for r in relationships
        ],
    }

    analytics = live_graph_analytics(graph_data)

    latest_runs = (
        supabase.table("analysis_runs")
        .select(
            "id, created_at, sources_processed, entities_extracted, "
            "candidate_links, suspicious_links, summary"
        )
        .eq("investigation_id", investigation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    latest = latest_runs.data[0] if latest_runs.data else None
    saved_summary = latest.get("summary", {}) if latest else {}

    # Return the investigator's saved raw source records together with the
    # persisted graph/analysis. This makes a reopened investigation fully
    # restorable for its owner after logout/login, without relying on any
    # browser-local state. Ownership was already verified above.
    saved_sources_result = (
        supabase.table("investigation_sources")
        .select(
            "id, investigation_id, source_type, title, content, language, created_at, updated_at"
        )
        .eq("investigation_id", investigation_id)
        .order("source_type")
        .execute()
    )
    saved_sources = [
        {
            **row,
            "content": "" if row.get("content") is None else str(row.get("content")),
        }
        for row in (saved_sources_result.data or [])
    ]

    return {
        "investigation_id": investigation_id,
        "analysis_mode": "persisted_current_investigation",
        "graph": graph_data,
        "sources": saved_sources,
        **analytics,
        "analysis_run": latest,
        "entity_counts": saved_summary.get("entity_counts", {}),
        "candidate_relationships": saved_summary.get(
            "top_relationships",
            [],
        ),
        "suspicious_patterns": saved_summary.get(
            "suspicious_patterns",
            [],
        ),
        "influential_persons": saved_summary.get(
            "influential_persons",
            analytics.get("influential_persons", []),
        ),
        "community_count": saved_summary.get(
            "community_count",
            analytics.get("community_count", 0),
        ),
        "summary_text": saved_summary.get("summary_text", ""),
        "source_snapshot_hash": saved_summary.get("source_snapshot_hash"),
    }


# -----------------------------------------------------------------------------
# Legacy-compatible endpoints
# -----------------------------------------------------------------------------


@app.post("/api/nlp/extract")
def nlp_extract(
    body: DocumentIn,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(body.investigation_id, user_id)
    entities = extract_entities(body.content)
    content_hash = sha256_json(
        {"content": body.content, "source_type": body.source_type}
    )
    document_row = {
        "investigation_id": body.investigation_id,
        "source_type": body.source_type,
        "title": body.title,
        "content": body.content,
        "language": body.language,
        "content_hash": content_hash,
        "extracted_entities": entities,
    }
    document = supabase.table("documents").insert(document_row).execute()
    return {
        "entities": entities,
        "document": document.data[0] if document.data else None,
    }


@app.post("/api/documents")
def create_document(
    body: DocumentIn,
    authorization: Optional[str] = Header(None),
):
    return nlp_extract(body, authorization)


@app.post("/api/links")
def create_link(
    body: LinkIn,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(body.investigation_id, user_id)
    last = (
        supabase.table("network_links")
        .select("link_hash")
        .eq("investigation_id", body.investigation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    previous = last.data[0]["link_hash"] if last.data else ""
    payload = body.model_dump()
    link_hash_value = hash_link(payload, previous)
    row = {
        **payload,
        "link_hash": link_hash_value,
        "previous_hash": previous,
        "created_at": utc_now(),
    }
    result = supabase.table("network_links").insert(row).execute()
    return result.data[0]


@app.get("/api/investigations/{investigation_id}/persons")
def get_persons(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    result = (
        supabase.table("persons")
        .select("*")
        .eq("investigation_id", investigation_id)
        .order("name")
        .execute()
    )
    return result.data or []


@app.get("/api/investigations/{investigation_id}/relationships")
def get_relationships(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    result = (
        supabase.table("person_relationships")
        .select("*")
        .eq("investigation_id", investigation_id)
        .execute()
    )
    return result.data or []


@app.get("/api/investigations/{investigation_id}/graph")
def graph(
    investigation_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    analysis = investigation_analysis(investigation_id, authorization)
    return analysis["graph"]


@app.get("/api/investigations/{investigation_id}/persons/search")
def search_persons(
    investigation_id: str,
    q: str = "",
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    value = q.strip()
    if not value:
        return []
    results = []
    base_select = (
        "id, person_id, name, age, location, phone_num, vehicle_num, "
        "org, bank_account, crime_recorded, fir_language"
    )
    for field in ["name", "person_id", "phone_num", "vehicle_num", "org"]:
        result = (
            supabase.table("persons")
            .select(base_select)
            .eq("investigation_id", investigation_id)
            .ilike(field, f"%{value}%")
            .limit(10)
            .execute()
        )
        results.extend(result.data or [])
    unique = {person["id"]: person for person in results}
    return list(unique.values())[:10]


@app.get("/api/investigations/{investigation_id}/network/{person_id}")
def get_person_network(
    investigation_id: str,
    person_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = require_user(authorization)
    require_investigation_owner(investigation_id, user_id)
    data = graph(investigation_id, authorization)
    ids = {person_id}
    for link in data["links"]:
        if link["source"] == person_id:
            ids.add(link["target"])
        elif link["target"] == person_id:
            ids.add(link["source"])
    return {
        "center": next(
            (n for n in data["nodes"] if n["id"] == person_id), {"id": person_id}
        ),
        "nodes": [n for n in data["nodes"] if n["id"] in ids],
        "links": [
            l for l in data["links"] if l["source"] in ids and l["target"] in ids
        ],
    }


@app.post("/api/tips/analyze")
def analyze_tip(
    body: TipIn,
    authorization: Optional[str] = Header(None),
):
    require_user(authorization)
    entities = extract_entities(body.text)
    return {
        "entities": entities,
        "message": "Use these extracted entities as candidate seeds. A full investigation analysis should be run against the submitted source corpus.",
    }