"""
Simple feedback storage.

For a production system this would write to a database.  Here we keep a
module-level list so nothing gets lost during a single server run, and
optionally persist to a JSON file on disk.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.models.schemas import FeedbackRequest

_FEEDBACK_FILE = Path("feedback_log.jsonl")
_in_memory: list[dict] = []


def record_feedback(req: FeedbackRequest) -> bool:
    """Append a feedback entry to memory and to the JSONL log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question":  req.question,
        "answer":    req.answer[:500],   # truncate long answers
        "rating":    req.rating,
        "comment":   req.comment,
    }
    _in_memory.append(entry)

    try:
        with _FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        logger.info("Feedback recorded: {}", req.rating)
        return True
    except OSError as exc:
        logger.warning("Could not write feedback file: {}", exc)
        return False


def get_all_feedback() -> list[dict]:
    return list(_in_memory)
