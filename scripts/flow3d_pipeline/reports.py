"""Report-writing interface for the Flow3D manuscript package."""

from __future__ import annotations

from .core import (
    write_cover_letter_draft,
    write_enhanced_method_draft,
    write_manuscript_draft_v6,
    write_reviewer_risk_response,
    write_submission_readiness_checklist,
)

__all__ = [
    "write_cover_letter_draft",
    "write_enhanced_method_draft",
    "write_manuscript_draft_v6",
    "write_reviewer_risk_response",
    "write_submission_readiness_checklist",
]
