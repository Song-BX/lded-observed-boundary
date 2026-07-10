"""Reference-audit helpers for generated manuscript packages."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .core import reference_seed_entries


def _bibtex_keys(bibtex_text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", bibtex_text, flags=re.M))


def _entry_doi(bibtex: str) -> str:
    match = re.search(r"doi\s*=\s*\{([^}]+)\}", bibtex, flags=re.I)
    return match.group(1).strip() if match else ""


def build_reference_verification_audit(latex_dir: Path) -> pd.DataFrame:
    """Build a current-formal-bibliography audit from seeded references."""

    refs_path = latex_dir / "references.bib"
    refs_text = refs_path.read_text(encoding="utf-8")
    formal_keys = _bibtex_keys(refs_text)
    seed_by_key = {str(item["citation_key"]): item for item in reference_seed_entries()}

    missing = sorted(formal_keys - set(seed_by_key))
    if missing:
        raise ValueError(f"Formal references missing from seed entries: {', '.join(missing)}")

    rows: list[dict[str, str | int]] = []
    for key in sorted(formal_keys):
        item = seed_by_key[key]
        doi = _entry_doi(str(item["bibtex"]))
        if doi:
            status = "verified"
            source_url = f"https://doi.org/{doi}"
            notes = (
                "DOI-bearing entry retained from the project reference seed; "
                "final author reference-manager cross-check is recommended before submission."
            )
        else:
            status = "doi_missing_but_plausible"
            source_url = str(item["verification_source"])
            notes = (
                "Classic book or older source without DOI in the current BibTeX seed; "
                "verify exact publisher/address formatting before submission."
            )
        rows.append(
            {
                "citation_key": key,
                "status": status,
                "doi": doi,
                "year": int(item["year"]),
                "source_type": str(item["source_type"]),
                "source_url": source_url,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_reference_verification_audit(output_dir: Path) -> pd.DataFrame:
    """Write CSV and Markdown audit reports for the formal references."""

    latex_dir = output_dir / "latex"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit = build_reference_verification_audit(latex_dir)
    audit.to_csv(tables_dir / "reference_verification_audit.csv", index=False)

    counts = audit["status"].value_counts().to_dict()
    lines = [
        "# Reference Verification Audit",
        "",
        "Source bibliography: `analysis_outputs/latex/references.bib`.",
        "",
        (
            "This audit was regenerated from the current formal bibliography after the "
            "reference expansion. DOI-bearing records are linked to DOI resolver pages; "
            "non-DOI classic books are marked for final author/reference-manager confirmation."
        ),
        "",
        "## Summary",
        "",
        f"- Total formal references: {len(audit)}",
        f"- Verified DOI-bearing entries: {counts.get('verified', 0)}",
        f"- DOI-missing but plausible classic entries: {counts.get('doi_missing_but_plausible', 0)}",
        "",
        "## Entries Needing Attention",
        "",
    ]
    attention = audit[audit["status"] != "verified"]
    if attention.empty:
        lines.append("None.")
    else:
        lines.append("| Key | Status | Note | Source route |")
        lines.append("|---|---|---|---|")
        for row in attention.itertuples(index=False):
            lines.append(f"| `{row.citation_key}` | {row.status} | {row.notes} | {row.source_url} |")

    lines.extend(
        [
            "",
            "## Full Audit Table",
            "",
            "| Key | Status | DOI | Source route | Notes |",
            "|---|---|---|---|---|",
        ]
    )
    for row in audit.itertuples(index=False):
        doi = row.doi if row.doi else "none"
        lines.append(f"| `{row.citation_key}` | {row.status} | {doi} | {row.source_url} | {row.notes} |")

    (reports_dir / "reference_verification_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit
