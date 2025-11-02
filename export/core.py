"""Clean exporter implementation to be used by the package export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import markdown2
from weasyprint import HTML

logger = logging.getLogger("agent.export.core")


@dataclass
class Report:
    title: str
    synthesis: str
    traces: List[Dict]


class Exporter:
    def __init__(self, out_dir: str = "output"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(out_dir)

    def export_markdown(self, report: Report, dest: str | Path):
        dest = Path(dest)
        md = [f"# {report.title}", "", report.synthesis, "", "## Reasoning Traces", ""]
        for t in report.traces:
            md.append(f"### Subquery: {t.get('subquery')}")
            for hit in t.get("hits", []):
                md.append(
                    f"- Source: {hit.get('meta', {}).get('source', 'unknown')} — {hit.get('score', 0):.3f}"
                )
                md.append(f"  - {hit.get('text', '')[:200]}...")
        dest.write_text("\n\n".join(md), encoding="utf8")
        logger.info("Wrote markdown report to %s", dest)
        return dest

    def export_pdf(self, report: Report, dest: str | Path):
        dest = Path(dest)
        md_text = self.export_markdown(report, dest.with_suffix(".md"))
        html = markdown2.markdown(md_text.read_text(encoding="utf8"))
        HTML(string=html).write_pdf(str(dest))
        logger.info("Wrote PDF report to %s", dest)
        return dest

    def export_last(self, format: str = "md"):
        files = sorted(
            self.out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not files:
            raise FileNotFoundError("No exported reports found")
        chosen = files[0]
        if format == "pdf" and chosen.suffix != ".pdf":
            pdf = chosen.with_suffix(".pdf")
            html = markdown2.markdown(chosen.read_text(encoding="utf8"))
            HTML(string=html).write_pdf(str(pdf))
            return pdf
        return chosen
