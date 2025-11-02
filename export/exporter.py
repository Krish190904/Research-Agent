"""Thin exporter wrapper that re-exports the clean exporter implementation.

This module keeps backward compatibility for `export.exporter` while delegating
the real implementation to `export.core`.
"""

from .core import Report, Exporter

__all__ = ["Report", "Exporter"]
"""Thin exporter wrapper that re-exports the clean exporter implementation.

This module keeps backward compatibility for `export.exporter` while delegating
the real implementation to `export.core`.
"""

from .core import Report, Exporter

__all__ = ["Report", "Exporter"]
"""Thin exporter wrapper that re-exports the clean exporter implementation.

This module keeps backward compatibility for `export.exporter` while delegating
the real implementation to `export.core`.
"""


__all__ = ["Report", "Exporter"]

"""Export reports to Markdown and PDF using markdown2 and WeasyPrint.
"""
from __future__ import annotations

"""Export reports to Markdown and PDF using markdown2 and WeasyPrint."""
from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("agent.export")


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


"""Export reports to Markdown and PDF using markdown2 and WeasyPrint."""
from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("agent.export")


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


"""Export reports to Markdown and PDF using markdown2 and WeasyPrint."""
from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("agent.export")


@dataclass
class Report:
    title: str
    synthesis: str
    traces: Dict


class Exporter:
    def __init__(self, out_dir: str = "output"):
        import os

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def export_markdown(self, report: Report, dest: str):
        md = [f"# {report.title}", "", report.synthesis, "", "## Reasoning Traces", ""]
        for t in report.traces:
            md.append(f"### Subquery: {t['subquery']}")
            for hit in t.get("hits", []):
                md.append(
                    f"- Source: {hit.get('meta', {}).get('source', 'unknown')} — {hit.get('score'):.3f}"
                )
                md.append(f"  - {hit.get('text')[:200]}...")
        text = "\n\n".join(md)
        with open(dest, "w", encoding="utf8") as f:
            f.write(text)
        logger.info("Wrote markdown report to %s", dest)

    def export_pdf(self, report: Report, dest: str):
        # convert to HTML using markdown2 and render with WeasyPrint
        md_text = f"# {report.title}\n\n{report.synthesis}"
        html = markdown2.markdown(md_text)
        HTML(string=html).write_pdf(dest)
        logger.info("Wrote PDF report to %s", dest)

    def export_last(self, format: str = "md"):
        # placeholder: in a full system we'd track sessions. Here we create a minimal demo.
        rep = Report(
            title="Demo Report", synthesis="This is a demo synthesis.", traces=[]
        )
        if format == "md":
            self.export_markdown(rep, "output/report.md")
        else:
            self.export_pdf(rep, "output/report.pdf")
