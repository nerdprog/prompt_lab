from __future__ import annotations

import os
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


class ReportGenerator:
    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir

    @staticmethod
    def _p(text: Any, style: ParagraphStyle) -> Paragraph:
        safe = escape(str(text if text is not None else "—")).replace("\n", "<br/>")
        return Paragraph(safe, style)

    def _page_decor(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7D7D2"))
        canvas.line(40, 34, letter[0] - 40, 34)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(40, 22, "PromptLab — in-memory optimization session report")
        canvas.drawRightString(letter[0] - 40, 22, f"Page {doc.page}")
        canvas.restoreState()

    @staticmethod
    def _add_section(story: list[Any], title: str, styles: Any) -> None:
        story.append(Spacer(1, 12))
        story.append(Paragraph(title, styles["Heading2"]))

    def _add_key_values(self, story: list[Any], rows: list[tuple[str, Any]], styles: Any) -> None:
        if not rows:
            return
        data = [[self._p(key, styles["Small"]), self._p(value, styles["BodyText"])] for key, value in rows]
        table = LongTable(data, colWidths=[150, 370], repeatRows=0, splitByRow=1, splitInRow=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D4D0")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F0EE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    def create_report(self, session: dict[str, Any]) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"{session['session_id']}-report.pdf")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#202124"),
        ))
        styles.add(ParagraphStyle(
            name="Small", parent=styles["BodyText"], fontSize=8, leading=10,
        ))
        styles.add(ParagraphStyle(
            name="Mono", parent=styles["Code"], fontName="Courier", fontSize=8, leading=11,
        ))

        story: list[Any] = [
            Spacer(1, 110),
            Paragraph("PromptLab", styles["CoverTitle"]),
            Paragraph("Prompt Optimization Report", styles["Heading1"]),
            Spacer(1, 24),
        ]
        self._add_key_values(story, [
            ("Session ID", session.get("session_id")),
            ("Created", session.get("created_at")),
            ("Completed", session.get("completed_at")),
            ("Status", session.get("status")),
            ("Stop reason", session.get("stop_reason")),
            ("Configuration", session.get("configuration", {})),
        ], styles)
        story.append(PageBreak())

        self._add_section(story, "1. Original Prompt and Context", styles)
        self._add_key_values(story, [
            ("Original prompt", session.get("original_prompt")),
            ("Additional context", session.get("optional_context")),
            ("Preferences", session.get("preferences")),
        ], styles)

        task = session.get("task_spec") or {}
        self._add_section(story, "2. Confirmed TaskIntent", styles)
        self._add_key_values(story, [(key, value) for key, value in task.items()], styles)

        rubric = session.get("rubric") or {}
        self._add_section(story, "3. Task-Specific Rubric", styles)
        rubric_rows = [[self._p("Criterion", styles["Small"]), self._p("Description", styles["Small"]),
                        self._p("Weight", styles["Small"]), self._p("Scoring guidance", styles["Small"])]]
        for item in rubric.get("criteria", []):
            rubric_rows.append([
                self._p(item.get("criterion"), styles["Small"]),
                self._p(item.get("description"), styles["Small"]),
                self._p(item.get("weight"), styles["Small"]),
                self._p(item.get("scoring_scale"), styles["Small"]),
            ])
        rubric_table = LongTable(
            rubric_rows, colWidths=[100, 215, 45, 160], repeatRows=1, splitByRow=1, splitInRow=1
        )
        rubric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D4D0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(rubric_table)

        self._add_section(story, "4. Samples", styles)
        for sample in session.get("samples", []):
            self._add_key_values(story, [
                ("Sample ID / origin", f"{sample.get('sample_id')} / {sample.get('origin')}"),
                ("Content", sample.get("content")),
            ], styles)

        candidates = session.get("candidates", [])
        by_id = {item.get("candidate_id"): item for item in candidates}
        self._add_section(story, "5. Candidate Pool and Lineage", styles)
        for candidate in candidates:
            parent = by_id.get(candidate.get("parent_id"), {})
            self._add_key_values(story, [
                ("Candidate / status / source", f"{candidate.get('candidate_id')} / {candidate.get('status')} / {candidate.get('source')}"),
                ("Parent / generation", f"{candidate.get('parent_id') or 'original'} / {candidate.get('generation')}"),
                ("Generation reason", candidate.get("generation_reason")),
                ("Preserved requirements", candidate.get("preserved_requirements", [])),
                ("New assumptions", candidate.get("new_assumptions", [])),
                ("Removed assumptions", candidate.get("removed_assumptions", [])),
                ("Parent prompt", parent.get("prompt_text") if parent else "—"),
                ("Complete candidate prompt", candidate.get("prompt_text")),
            ], styles)

        evaluations = session.get("evaluations", [])
        self._add_section(story, "6. Optimization Evaluation History", styles)
        for evaluation in evaluations:
            self._add_key_values(story, [
                ("Evaluation / iteration", f"{evaluation.get('evaluation_id')} / {evaluation.get('iteration')}"),
                ("Candidate / sample / status", f"{evaluation.get('candidate_id')} / {evaluation.get('sample_id')} / {evaluation.get('status')}"),
                ("Evaluation source", evaluation.get("evaluation_source")),
                ("Performer response", evaluation.get("response")),
                ("Authoritative reward", evaluation.get("authoritative_score")),
                ("Judge informational score", evaluation.get("judge_overall_score")),
                ("Criterion scores", evaluation.get("criterion_scores", [])),
                ("Strengths", evaluation.get("strengths", [])),
                ("Weaknesses", evaluation.get("weaknesses", [])),
                ("Evidence", evaluation.get("evidence", [])),
                ("Root cause", evaluation.get("root_cause", [])),
                ("Improvement suggestions", evaluation.get("improvement_suggestion", [])),
                ("Intent fidelity / unsupported assumptions", f"{evaluation.get('intent_fidelity')} / {evaluation.get('unsupported_assumptions')}"),
                ("Failure", evaluation.get("failure_reason") or evaluation.get("error")),
            ], styles)

        self._add_section(story, "7. Iteration and UCB History", styles)
        for iteration in session.get("iterations", []):
            self._add_key_values(story, [
                ("Iteration / sample", f"{iteration.get('iteration')} / {iteration.get('sample_id')}"),
                ("Active candidates", iteration.get("active_candidates")),
                ("Evaluated candidates", iteration.get("evaluated_candidates")),
                ("Parents / children", f"{iteration.get('parent_candidates')} / {iteration.get('edited_candidates')}"),
                ("Best score", iteration.get("best_score")),
                ("UCB decisions", iteration.get("ucb_decisions")),
            ], styles)
        for history in session.get("ucb_history", []):
            self._add_key_values(story, [
                ("UCB history iteration", history.get("iteration")),
                ("Total pulls before round", history.get("total_pulls_before_round")),
                ("Selected candidate IDs", history.get("selected_candidate_ids")),
                ("Selection decisions", history.get("decisions")),
            ], styles)

        self._add_section(story, "8. Session Insight Store", styles)
        for insight in session.get("insights", []):
            self._add_key_values(story, [(key, value) for key, value in insight.items()], styles)
        if not session.get("insights"):
            story.append(self._p("No insights were recorded.", styles["BodyText"]))

        best_scores = [
            (item.get("iteration"), item.get("best_score"))
            for item in session.get("iterations", [])
            if item.get("best_score") is not None
        ]
        if len(best_scores) >= 2:
            self._add_section(story, "9. Quality by Iteration", styles)
            chart = Drawing(480, 230)
            plot = LinePlot()
            plot.x = 45
            plot.y = 35
            plot.width = 390
            plot.height = 165
            plot.data = [best_scores]
            plot.xValueAxis.valueMin = min(item[0] for item in best_scores)
            plot.xValueAxis.valueMax = max(item[0] for item in best_scores)
            plot.yValueAxis.valueMin = 0
            plot.yValueAxis.valueMax = 1
            plot.lines[0].strokeColor = colors.HexColor("#333333")
            plot.lines[0].strokeWidth = 1.5
            chart.add(plot)
            story.append(chart)

        final = session.get("final_evaluation") or {}
        self._add_section(story, "10. Final Evaluation and Original Baseline", styles)
        baseline = final.get("baseline") or {}
        self._add_key_values(story, [
            ("Stop reason", final.get("stop_reason") or session.get("stop_reason")),
            ("Baseline prompt", baseline.get("prompt_text", session.get("original_prompt"))),
            ("Baseline mean score", baseline.get("score")),
        ], styles)
        for evaluation in baseline.get("evaluations", []):
            self._add_key_values(story, [
                ("Baseline sample / score", f"{evaluation.get('sample_id')} / {evaluation.get('authoritative_score')}"),
                ("Baseline response", evaluation.get("response")),
                ("Baseline criterion scores", evaluation.get("criterion_scores", [])),
                ("Baseline feedback", evaluation.get("feedback", {})),
            ], styles)
        for rank, item in enumerate(final.get("final_ranking", []), start=1):
            self._add_key_values(story, [
                ("Final rank / candidate / score", f"{rank} / {item.get('candidate_id')} / {item.get('score')}"),
                ("Final prompt", item.get("prompt_text")),
                ("Lineage", f"{item.get('parent_id') or 'original'} / generation {item.get('generation')}"),
            ], styles)
            for evaluation in item.get("final_evaluations", []):
                self._add_key_values(story, [
                    ("Final sample / score", f"{evaluation.get('sample_id')} / {evaluation.get('authoritative_score')}"),
                    ("Final response", evaluation.get("response")),
                    ("Final criterion scores", evaluation.get("criterion_scores", [])),
                    ("Final feedback", evaluation.get("feedback", {})),
                ], styles)

        self._add_section(story, "11. Top 3 Prompts", styles)
        for rank, item in enumerate(final.get("top3", []), start=1):
            self._add_key_values(story, [
                ("Rank / candidate / score", f"{rank} / {item.get('candidate_id')} / {item.get('score')}"),
                ("Parent / generation", f"{item.get('parent_id') or 'original'} / {item.get('generation')}"),
                ("Complete prompt", item.get("prompt_text")),
            ], styles)

        self._add_section(story, "12. Limitations and Methodology", styles)
        limitations = [
            "LLM judge scores are estimates, not objective ground truth.",
            "Subjective tasks may not have a uniquely correct response.",
            "Optimization is bounded by the recorded call and iteration budgets.",
            "UCB is a candidate-selection strategy, not proof of global optimality.",
            "Mock-source evaluations are explicitly marked and are not live provider evaluations.",
        ]
        for limitation in limitations:
            story.append(self._p(f"- {limitation}", styles["BodyText"]))
        story.append(Spacer(1, 8))
        self._add_key_values(story, [
            ("Model configuration", session.get("model_configuration", {})),
            ("Call count / budget", f"{session.get('llm_call_count')} / {session.get('configuration', {}).get('max_llm_calls')}"),
            ("Session status / reason", f"{session.get('status')} / {session.get('stop_reason')}"),
            ("Errors", session.get("errors", [])),
        ], styles)

        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=42,
            bottomMargin=48,
            title="PromptLab Prompt Optimization Report",
            author="PromptLab",
        )
        doc.build(story, onFirstPage=self._page_decor, onLaterPages=self._page_decor)
        return filename
