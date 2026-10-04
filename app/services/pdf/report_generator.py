from __future__ import annotations

import os
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
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

    @staticmethod
    def _score_percent(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not numeric or numeric < 0:
            return None
        if numeric <= 1.0:
            return numeric * 100.0
        if numeric <= 100.0:
            return numeric
        return None

    @staticmethod
    def _score_label(value: Any) -> str:
        percent = ReportGenerator._score_percent(value)
        if percent is None:
            return "—"
        return f"{percent:.1f}%"

    @staticmethod
    def _preview_text(text: Any, limit: int = 220) -> str:
        value = str(text if text is not None else "")
        value = value.strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                dumped = value.dict()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        return {}

    @staticmethod
    def _coerce_items(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        return [str(value)]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            text = str(item).strip()
            lowered = text.lower()
            if not text or lowered in seen:
                continue
            seen.add(lowered)
            result.append(text)
        return result

    @staticmethod
    def _details_for_task(task_spec: dict[str, Any]) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        goal = (task_spec or {}).get("primary_intent")
        if goal:
            rows.append(("Goal", str(goal)))
        audience = (task_spec or {}).get("audience")
        if audience:
            rows.append(("Audience", str(audience)))
        output_format = (task_spec or {}).get("output_format")
        if output_format:
            rows.append(("Expected output", str(output_format)))
        requirements = ReportGenerator._dedupe((task_spec or {}).get("explicit_requirements", []) + (task_spec or {}).get("inferred_requirements", []))
        if requirements:
            rows.append(("Key requirements", "• " + "\n• ".join(requirements)))
        constraints = ReportGenerator._dedupe((task_spec or {}).get("constraints", []))
        if constraints:
            rows.append(("Constraints", "• " + "\n• ".join(constraints)))
        ambiguities = ReportGenerator._dedupe((task_spec or {}).get("ambiguities", []))
        if ambiguities:
            rows.append(("Important considerations", "• " + "\n• ".join(ambiguities)))
        return rows

    @staticmethod
    def _collect_feedback_texts(
        candidate: dict[str, Any],
        field_name: str,
        evaluations: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        texts: list[str] = []
        for evaluation in candidate.get("final_evaluations", []) or []:
            feedback = evaluation.get("feedback") or {}
            texts.extend(ReportGenerator._coerce_items(feedback.get(field_name)))
        candidate_id = candidate.get("candidate_id")
        for evaluation in evaluations or []:
            if evaluation.get("candidate_id") == candidate_id:
                texts.extend(ReportGenerator._coerce_items(evaluation.get(field_name)))
        return ReportGenerator._dedupe(texts)

    @staticmethod
    def _count_evaluated_candidates(session: dict[str, Any]) -> int:
        candidate_ids = {
            evaluation.get("candidate_id")
            for evaluation in session.get("evaluations") or []
            if evaluation.get("status") in {"success", "mock"}
            and evaluation.get("candidate_id")
        }
        return len(candidate_ids)

    @staticmethod
    def _build_summary_sentence(
        top_candidate: dict[str, Any] | None,
        evaluations: list[dict[str, Any]] | None = None,
    ) -> str:
        if top_candidate:
            feedback = (
                ReportGenerator._collect_feedback_texts(top_candidate, "strengths", evaluations)
                or ReportGenerator._collect_feedback_texts(top_candidate, "evidence", evaluations)
            )
            if feedback:
                return f"Final evaluation feedback for the top-ranked prompt highlighted: {feedback[0]}"
        return "No specific improvement evidence was recorded for the best candidate."

    @staticmethod
    def _iter_score_points(session: dict[str, Any]) -> list[tuple[str, float]]:
        points: list[tuple[str, float]] = []
        baseline = (session.get("final_evaluation") or {}).get("baseline") or {}
        baseline_score = baseline.get("score")
        if baseline_score is not None:
            points.append(("Original", float(baseline_score)))
        iterations = session.get("iterations") or []
        for index, iteration in enumerate(iterations, start=1):
            value = iteration.get("best_score")
            if value is not None:
                try:
                    points.append((f"Iteration {index}", float(value)))
                except (TypeError, ValueError):
                    continue
        if not points:
            best = session.get("current_best_quality")
            if best is not None:
                points.append(("Original", float(best)))
        return points

    def _page_decor(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7D7D2"))
        canvas.line(40, 34, letter[0] - 40, 34)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(40, 22, "PromptLab — Prompt Optimization Report")
        canvas.drawRightString(letter[0] - 40, 22, f"Page {doc.page}")
        canvas.restoreState()

    @staticmethod
    def _add_section(story: list[Any], title: str, styles: Any) -> None:
        story.append(Spacer(1, 12))
        story.append(Paragraph(title, styles["Heading2"]))

    def _add_key_values(self, story: list[Any], rows: list[tuple[str, Any]], styles: Any) -> None:
        if not rows:
            return
        data: list[list[Any]] = []
        for key, value in rows:
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and not value:
                continue
            data.append([
                self._p(key, styles["Small"]),
                self._p(value if not isinstance(value, (list, tuple)) else "• " + "\n• ".join(str(item) for item in value), styles["BodyText"]),
            ])
        if not data:
            return
        table = LongTable(data, colWidths=[165, 345], repeatRows=0, splitByRow=1, splitInRow=1)
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
        session_id = str(session.get("session_id") or "session")
        filename = os.path.join(self.output_dir, f"{session_id}-report.pdf")

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=26, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#202124"),
        ))
        styles.add(ParagraphStyle(
            name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, spaceAfter=8, textColor=colors.HexColor("#202124"),
        ))
        styles.add(ParagraphStyle(
            name="PromptBox", parent=styles["BodyText"], fontName="Helvetica",
            fontSize=9, leading=12, borderColor=colors.HexColor("#C7C7C3"),
            borderWidth=1, borderPadding=8, backColor=colors.HexColor("#F5F5F4"),
        ))
        styles.add(ParagraphStyle(
            name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11,
        ))

        story: list[Any] = []
        task_spec = self._as_mapping(session.get("task_spec"))
        final = self._as_mapping(session.get("final_evaluation"))
        top3 = final.get("top3") or final.get("final_ranking") or []
        top_candidate = top3[0] if top3 else {}
        baseline = self._as_mapping(final.get("baseline"))
        original_quality = self._score_percent(baseline.get("score"))
        best_quality = self._score_percent(top_candidate.get("score"))
        improvement_points = None
        if original_quality is not None and best_quality is not None:
            improvement_points = best_quality - original_quality

        session_evaluations = session.get("evaluations") or []
        summary_sentence = self._build_summary_sentence(top_candidate, session_evaluations)
        original_prompt = session.get("original_prompt") or ""
        best_prompt = (top_candidate.get("prompt_text") or original_prompt).strip()

        story.extend([
            Spacer(1, 36),
            Paragraph("PromptLab", styles["CoverTitle"]),
            Paragraph("Prompt Optimization Report", styles["Heading1"]),
            Spacer(1, 20),
            Paragraph("1. EXECUTIVE SUMMARY", styles["SectionTitle"]),
            self._p(summary_sentence, styles["BodyText"]),
            Spacer(1, 12),
            Paragraph("Original Prompt", styles["Heading2"]),
            self._p(self._preview_text(original_prompt, 260), styles["BodyText"]),
            Spacer(1, 8),
            Paragraph("Best Optimized Prompt", styles["Heading2"]),
            self._p(self._preview_text(best_prompt, 260), styles["BodyText"]),
            Spacer(1, 12),
        ])
        summary_rows = [
            ("Original Quality", self._score_label(baseline.get("score"))),
            ("Best Optimized Quality", self._score_label(top_candidate.get("score"))),
            ("Improvement", f"+{improvement_points:.1f} points" if improvement_points is not None else "—"),
            ("Iterations", str(len(session.get("iterations") or []))),
            ("Candidates Evaluated", str(self._count_evaluated_candidates(session))),
            ("Model Calls", str(session.get("llm_call_count") or 0)),
        ]
        self._add_key_values(story, summary_rows, styles)
        story.append(PageBreak())

        self._add_section(story, "2. TASK UNDERSTANDING", styles)
        rows = self._details_for_task(task_spec)
        if rows:
            detail_table = LongTable(
                [[self._p(title, styles["Small"]), self._p(value, styles["BodyText"])] for title, value in rows],
                colWidths=[155, 355], repeatRows=0, splitByRow=1, splitInRow=1,
            )
            detail_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D4D0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F0EE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(detail_table)
        else:
            story.append(Paragraph("No task understanding details were recorded in the session.", styles["BodyText"]))

        self._add_section(story, "3. EVALUATION RUBRIC", styles)
        story.append(Paragraph("PromptLab evaluates candidate prompts by comparing the quality of the responses they produce against this task-specific rubric.", styles["BodyText"]))
        rubric = self._as_mapping(session.get("rubric"))
        criteria = rubric.get("criteria") or []
        if criteria:
            rubric_rows = [[self._p("Criterion", styles["Small"]), self._p("Weight", styles["Small"]), self._p("What it measures", styles["Small"])] ]
            for item in criteria:
                weight_value = item.get("weight")
                rubric_rows.append([
                    self._p(item.get("criterion") or "—", styles["BodyText"]),
                    self._p(f"{float(weight_value) * 100:.0f}%" if weight_value is not None else "—", styles["BodyText"]),
                    self._p(item.get("description") or item.get("scoring_scale") or "—", styles["BodyText"]),
                ])
            rubric_table = LongTable(rubric_rows, colWidths=[170, 70, 290], repeatRows=1, splitByRow=1, splitInRow=1)
            rubric_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D4D0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(rubric_table)
        else:
            story.append(Paragraph("No rubric was recorded for this session.", styles["BodyText"]))

        self._add_section(story, "4. OPTIMIZATION JOURNEY", styles)
        score_points = self._iter_score_points(session)
        if score_points:
            if score_points[0][0] != "Original":
                score_points = [("Original", score_points[0][1])] + score_points
            score_labels = [label for label, _ in score_points]
            score_values = [self._score_percent(value) or 0.0 for _, value in score_points]
            chart = Drawing(500, 210)
            bar = VerticalBarChart()
            bar.x = 40
            bar.y = 35
            bar.width = 430
            bar.height = 140
            bar.data = [score_values]
            bar.categoryAxis.categoryNames = score_labels
            bar.categoryAxis.labels.dx = 0
            bar.categoryAxis.labels.dy = -3
            bar.categoryAxis.labels.fontName = 'Helvetica'
            bar.categoryAxis.labels.fontSize = 8
            bar.valueAxis.valueMin = 0
            bar.valueAxis.valueMax = max(100.0, max(score_values) * 1.15)
            bar.valueAxis.valueStep = 10
            bar.barLabelFormat = '%d%%'
            bar.valueAxis.labelTextFormat = '%d%%'
            chart.add(bar)
            story.append(chart)
        else:
            story.append(Paragraph("Quality progression could not be plotted because insufficient evaluation data was recorded.", styles["BodyText"]))

        progression = []
        if baseline.get("score") is not None:
            progression.append(("Original", baseline.get("score")))
        for index, iteration in enumerate(session.get("iterations") or [], start=1):
            value = iteration.get("best_score")
            if value is not None:
                progression.append((f"Iteration {index}", value))
        if progression:
            for label, value in progression:
                story.append(Paragraph(f"{label}: {self._score_label(value)}", styles["BodyText"]))
        else:
            story.append(Paragraph("No iteration scores were recorded for this session.", styles["BodyText"]))

        improvement_bullets: list[str] = []
        for candidate in top3:
            improvement_bullets.extend(self._collect_feedback_texts(candidate, "strengths", session_evaluations))
        if not improvement_bullets:
            for candidate in top3:
                improvement_bullets.extend(self._collect_feedback_texts(candidate, "evidence", session_evaluations))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Key improvements", styles["Heading2"]))
        distinct_improvements = self._dedupe(improvement_bullets)[:3]
        if distinct_improvements:
            for idx, point in enumerate(distinct_improvements, start=1):
                story.append(self._p(f"{idx}. {point}", styles["BodyText"]))
        else:
            story.append(self._p("No specific improvement evidence was recorded for the top prompts.", styles["BodyText"]))

        self._add_section(story, "5. FINAL OPTIMIZED PROMPT", styles)
        if best_prompt:
            story.append(Paragraph(best_prompt, styles["PromptBox"]))
        else:
            story.append(Paragraph("No final optimized prompt was recorded for this session.", styles["BodyText"]))
        story.append(Spacer(1, 8))
        meta_rows = [
            ("Final Quality", self._score_label(top_candidate.get("score"))),
            ("Original Quality", self._score_label(baseline.get("score"))),
            ("Improvement", f"+{improvement_points:.1f} points / +{(improvement_points * 100.0 / original_quality):.1f}%" if improvement_points is not None and original_quality not in (None, 0) else "—"),
        ]
        if session.get("current_best_prompt_token_count") is not None:
            meta_rows.append(("Prompt Length", f"{int(session['current_best_prompt_token_count'])} tokens"))
        self._add_key_values(story, meta_rows, styles)
        story.append(Spacer(1, 8))
        story.append(Paragraph("WHY THIS PROMPT IS BETTER", styles["Heading2"]))
        why = self._collect_feedback_texts(top_candidate, "strengths", session_evaluations)
        if not why:
            why = self._collect_feedback_texts(top_candidate, "evidence", session_evaluations)
        if not why:
            why = ["No specific improvement evidence was recorded for this candidate."]
        for item in self._dedupe(why)[:4]:
            story.append(self._p(f"• {item}", styles["BodyText"]))

        self._add_section(story, "6. TOP 3 OPTIMIZED PROMPTS", styles)
        if not top3:
            story.append(Paragraph("No final ranked prompts were recorded for this session.", styles["BodyText"]))
        for rank, item in enumerate(top3[:3], start=1):
            label = "Best Overall" if rank == 1 else "Alternative"
            story.append(Paragraph(f"#{rank} — {label}", styles["Heading2"]))
            story.append(Paragraph(f"Quality: {self._score_label(item.get('score'))}", styles["BodyText"]))
            story.append(Paragraph("Prompt:", styles["BodyText"]))
            story.append(self._p(item.get("prompt_text") or "—", styles["PromptBox"]))
            why_rank = self._collect_feedback_texts(item, "strengths", session_evaluations) or self._collect_feedback_texts(item, "evidence", session_evaluations)
            if why_rank:
                story.append(Paragraph("Why it performed well:", styles["BodyText"]))
                story.append(self._p(self._dedupe(why_rank)[0], styles["BodyText"]))
            else:
                story.append(self._p("Why it performed well: No specific improvement evidence was recorded for this candidate.", styles["BodyText"]))
            weaknesses = self._collect_feedback_texts(item, "weaknesses", session_evaluations)
            if weaknesses:
                story.append(Paragraph("Remaining weakness:", styles["BodyText"]))
                story.append(self._p(weaknesses[0], styles["BodyText"]))
            story.append(Spacer(1, 12))

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
