from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app.schemas.task import IntentEditRequest, OptimizationStartRequest
from app.services.optimizer.optimization_service import OptimizationService
from app.services.pdf.report_generator import ReportGenerator
from app.state.session_state import get_session, update_session

router = APIRouter(prefix="/api/optimization")
service = OptimizationService()
logger = logging.getLogger(__name__)


@router.post("/start")
async def start_optimization(payload: OptimizationStartRequest):
    try:
        session = service.start(payload)
        return {
            "success": True,
            "data": {
                "sessionId": session["session_id"],
                "status": session["status"],
                "taskSpec": session["task_spec"],
                "configuration": session["configuration"],
            },
        }
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": {"code": "start_error", "message": str(exc), "details": {}},
        })
    except Exception as exc:
        logger.error("Task understanding failed (%s).", type(exc).__name__)
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": {"code": "task_understanding_failed", "message": "Task understanding failed. Please retry.", "details": {}},
        })


@router.post("/{session_id}/confirm-understanding")
async def confirm_understanding(
    session_id: str,
    payload: IntentEditRequest,
    background_tasks: BackgroundTasks,
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        session = service.confirm_and_prepare(session_id, payload.task_spec.model_dump())
        background_tasks.add_task(service.run_optimization, session_id)
    except (ValueError, RuntimeError, ValidationError) as exc:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": {"code": "confirmation_error", "message": str(exc), "details": {}},
        })
    return {
        "success": True,
        "data": {
            "sessionId": session_id,
            "status": session["status"],
            "candidateCount": len(session["candidates"]),
            "rubric": session["rubric"],
        },
    }


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "data": session}


@router.get("/{session_id}/progress")
async def get_progress(session_id: str):
    try:
        return {"success": True, "data": service.get_status(session_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/{session_id}/candidates")
async def get_candidates(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "data": {
        "candidates": session.get("candidates", []),
        "active_candidate_ids": session.get("active_candidate_ids", []),
    }}


@router.get("/{session_id}/iterations")
async def get_iterations(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "data": {"iterations": session.get("iterations", [])}}


@router.get("/{session_id}/results")
async def get_results(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "data": {
        "status": session.get("status"),
        "stopReason": session.get("stop_reason"),
        "finalEvaluation": session.get("final_evaluation"),
        "bestScore": max(
            (candidate.get("mean_reward", 0.0) for candidate in session.get("candidates", [])),
            default=0.0,
        ),
        "errors": session.get("errors", []),
    }}


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        generator = ReportGenerator(output_dir="reports")
        report_path = generator.create_report(session)
        update_session(session_id, report_path=report_path, report_error=None)
        return FileResponse(report_path, media_type="application/pdf", filename=Path(report_path).name)
    except Exception as exc:
        logger.error("PDF generation failed for session %s (%s).", session_id, type(exc).__name__)
        update_session(session_id, report_error="Report generation failed.")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": {"code": "pdf_generation_failed", "message": "Optimization results remain available, but the PDF could not be generated.", "details": {}},
        })


@router.post("/{session_id}/retry")
async def retry(session_id: str):
    raise HTTPException(status_code=409, detail="Manual rounds are disabled. Confirm the understanding to run the configured optimization loop.")


@router.post("/{session_id}/cancel")
async def cancel(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    update_session(session_id, status="cancelled", stop_reason="cancelled")
    return {"success": True, "data": {"sessionId": session_id, "status": "cancelled"}}
