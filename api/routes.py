"""
FastAPI Route Handlers
=======================
All REST endpoints for the Marine Debris Detection API.

Base path: /api
  POST   /api/analyze                       — Upload + run full pipeline
  GET    /api/results/{job_id}              — Get job status + result summary
  GET    /api/results/{job_id}/report.json  — Download full JSON report
  GET    /api/results/{job_id}/report.csv   — Download CSV report (GIS-ready)
  GET    /api/results/{job_id}/annotated    — Download annotated sonar image
  DELETE /api/results/{job_id}              — Delete job and its files
  GET    /api/jobs                          — List all jobs
  GET    /api/classes                       — Return class definitions
  GET    /health                            — Health check
"""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    BackgroundTasks,
)
from fastapi.responses import FileResponse, JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    CLASS_NAMES,
    CLASS_LABELS,
    CLASS_COLORS,
    UPLOADS_DIR,
)
from api.job_store import JobStore, JobStatus, get_job_store
from api.pipeline import DetectionPipeline, get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Type aliases for dependency injection ────────────────────────────────────
PipelineDep = Annotated[DetectionPipeline, Depends(get_pipeline)]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]


# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check(pipeline: PipelineDep):
    """Returns API health status and model information."""
    return {
        "status": "ok",
        "api_version": "1.0.0",
        "model": pipeline.model_info,
    }


# ─── Class Definitions ────────────────────────────────────────────────────────

@router.get("/api/classes", tags=["reference"])
async def get_classes():
    """Return all detectable debris class definitions."""
    classes = []
    for cls_id, name in CLASS_NAMES.items():
        classes.append({
            "class_id": cls_id,
            "class_name": name,
            "class_label": CLASS_LABELS.get(cls_id, name),
            "color": CLASS_COLORS.get(cls_id, "#ffffff"),
        })
    return {"classes": classes, "total": len(classes)}


@router.get("/api/samples", tags=["reference"])
async def list_samples():
    """List sample sonar images available for immediate analysis testing."""
    samples = []
    for p in sorted(UPLOADS_DIR.glob("sonar_test_*.png")):
        samples.append({
            "filename": p.name,
            "url": f"/uploads/{p.name}",
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    return {"samples": samples, "total": len(samples)}


# ─── Job Listing ──────────────────────────────────────────────────────────────

@router.get("/api/jobs", tags=["jobs"])
async def list_jobs(store: JobStoreDep):
    """List all known jobs and their statuses."""
    return {"jobs": store.list_jobs()}


# ─── Main Analysis Endpoint ───────────────────────────────────────────────────

@router.post("/api/analyze", tags=["detection"])
async def analyze_sonar(
    background_tasks: BackgroundTasks,
    pipeline: PipelineDep,
    store: JobStoreDep,
    file: UploadFile = File(..., description="Sonar image (TIFF, PNG, JPG, BMP)"),
    origin_lat: float = Form(12.9716, description="Survey origin latitude"),
    origin_lon: float = Form(77.5946, description="Survey origin longitude"),
    heading_deg: float = Form(0.0, description="Vessel heading in degrees (0=North)"),
    sonar_range_m: float = Form(75.0, description="One-side sonar swath in metres"),
    towfish_altitude_m: float = Form(0.0, description="Towfish altitude above seabed in metres"),
    layback_m: float = Form(0.0, description="Towfish layback distance behind vessel GPS in metres"),
    frequency: Optional[str] = Form(None, description="Operating frequency (e.g. 100kHz, 300kHz, 600kHz, 900kHz)"),
):
    """
    Upload a sonar image and run the full debris detection pipeline.

    Returns immediately with a job_id; processing is synchronous for simplicity
    (the response is returned once processing is complete).

    For large files (>10 MB) consider the async variant via WebSocket.
    """
    # ── Validate file extension ───────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Read and validate file size ───────────────────────────────────
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_UPLOAD_SIZE_MB} MB",
        )

    # ── Create job ────────────────────────────────────────────────────
    job = store.create_job(source_filename=file.filename or "upload")
    store.set_processing(job.job_id)

    logger.info(
        f"Job {job.job_id} — file: {file.filename} "
        f"({size_mb:.2f} MB) — params: lat={origin_lat} lon={origin_lon} "
        f"heading={heading_deg}° range={sonar_range_m}m altitude={towfish_altitude_m}m "
        f"layback={layback_m}m freq={frequency}"
    )

    # ── Run pipeline ──────────────────────────────────────────────────
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.run(
                image_bytes=image_bytes,
                job_id=job.job_id,
                source_filename=file.filename or "upload",
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                heading_deg=heading_deg,
                sonar_range_m=sonar_range_m,
                towfish_altitude_m=towfish_altitude_m,
                layback_m=layback_m,
                frequency=frequency if frequency and frequency != "auto" else None,
            ),
        )
    except ValueError as e:
        store.set_error(job.job_id, str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        store.set_error(job.job_id, str(e))
        logger.exception(f"Pipeline error for job {job.job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # ── Store results ─────────────────────────────────────────────────
    store.set_done(
        job_id=job.job_id,
        result_data=result.json_report,
        annotated_path=result.annotated_image_path,
        clean_path=result.clean_image_path,
        json_path=result.json_report_path,
        csv_path=result.csv_report_path,
        geojson_path=result.geojson_report_path,
    )

    # ── Build response ────────────────────────────────────────────────
    report = result.json_report
    return {
        "job_id": job.job_id,
        "status": "done",
        "source_file": file.filename,
        "processing_time_ms": round(result.total_time_ms, 1),
        "total_anomalies": report.get("total_anomalies", 0),
        "report_id": report.get("report_id"),
        "image_width": result.annotated_image.shape[1] if hasattr(result, "annotated_image") and result.annotated_image is not None else 1000,
        "image_height": result.annotated_image.shape[0] if hasattr(result, "annotated_image") and result.annotated_image is not None else 512,
        "summary": report.get("summary", {}),
        "anomalies": report.get("anomalies", []),
        "links": {
            "results": f"/api/results/{job.job_id}",
            "clean_image": f"/api/results/{job.job_id}/clean",
            "annotated_image": f"/api/results/{job.job_id}/annotated",
            "report_json": f"/api/results/{job.job_id}/report.json",
            "report_csv": f"/api/results/{job.job_id}/report.csv",
            "report_geojson": f"/api/results/{job.job_id}/report.geojson",
        },
    }


# ─── Result Retrieval ─────────────────────────────────────────────────────────

@router.get("/api/results/{job_id}", tags=["results"])
async def get_result(job_id: str, store: JobStoreDep):
    """
    Get the status and full result data for a job.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = job.to_dict()

    if job.status == JobStatus.DONE and job.result_data:
        response["report"] = job.result_data
        response["links"] = {
            "annotated_image": f"/api/results/{job_id}/annotated",
            "report_json": f"/api/results/{job_id}/report.json",
            "report_csv": f"/api/results/{job_id}/report.csv",
            "report_geojson": f"/api/results/{job_id}/report.geojson",
        }
    elif job.status == JobStatus.ERROR:
        response["error"] = job.error_message

    return response


@router.get("/api/results/{job_id}/report.json", tags=["results"])
async def download_json_report(job_id: str, store: JobStoreDep):
    """Download the full JSON detection report."""
    job = _get_done_job(store, job_id)

    if not job.json_report_path or not Path(job.json_report_path).exists():
        raise HTTPException(status_code=404, detail="JSON report file not found on disk")

    return FileResponse(
        path=job.json_report_path,
        media_type="application/json",
        filename=f"sonar_report_{job_id}.json",
    )


@router.get("/api/results/{job_id}/report.csv", tags=["results"])
async def download_csv_report(job_id: str, store: JobStoreDep):
    """Download the CSV detection report (GIS-ready)."""
    job = _get_done_job(store, job_id)

    if not job.csv_report_path or not Path(job.csv_report_path).exists():
        raise HTTPException(status_code=404, detail="CSV report file not found on disk")

    return FileResponse(
        path=job.csv_report_path,
        media_type="text/csv",
        filename=f"sonar_report_{job_id}.csv",
    )


@router.get("/api/results/{job_id}/report.geojson", tags=["results"])
async def download_geojson_report(job_id: str, store: JobStoreDep):
    """Download the GeoJSON detection report (GIS-ready for QGIS/ArcGIS)."""
    job = _get_done_job(store, job_id)

    if not job.geojson_report_path or not Path(job.geojson_report_path).exists():
        raise HTTPException(status_code=404, detail="GeoJSON report file not found on disk")

    return FileResponse(
        path=job.geojson_report_path,
        media_type="application/geo+json",
        filename=f"sonar_report_{job_id}.geojson",
    )


@router.get("/api/results/{job_id}/annotated", tags=["results"])
async def download_annotated_image(job_id: str, store: JobStoreDep):
    """Download the annotated sonar image with detection overlays."""
    job = _get_done_job(store, job_id)

    if not job.annotated_image_path or not Path(job.annotated_image_path).exists():
        raise HTTPException(status_code=404, detail="Annotated image not found on disk")

    return FileResponse(
        path=job.annotated_image_path,
        media_type="image/jpeg",
        filename=f"annotated_{job_id}.jpg",
    )


@router.get("/api/results/{job_id}/clean", tags=["results"])
async def download_clean_image(job_id: str, store: JobStoreDep):
    """Download or display the clean preprocessed sonar image without burned-in overlays."""
    job = _get_done_job(store, job_id)

    target_path = job.clean_image_path if (job.clean_image_path and Path(job.clean_image_path).exists()) else job.annotated_image_path
    if not target_path or not Path(target_path).exists():
        raise HTTPException(status_code=404, detail="Clean sonar image not found on disk")

    return FileResponse(
        path=target_path,
        media_type="image/jpeg",
        filename=f"clean_{job_id}.jpg",
    )


# ─── Job Deletion ─────────────────────────────────────────────────────────────

@router.delete("/api/results/{job_id}", tags=["jobs"])
async def delete_job(job_id: str, store: JobStoreDep):
    """Delete a job and its associated result files."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    deleted = store.delete(job_id)
    return {"deleted": deleted, "job_id": job_id}


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_done_job(store: JobStore, job_id: str):
    """Retrieve a completed job or raise an appropriate HTTP error."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job is still processing")
    if job.status == JobStatus.ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.error_message}",
        )
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Job not complete (status: {job.status})")
    return job
