from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.model_service import ModelService


class PredictionPayload(BaseModel):
    age: float = Field(..., gt=0)
    income: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=0)
    years_employed: float = Field(..., ge=0)
    education: str = Field(..., min_length=1)
    employment_status: str = Field(..., min_length=1)
    marital_status: str = Field(..., min_length=1)
    housing_type: str = Field(..., min_length=1)
    application_date: str = Field(..., min_length=1)


class PredictRequest(PredictionPayload):
    model: Optional[str] = Field(default=None, description="Optional model name to target.")


class BatchPredictRequest(BaseModel):
    records: List[PredictionPayload] = Field(..., min_length=1)
    model: Optional[str] = Field(default=None, description="Optional model name to target.")


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: int
    models_failed: int = 0
    failed_models: Optional[List[Dict[str, str]]] = None


class ModelSummary(BaseModel):
    name: str
    type: str
    status: str


class ModelsResponse(BaseModel):
    models: List[ModelSummary]


class ModelInfoResponse(BaseModel):
    name: str
    type: str
    task_type: str
    algorithm: str
    status: str
    supported_prediction_mode: str


class PredictionResponse(BaseModel):
    success: bool = True
    model: str
    prediction: Any
    confidence: Optional[float] = None
    probability: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    success: bool = True
    model: str
    predictions: List[Any]
    count: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


def create_app(model_service: Optional[ModelService] = None) -> FastAPI:
    service = model_service or ModelService()
    app = FastAPI(
        title="CARIVIX AI Model Service",
        version="1.0.0",
        description="Production inference API for the CARIVIX trained ML models.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.model_service = service

    # Register AI integration routes (nlp_module + RAG + ML placeholder)
    try:
        from ai_integration import register_ai_routes
        register_ai_routes(app)
    except Exception:
        # If registration fails, continue without AI routes — the rest of the API stays functional
        pass

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root to the interactive API docs."""
        return RedirectResponse(url="/docs")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": "Invalid request data."},
        )

    @app.get("/health", response_model=HealthResponse, summary="Check service health")
    async def health() -> Dict[str, Any]:
        """Return whether the service is running and whether at least one model is loaded.

        This endpoint now includes information about any model files that failed to load during
        service initialization so operators can diagnose configuration or serialization issues.
        """
        loaded_models = app.state.model_service.list_models()
        load_errors = []
        if hasattr(app.state.model_service, "get_load_errors"):
            load_errors = app.state.model_service.get_load_errors()

        loaded = len(loaded_models)
        failed_count = len(load_errors)

        # If nothing was loaded at startup, attempt a reload in case models were added later
        if loaded == 0 and hasattr(app.state.model_service, "reload_models"):
            try:
                app.state.model_service.reload_models()
                loaded_models = app.state.model_service.list_models()
                if hasattr(app.state.model_service, "get_load_errors"):
                    load_errors = app.state.model_service.get_load_errors()
                loaded = len(loaded_models)
                failed_count = len(load_errors)
            except Exception:
                # Do not fail the health endpoint; report degraded status and any load errors
                logger = getattr(app.state.model_service, "logger", None)

        if loaded > 0:
            status_value = "healthy"
        else:
            # If no models loaded the service is degraded regardless of whether there were errors
            status_value = "degraded"

        return {
            "status": status_value,
            "service": "CARIVIX AI Model Service",
            "models_loaded": loaded,
            "models_failed": failed_count,
            "failed_models": load_errors or None,
        }

    @app.get("/api/v1/models", response_model=ModelsResponse, summary="List available models")
    async def list_models() -> Dict[str, List[Dict[str, str]]]:
        """Return the models that are currently available for inference."""
        return {"models": app.state.model_service.list_models()}

    @app.get("/api/v1/model/info", response_model=ModelInfoResponse, summary="Get default model metadata")
    async def model_info() -> Dict[str, Any]:
        """Return metadata for the currently selected default model."""
        if not app.state.model_service.has_loaded_models():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model is currently available.",
            )
        return app.state.model_service.get_model_info()

    @app.post("/api/v1/predict", response_model=PredictionResponse, summary="Run single-record inference")
    async def predict(payload: PredictRequest) -> Dict[str, Any]:
        """Run prediction on a single record using the configured preprocessing pipeline and loaded model."""
        if not app.state.model_service.has_loaded_models():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model is currently available.",
            )

        service = app.state.model_service
        try:
            request_payload = payload.model_dump(exclude={"model"}, exclude_none=True)
            result = service.predict(request_payload, model_name=payload.model)
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is unavailable.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request data.") from exc
        except RuntimeError as exc:
            if "No trained model" in str(exc):
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No trained model is currently available.") from exc
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from exc

    @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, summary="Run batch inference")
    async def predict_batch(payload: BatchPredictRequest) -> Dict[str, Any]:
        """Run prediction for multiple records using the same model and preprocessing pipeline."""
        if not app.state.model_service.has_loaded_models():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model is currently available.",
            )

        service = app.state.model_service
        try:
            records = [record.model_dump(exclude_none=True) for record in payload.records]
            result = service.predict_batch(records, model_name=payload.model)
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is unavailable.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request data.") from exc
        except RuntimeError as exc:
            if "No trained model" in str(exc):
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No trained model is currently available.") from exc
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from exc

    return app


app = create_app()
SERVICE = app.state.model_service

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
