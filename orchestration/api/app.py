from fastapi import FastAPI

from sentinel_contracts import SentinelRequest, SentinelResponse

from orchestration.api.guard_service import process_request


app = FastAPI(
    title="SENTINEL",
    description="Inline guard for AI-agent payment operations",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sentinel",
        "version": "0.1.0",
    }


@app.post("/v1/guard", response_model=SentinelResponse)
def guard(request: SentinelRequest) -> SentinelResponse:
    return process_request(request)
