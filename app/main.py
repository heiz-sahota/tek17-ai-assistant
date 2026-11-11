"""
FastAPI service.

    uvicorn app.main:app --reload
    open http://127.0.0.1:8000/docs to try it
"""
import logging
import os

from fastapi import Depends, FastAPI, HTTPException

from app import rag
from app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="TEK17 AI assistant")


def current_user_roles() -> list[str]:
    """
    The signed-in user's roles, decided on the server, never taken from the request body.
    Locally they come from DEMO_ROLES in .env. In Azure they would come from the
    "roles" claim of the user's Entra ID token.

    Roles: admin (TEK17), sales (Lyngpakke pricing), both (everything).

    Returns: list[str], e.g. ["sales"].
    """
    return [r.strip() for r in os.environ.get("DEMO_ROLES", "both").split(",") if r.strip()]


@app.get("/health")
def health() -> dict:
    """
    Liveness check for Container Apps.

    Returns: dict, {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, roles: list[str] = Depends(current_user_roles)) -> ChatResponse:
    """
    Answer a question from the sources the user may see.

    Returns: ChatResponse.
      403 if the user's roles are not recognised.
      429 with Retry-After if the model quota is used up.
      503 if search or the model is unavailable; details are in the logs, not the response.
    """
    try:
        return rag.answer(request.question, roles)
    except ValueError as e:
        log.warning("Rejected roles: %s", e)
        raise HTTPException(status_code=403, detail="Not allowed")
    except rag.RateLimitedError as e:
        log.warning("Rate limited: %s", e)
        raise HTTPException(status_code=429, detail="Too many requests right now, try again shortly",
                            headers={"Retry-After": "20"})
    except rag.RetrievalError as e:
        log.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=503, detail="Search is unavailable, try again shortly")
    except rag.GenerationError as e:
        log.error("Generation failed: %s", e)
        raise HTTPException(status_code=503, detail="The model is unavailable, try again shortly")


def run() -> None:
    """
    Start the API with uvicorn. Settings come from the environment:
      APP_HOST    127.0.0.1 locally; 0.0.0.0 in a container so it can be reached from outside
      APP_PORT    8000
      APP_RELOAD  true while developing: restart when a .py file is saved. Never in production.

    Returns: None. Blocks until the server is stopped with Ctrl+C.
    """
    import uvicorn

    reload = os.environ.get("APP_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "app.main:app",  # an import string, not the app object: required for reload to work
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "8000")),
        reload=reload,
    )


if __name__ == "__main__":
    run()