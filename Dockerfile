FROM python:3.12-slim

ARG GIT_SHA=local
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=container \
    APP_VERSION=0.5.0 \
    GIT_SHA=${GIT_SHA}

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

RUN groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid 10001 appuser \
    && chown -R appuser:appuser /app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "25"]
