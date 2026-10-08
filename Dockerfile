FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
ARG INSTALL_LOCAL_EMBEDDINGS=false
RUN if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
      pip install --no-cache-dir sentence-transformers; \
    fi

COPY app.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates

RUN useradd --create-home --uid 10001 api
USER api

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
