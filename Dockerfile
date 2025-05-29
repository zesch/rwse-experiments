# app/Dockerfile

# The builder image, used to build the virtual environment
FROM python:3.11-bookworm AS builder

RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    BLIS_ARCH="generic"

WORKDIR /app

COPY pyproject.toml ./

RUN poetry install --no-cache --without dev

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11-slim-bookworm AS runtime

# Install curl for the healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    BLIS_ARCH="generic"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY experiments/demo.py /app/experiments/demo.py
COPY experiments/input /app/experiments/input

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "experiments/demo.py", "--server.port=8501", "--server.address=0.0.0.0"]