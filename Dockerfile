# ============================================================================
# Stage 1: Build and dependency resolution
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

# Install system utilities required for basic environment setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly from the official binary distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Configure uv to isolate its Python installations
ENV UV_PYTHON_INSTALL_DIR=/app/.python
ENV UV_COMPILE_BYTECODE=1

# Copy tracking configuration files
COPY pyproject.toml uv.lock ./

# Force uv to install Python 3.12 within the defined directory and sync dependencies
RUN uv python install 3.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ============================================================================
# Stage 2: Final GPU-accelerated runtime environment
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Production environment variables
ENV PYTHONUTF8=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish path routing for uv managed Python and virtual environment
ENV PATH="/app/.venv/bin:/app/.python/python3.12/bin:$PATH"

# Copy the pre-compiled Python installation and virtual environment from builder
COPY --from=builder /app/.python /app/.python
COPY --from=builder /app/.venv /app/.venv

# Copy configuration files and core codebase
COPY configs/ /app/configs/
COPY src/ /app/src/

# Verify CUDA visibility during container build phase
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Define the immutable executable core interface
ENTRYPOINT ["python", "-m"]

# Define the default script to execute if no arguments are provided
CMD ["src.data.build"]
