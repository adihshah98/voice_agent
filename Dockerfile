FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy everything first (hatchling needs the voice_agent/ package to build)
COPY . .

# Install production deps
RUN uv sync --frozen --no-dev

# Expose FastAPI port
EXPOSE 8000

# Run Alembic migrations then start the server.
# DATABASE_URL is injected by Render at runtime.
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn voice_agent.server:app --host 0.0.0.0 --port 8000"]
