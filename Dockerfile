# Start from a small official Linux image that already has Python 3.12.
FROM python:3.12-slim

# Do not write .pyc files; print logs immediately; draw plots without a screen.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

WORKDIR /app

# 1. Install exact library versions first (this layer is cached between builds).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# 2. Copy the code and install covaudit itself without re-resolving libraries.
COPY . .
RUN pip install --no-cache-dir --no-deps .

# 3. Run as a normal user instead of root (security good practice).
RUN useradd --create-home appuser && mkdir -p /app/data /app/outputs \
    && chown -R appuser /app
USER appuser

CMD ["covaudit", "--help"]
