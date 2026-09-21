# Start from a small official Linux image that already has Python 3.12.
FROM python:3.12-slim

# Do not write .pyc files; print logs immediately; draw plots without a screen.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# All following commands run inside /app in the container.
WORKDIR /app

# Copy the project into the image and install it with test tools.
COPY . .
RUN pip install --no-cache-dir ".[dev,notebook]"

# What runs if no command is given.
CMD ["covaudit", "--help"]
