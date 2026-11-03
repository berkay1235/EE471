FROM python:3.12-slim-bookworm

# The Azure Speech SDK needs the ALSA and OpenSSL libraries on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first, so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the application code
COPY . .

EXPOSE 5002

# Flask must listen on 0.0.0.0, otherwise the port is unreachable from outside the container
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0", "--port=5002"]