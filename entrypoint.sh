#!/bin/sh
# Decode GCP Service Account key from HF Secret, then start uvicorn.
# Set GCP_SA_KEY (base64-encoded JSON) in Space → Settings → Secrets.
if [ -n "$GCP_SA_KEY" ]; then
    echo "$GCP_SA_KEY" | base64 -d > /tmp/gcp-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
fi

exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-7860}"
