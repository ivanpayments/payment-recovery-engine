#!/usr/bin/env bash
set -euo pipefail

DROPLET="${DROPLET:-root@209.38.71.25}"
REMOTE_PATH="${REMOTE_PATH:-/opt/project3-recovery}"
HEALTH_URL="${HEALTH_URL:-https://ivanantonov.com/recovery/health}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

echo "==> sync project to $DROPLET:$REMOTE_PATH"
ssh "$DROPLET" "mkdir -p $REMOTE_PATH"
tar --exclude='./__pycache__' \
    --exclude='./tests/__pycache__' \
    --exclude='./mlruns' \
    --exclude='./*.pyc' \
    -czf - . | ssh "$DROPLET" "cd $REMOTE_PATH && tar -xzf -"

echo "==> compose up"
ssh "$DROPLET" "cd $REMOTE_PATH && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build"

echo "==> caddy route reminder"
echo "First-time setup only: add the /recovery/* handle blocks (see README.md) to /etc/caddy/Caddyfile and run 'systemctl reload caddy'."

echo "==> health probe"
for i in {1..30}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "healthy: $HEALTH_URL"
    exit 0
  fi
  sleep 2
done

echo "health check failed: $HEALTH_URL"
exit 1
