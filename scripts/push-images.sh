#!/usr/bin/env bash
# Push the locally-built turing images (turing-agent / turing-runner /
# turing-toolbox) to the local registry defined in docker-compose.yml (the
# `registry` service, registry:2 on host :5001). Named image tags + BuildKit
# pip caching (S9) make builds reproducible; this script makes them shareable
# so a fresh `docker compose pull` deploy reuses them instead of cold-rebuilding.
#
# Usage:
#   # 0. start the registry first (one-time; gated behind the `registry` profile):
#   docker compose --profile registry up -d registry
#   # 1. build (if needed) + push all three default images:
#   scripts/push-images.sh
#   # override the registry host:port (non-localhost hosts need insecure-registry
#   # config in the Docker daemon; localhost:5001 is permitted over HTTP as-is):
#   REGISTRY=my-host:5000 scripts/push-images.sh
#   # push a specific subset instead of all three:
#   scripts/push-images.sh turing-agent turing-runner
set -euo pipefail

REGISTRY="${REGISTRY:-localhost:5001}"

# Default image set; override by passing image base-names as args.
if [ "$#" -gt 0 ]; then
  IMAGES=("$@")
else
  IMAGES=(turing-agent turing-runner turing-toolbox)
fi

echo "→ Pushing to registry: ${REGISTRY}"
echo "→ Images: ${IMAGES[*]}"

failed=0
for img in "${IMAGES[@]}"; do
  src="${img}:latest"
  dst="${REGISTRY}/${img}:latest"
  if ! docker image inspect "${src}" >/dev/null 2>&1; then
    echo "  ✗ ${src} not found locally — build it first (e.g. docker compose build)" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "  • ${src} → ${dst}"
  docker tag "${src}" "${dst}"
  docker push "${dst}"
done

if [ "${failed}" -ne 0 ]; then
  echo "✗ ${failed} image(s) skipped (not built locally)." >&2
  exit 1
fi

echo "✓ Done. Pull with: docker pull ${REGISTRY}/<image>:latest"
