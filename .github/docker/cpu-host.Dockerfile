# syntax=docker/dockerfile:1.7

# CPU-only GenieX for non-Snapdragon arm64 hosts, e.g. Apple Silicon Macs
# under Docker Desktop. Wraps the published -cpu image and pre-sets the
# chipset at container start: without it `geniex pull` stops at an
# interactive picker that fails with no TTY. qualcomm-qcs6490 is used
# because chipsetOverrides (cli/internal/config/chipset.go) defaults
# --compute to cpu for it, so `geniex infer` needs no --compute flag.
# The config lives under /data, which is usually bind-mounted, so it is
# written at start rather than baked in, and only when no chipset is set.
#
#   docker build -f .github/docker/cpu-host.Dockerfile -t geniex-cpu-host .
#   docker run -it --rm -v "$PWD/data:/data" geniex-cpu-host
#   geniex pull unsloth/Qwen3-VL-2B-Instruct-GGUF:Q4_0 --model-type vlm
#   geniex infer unsloth/Qwen3-VL-2B-Instruct-GGUF:Q4_0 -p "Describe /data/img.jpg"
ARG GENIEX_TAG=latest-cpu
FROM docker.io/qualcomm/geniex:${GENIEX_TAG}

ENV GENIEX_CHIPSET=qualcomm-qcs6490

COPY --chmod=0755 <<'SH' /opt/geniex/entrypoint-cpu-host.sh
#!/bin/sh
set -e
if [ "$(geniex config get chipset --skip-update 2>/dev/null)" = "unknown" ]; then
    geniex config set chipset "$GENIEX_CHIPSET" --skip-update >/dev/null
fi
exec /opt/geniex/entrypoint.sh "$@"
SH

ENTRYPOINT ["/opt/geniex/entrypoint-cpu-host.sh"]
