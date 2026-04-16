# Usage: docker run --rm -it -v /path/to/data:/root/.cache/geniex geniex4ai/geniex-cli:latest

ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GenieX-CLI"
LABEL org.opencontainers.image.description="GenieX-CLI for running multi models"
LABEL org.opencontainers.image.vendor="GenieX"

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update \
    && apt-get install -y libgomp1 curl ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

COPY geniex/ /opt/geniex/
RUN ln -s /opt/geniex/geniex /usr/local/bin/geniex

RUN mkdir -p /data && mkdir -p /root/.cache && ln -s /data /root/.cache/geniex

EXPOSE 18181
ENTRYPOINT ["geniex"]
