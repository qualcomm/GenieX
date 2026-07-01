# syntax=docker/dockerfile:1.7

# Derived toolchain image for arm64-android SDK builds in CI.
# Base image provides Android NDK r29 / Hexagon SDK / OpenCL. This layer
# bakes in build-essential (for Rust proc-macro host linking), ccache,
# and the aarch64-linux-android Rust target so _build-sdk.yml can skip
# the per-run apt-get / rustup installation.
ARG UPSTREAM_TAG=v0.7
FROM ghcr.io/snapdragon-toolchain/arm64-android:${UPSTREAM_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:$PATH

RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustup target add aarch64-linux-android \
    && chmod -R a+rwX /opt/rust
