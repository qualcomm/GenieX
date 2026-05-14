# syntax=docker/dockerfile:1.7

# Derived toolchain image for arm64-linux SDK builds in CI.
# Base image provides Hexagon SDK / OpenCL / crossbuild-essential-arm64.
# This layer bakes in build-essential, ccache, rustup, the aarch64 Rust
# target, and the cc-rs compatibility symlinks so _build-sdk.yml can skip
# the per-run apt-get / rustup installation.
ARG UPSTREAM_TAG=v0.1
FROM ghcr.io/snapdragon-toolchain/arm64-linux:${UPSTREAM_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# gcc-13 cross: base image's gcc-14 emits CXXABI_1.3.15 which Qualcomm
# Linux on-device libstdc++ 6.0.32 lacks. See #458.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc-13-aarch64-linux-gnu \
        g++-13-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:$PATH

RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustup target add aarch64-unknown-linux-gnu \
    && chmod -R a+rwX /opt/rust

# cc-rs derives the tool name as aarch64-unknown-linux-gnu-{gcc,g++,ar},
# but Debian ships them as aarch64-linux-gnu-*. Symlink so onig_sys (via
# tokenizers-cpp) cross-compiles without ToolNotFound.
RUN for tool in gcc g++ ar; do \
        ln -sf "/usr/bin/aarch64-linux-gnu-${tool}" \
               "/usr/local/bin/aarch64-unknown-linux-gnu-${tool}"; \
    done
