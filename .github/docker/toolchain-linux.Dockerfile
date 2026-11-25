# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=docker.io/qualcomm/geniex-toolchain-linux:v0.2.0@sha256:ba89cde82441129dfae2f1dc6e379f10db1dc26a861e455eebc01f1052223937
FROM ${BASE_IMAGE}

ARG QEMU_GUEST_BASE

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eu; \
    baseline="$(mktemp -d)"; \
    dpkg-query -W -f='${Package}=${Version}\n' \
      libc6-arm64-cross libc6-dev-arm64-cross libstdc++6-arm64-cross \
      > "$baseline/packages.before"; \
    case "$(dpkg-query -W -f='${Version}' libc6-arm64-cross)" in \
      2.41-*) ;; \
      *) echo 'Expected the published GLIBC 2.41 ARM64 sysroot' >&2; exit 1 ;; \
    esac; \
    sha256sum /usr/aarch64-linux-gnu/include/features.h \
      /usr/aarch64-linux-gnu/lib/libc.so.6 \
      /usr/aarch64-linux-gnu/lib/libstdc++.so.6 > "$baseline/sysroot.sha256"; \
    printf '%s\n' \
      'deb [arch=amd64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian bookworm main' \
      > /etc/apt/sources.list.d/geniex-gcc11.list; \
    printf '%s\n' 'Package: *' 'Pin: release n=bookworm' 'Pin-Priority: 100' \
      > /etc/apt/preferences.d/geniex-gcc11; \
    apt-get update; \
    apt-get install -y --no-install-recommends --no-remove \
      gcc-11-aarch64-linux-gnu g++-11-aarch64-linux-gnu; \
    dpkg-query -W -f='${Package}=${Version}\n' \
      libc6-arm64-cross libc6-dev-arm64-cross libstdc++6-arm64-cross \
      > "$baseline/packages.after"; \
    cmp "$baseline/packages.before" "$baseline/packages.after"; \
    sha256sum -c "$baseline/sysroot.sha256"; \
    rm -f /etc/apt/sources.list.d/geniex-gcc11.list \
      /etc/apt/preferences.d/geniex-gcc11; \
    rm -rf "$baseline" /var/lib/apt/lists/*

RUN set -eu; \
    for tool in gcc g++ ar; do \
      case "$tool" in \
        gcc|g++) binary="/usr/bin/aarch64-linux-gnu-${tool}-11" ;; \
        ar) binary=/usr/bin/aarch64-linux-gnu-gcc-ar-11 ;; \
      esac; \
      ln -sf "$binary" "/usr/local/bin/aarch64-unknown-linux-gnu-${tool}"; \
    done

RUN set -eu; \
    test "$(aarch64-linux-gnu-gcc-11 -dumpversion)" = 11; \
    test "$(aarch64-linux-gnu-g++-11 -dumpversion)" = 11; \
    smoke="$(mktemp -d)"; \
    printf '%s\n' '#include <features.h>' '#include <stdio.h>' \
      '#if __GLIBC__ != 2 || __GLIBC_MINOR__ != 41' \
      '#error Expected GLIBC 2.41 target headers' '#endif' \
      'int main(void) { return puts("GCC 11 / GLIBC 2.41") < 0; }' \
      > "$smoke/smoke.c"; \
    printf '%s\n' '#include <iostream>' '#include <vector>' '#include <omp.h>' \
      'int main() { std::vector<int> v{11, 41};' \
      'std::cout << v.at(0) << "/" << v.at(1) << " " << omp_get_max_threads() << "\n"; }' \
      > "$smoke/smoke.cpp"; \
    aarch64-unknown-linux-gnu-gcc "$smoke/smoke.c" -o "$smoke/smoke-c"; \
    aarch64-unknown-linux-gnu-g++ -std=c++17 -fopenmp \
      "$smoke/smoke.cpp" -o "$smoke/smoke-cxx"; \
    aarch64-linux-gnu-readelf -h "$smoke/smoke-c"; \
    aarch64-linux-gnu-readelf -h "$smoke/smoke-cxx"; \
    rm -rf "$smoke"
