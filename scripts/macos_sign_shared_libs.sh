#!/bin/bash
# macOS code signing is not applicable to Qualcomm platforms (Windows ARM64 / Linux ARM64 / Android).
echo "macos_sign_shared_libs.sh: not applicable on this platform" >&2
exit 1#!/usr/bin/env bash
set -euo pipefail

##
## CONFIG — fill in the values below
##

# Your base64-encoded .p12 (the huge blob you pasted). Keep quotes intact.
CERT_BASE64='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'

# The password that protects the .p12:
CERT_PASSWORD='Nexaai2025!@#'

# Your Developer ID identity (exact string):
IDENTITY='Developer ID Application: Zhiyuan Li (B4Z6QVPW3V)'

# Folder that contains the libs/executables you want to sign:
TARGET_DIR="${1:-./plugins}"   # pass a directory as $1, defaults to ./libs

# Optional: an entitlements file (uncomment if you have one)
# ENTITLEMENTS=entitlements.plist

echo "Signing all code in: ${TARGET_DIR}"
test -d "${TARGET_DIR}" || { echo "Directory not found: ${TARGET_DIR}"; exit 2; }

##
## STEP A — Create a temporary keychain and import the p12
##

TMPDIR="$(mktemp -d)"
P12_PATH="${TMPDIR}/cert.p12"
KEYCHAIN_NAME="build.keychain"
KEYCHAIN_PWD=""   # empty password is fine for CI runners; use a real one if you prefer

# Write cert to disk
printf '%s' "${CERT_BASE64}" | base64 --decode > "${P12_PATH}"

# Clean up any stale keychain with the same name
security delete-keychain "${KEYCHAIN_NAME}" >/dev/null 2>&1 || true

# Create & unlock dedicated keychain
security create-keychain -p "${KEYCHAIN_PWD}" "${KEYCHAIN_NAME}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_NAME}"    # auto-unlock timeout 6h
security unlock-keychain -p "${KEYCHAIN_PWD}" "${KEYCHAIN_NAME}"

# Add keychain to search list so codesign can find the identity
security list-keychains -d user -s "${KEYCHAIN_NAME}" $(security list-keychains -d user | sed 's/"//g')

# Import certificate (.p12) and allow codesign to use it
security import "${P12_PATH}" \
  -k "${KEYCHAIN_NAME}" -P "${CERT_PASSWORD}" -T /usr/bin/codesign

# Ensure the private key is usable by codesign in non-interactive mode
security set-key-partition-list -S apple-tool:,apple:,codesign: \
  -s -k "${KEYCHAIN_PWD}" "${KEYCHAIN_NAME}"

# Sanity-check the identity is visible
echo "Available identities in ${KEYCHAIN_NAME}:"
security find-identity -v -p codesigning "${KEYCHAIN_NAME}" || true

##
## STEP B — (Recommended) remove quarantine flags, if any
## (Files fetched via browser often get com.apple.quarantine; git/LFS usually don’t)
##
if command -v xattr >/dev/null 2>&1; then
  xattr -dr com.apple.quarantine "${TARGET_DIR}" || true
fi

##
## STEP C — Sign everything that’s code: .dylib, .so, Mach-O binaries
## Sign with hardened runtime (--options runtime) and timestamp.
## If you have entitlements, add: --entitlements "$ENTITLEMENTS"
##

echo "Signing .dylib/.so and executables..."
# First: shared libs
while IFS= read -r -d '' f; do
  echo "  codesign: $f"
  codesign --force --timestamp --options runtime \
    --keychain "${KEYCHAIN_NAME}" \
    -s "${IDENTITY}" "$f"
done < <(find "${TARGET_DIR}" -type f \( -name '*.dylib' -o -name '*.so' \) -print0)

# Then: any Mach-O binaries without extension (CLI helpers, etc.)
# file(1) is a bit slow; skip if you know there are none.
while IFS= read -r -d '' f; do
  if file "$f" | grep -q 'Mach-O'; then
    echo "  codesign (exe): $f"
    codesign --force --timestamp --options runtime \
      --keychain "${KEYCHAIN_NAME}" \
      -s "${IDENTITY}" "$f"
  fi
done < <(find "${TARGET_DIR}" -type f ! -name '*.dylib' ! -name '*.so' -print0)

##
## STEP D — Verify signatures strictly
##

echo "Verifying signatures..."
FAILED=0
while IFS= read -r -d '' f; do
  if ! codesign --verify --verbose=2 --strict "$f"; then
    echo "  VERIFY FAILED: $f"
    FAILED=1
  fi
done < <(find "${TARGET_DIR}" -type f \( -name '*.dylib' -o -name '*.so' \) -print0)

# Verify executables too (optional)
while IFS= read -r -d '' f; do
  if file "$f" | grep -q 'Mach-O'; then
    if ! codesign --verify --verbose=2 --strict "$f"; then
      echo "  VERIFY FAILED (exe): $f"
      FAILED=1
    fi
  fi
done < <(find "${TARGET_DIR}" -type f ! -name '*.dylib' ! -name '*.so' -print0)

if [[ $FAILED -ne 0 ]]; then
  echo "One or more items failed verification."
  exit 1
fi

echo "All code signed and verified."

##
## STEP E — (Optional) Gatekeeper check
## This simulates what Gatekeeper would say about these files.
##
if command -v spctl >/dev/null 2>&1; then
  echo "Running spctl assessment on a sample item..."
  SAMPLE="$(find "${TARGET_DIR}" -type f -name '*.dylib' | head -n 1 || true)"
  if [[ -n "${SAMPLE}" ]]; then
    spctl --assess --type execute --verbose=4 "${SAMPLE}" || true
  fi
fi

##
## STEP F — Clean up the temporary keychain
##

echo "Cleaning up temporary keychain..."
security delete-keychain "${KEYCHAIN_NAME}" || true
rm -rf "${TMPDIR}"

echo "Done. All code signed, verified, and keychain cleaned up."
