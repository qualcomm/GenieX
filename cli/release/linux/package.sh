#!/bin/bash
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script creates a self-contained installer for a given platform.
# It takes a directory of build artifacts, creates a compressed payload,
# and embeds it into an installer script template.
#
# Usage:
#   ./package.sh <source_dir> <template_path> <output_name>

set -eu

# --- 1. Argument validation ---
SOURCE_DIR="$1"
TEMPLATE_PATH="$2"
OUTPUT_NAME="$3"

if [ -z "$SOURCE_DIR" ] || [ -z "$TEMPLATE_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Error: Missing arguments." >&2
    echo "Usage: $0 <source_dir> <template_path> <output_name>" >&2
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found." >&2
    exit 1
fi

if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "Error: Installer template '$TEMPLATE_PATH' not found." >&2
    exit 1
fi

# Create a temporary file for the payload
PAYLOAD_FILE=$(mktemp)
# Ensure the temporary payload is cleaned up on exit
trap 'rm -f "$PAYLOAD_FILE"' EXIT

echo "--- Creating Self-Contained Installer ---"
echo "Source:      $SOURCE_DIR"
echo "Template:    $TEMPLATE_PATH"
echo "Output:      $OUTPUT_NAME"

# --- 2. Create the compressed payload ---
echo "Step 1: Creating compressed payload from source directory..."
# -C changes directory to SOURCE_DIR before archiving, so the paths in the archive are relative
tar -czf "$PAYLOAD_FILE" -C "$SOURCE_DIR" .
if [ $? -ne 0 ]; then
    echo "Error: Failed to create tarball from '$SOURCE_DIR'." >&2
    exit 1
fi
echo "Payload created successfully."

# --- 3. Generate the final installer script ---
echo "Step 2: Generating final installer script..."
# Copy the template to the final output file
cp "$TEMPLATE_PATH" "$OUTPUT_NAME"
# Make the output script executable
chmod +x "$OUTPUT_NAME"
echo "Template copied and made executable."

# --- 4. Embed the payload ---
echo "Step 3: Encoding and embedding payload..."
# Append the base64 encoded payload to the end of the script
cat "$PAYLOAD_FILE" >> "$OUTPUT_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to embed payload into '$OUTPUT_NAME'." >&2
    exit 1
fi
echo "Payload embedded successfully."

# The temporary payload file will be removed by the trap
echo "----------------------------------------"
echo "Installer script '$OUTPUT_NAME' created successfully!"
echo "----------------------------------------"