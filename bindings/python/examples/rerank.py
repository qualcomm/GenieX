# Copyright 2024-2026 Qualcomm
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

#!/usr/bin/env python3

"""
GenieX-Bridge Rerank Example - Document Reranking

This example demonstrates how to use the GenieX-Bridge Python binding to rerank documents based on a query.
It includes basic model initialization, document reranking, and score analysis.
"""

import argparse
import logging
import os

from geniex import setup_logging
from geniex.rerank import Reranker


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge Rerank Example')
    parser.add_argument(
        '-m',
        '--model',
        default='~/.cache/geniex/geniex_sdk/models/Geniex/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf',
        help='Path to the rerank model',
    )
    parser.add_argument(
        '--query',
        default='Where is on-device AI?',
        help='Query text for reranking',
    )
    parser.add_argument(
        '--documents',
        nargs='+',
        default=[
            'On-device AI is a type of AI that is processed on the device itself, rather than in the cloud.',
            'edge computing',
            'A ragdoll is a breed of cat that is known for its long, flowing hair and gentle personality.',
            'The capital of France is Paris.',
        ],
        help='Documents to rerank',
    )
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    args = parser.parse_args()

    reranker = Reranker.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
    )

    batch_size = args.batch_size or len(args.documents)
    result = reranker.rerank(
        query=args.query,
        documents=args.documents,
        batch_size=batch_size,
    )
    scores = result.scores

    print(f'Query: {args.query}')
    print(f'Documents: {len(args.documents)} documents')
    print('-' * 50)
    for i, score in enumerate(scores):
        print(f'[{score:.4f}] : {args.documents[i]}')


if __name__ == '__main__':
    main()
