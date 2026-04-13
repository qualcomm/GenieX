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
GenieX-Bridge Embedding Example - Text Embedding Generation

This example demonstrates how to use the GenieX-Bridge Python binding to generate text embeddings.
It includes basic model initialization, single and batch embedding generation, and embedding analysis.
"""

import argparse
import logging
import os

from geniex import setup_logging
from geniex.embedding import Embedder


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge Embedding Example')
    parser.add_argument(
        '-m',
        '--model',
        default='Geniex/Qwen3-0.6B-GGUF',
        help='Path to the embedding model',
    )
    parser.add_argument(
        '--texts',
        nargs='+',
        default=[
            'On-device AI is a type of AI that is processed on the device itself, rather than in the cloud.',
            'GenieX-Bridge allows you to run state-of-the-art AI models locally on CPU, GPU, or NPU for Qualcomm-oriented deployments.',
            'A ragdoll is a breed of cat that is known for its long, flowing hair and gentle personality.',
            'The capital of France is Paris.',
        ],
        help='Texts to embed',
    )
    parser.add_argument(
        '--query',
        default='what is on device AI',
        help='Query text for similarity analysis',
    )
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    args = parser.parse_args()

    embedder = Embedder.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
    )
    print('Embedder loaded successfully!')

    dim = embedder.embedding_dim()
    print(f'Dimension: {dim}')

    batch_size = args.batch_size or len(args.texts)
    result = embedder.embed(
        texts=args.texts,
        batch_size=batch_size,
    )
    embeddings = result.embeddings

    print('\n' + '=' * 80)
    print('GENERATED EMBEDDINGS')
    print('=' * 80)

    for i, (text, embedding) in enumerate(zip(args.texts, embeddings)):
        print(f'\nText {i + 1}:')
        print(f'  Content: {text}')
        print(f'  Embedding shape: {len(embedding)} dimensions')
        print(f'  First 10 elements: {embedding[:10]}')
        print('-' * 70)

    print('\n' + '=' * 80)
    print('QUERY PROCESSING')
    print('=' * 80)
    print(f"Query: '{args.query}'")

    query_result = embedder.embed(
        texts=[args.query],
        batch_size=1,
    )
    query_embedding = query_result.embeddings[0]
    print(f'Query embedding shape: {len(query_embedding)} dimensions')

    print('\n' + '=' * 80)
    print('SIMILARITY ANALYSIS (Inner Product)')
    print('=' * 80)

    for i, (text, embedding) in enumerate(zip(args.texts, embeddings)):
        inner_product = sum(a * b for a, b in zip(query_embedding, embedding))
        print(f'\nText {i + 1}:')
        print(f'  Content: {text}')
        print(f'  Inner product with query: {inner_product:.6f}')
        print('-' * 70)


if __name__ == '__main__':
    main()
