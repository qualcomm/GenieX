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

"""
GenieX-Bridge ImageGen Example - Text to Image / Image to Image

This example demonstrates how to use the GenieX-Bridge Python binding to generate images from text prompts
or modify existing images with text prompts.
"""

import argparse
import logging
import os
import time

from geniex import setup_logging
from geniex.image_gen import ImageGen


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge ImageGen Example')
    parser.add_argument(
        '-m',
        '--model',
        default='Geniex/sdxl-turbo',
        help='Model id or path',
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='Text prompt for image generation',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output image file path (default: auto-generated)',
    )
    parser.add_argument(
        '--init-image',
        default=None,
        help='Initial image path for img2img (optional)',
    )
    parser.add_argument(
        '--negative-prompt',
        default=None,
        help='Negative prompt (optional)',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height in pixels (default: 512)',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width in pixels (default: 512)',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of diffusion steps (default: 50)',
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale for classifier-free guidance (default: 7.5)',
    )
    parser.add_argument(
        '--strength',
        type=float,
        default=0.8,
        help='Strength parameter for img2img (0.0-1.0, default: 0.8)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='Random seed (-1 for random, default: -1)',
    )
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    parser.add_argument('--device', default=None, help='Device to run on (e.g., cpu, gpu, 0)')
    args = parser.parse_args()

    # Generate output filename if not specified
    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        output_path = f'imagegen_output_{int(time.time())}.png'

    # Create ImageGen instance
    image_gen = ImageGen.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
        device_id=args.device,
    )

    # Prepare negative prompts
    negative_prompts = None
    if args.negative_prompt:
        negative_prompts = [args.negative_prompt]

    # Generate image
    if args.init_image:
        # Image-to-image generation
        init_image_path = os.path.expanduser(args.init_image)
        if not os.path.exists(init_image_path):
            raise FileNotFoundError(f'Initial image file not found: {init_image_path}')

        print(f'Generating image from: {init_image_path}')
        print(f'Prompt: {args.prompt}')
        result = image_gen.img2img(
            init_image_path=init_image_path,
            prompt=args.prompt,
            output_path=output_path,
            negative_prompts=negative_prompts,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            strength=args.strength,
        )
    else:
        # Text-to-image generation
        print(f'Generating image from text prompt: {args.prompt}')
        result = image_gen.txt2img(
            prompt=args.prompt,
            output_path=output_path,
            negative_prompts=negative_prompts,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

    print(f'Image saved to: {result.output_image_path}')


if __name__ == '__main__':
    main()
