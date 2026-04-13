#!/usr/bin/env python3
#pyright: basic

import os
import shutil
import time
from typing import Optional

_ = os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER',
                          '1')  # must be set before importing huggingface_hub

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


def print(msg: str):
    __builtins__.print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}',
                       flush=True)


def check_update(repo_id: str, token: str, data_dir: Path) -> Optional[str]:
    api = HfApi(token=token)

    server_sha = api.model_info(repo_id).sha
    if server_sha is None:
        print(f'[warn] Failed to get sha for repo: {repo_id}')
        return None

    repo_name = repo_id.split('/')[-1]
    if not os.path.exists(data_dir / f'{repo_name}.sha'):
        return server_sha
    with open(data_dir / f'{repo_name}.sha', 'r') as f:
        local_sha = f.read().strip()

    return server_sha if server_sha != local_sha else None


def get_repo_in_collections(token: str) -> list[str]:
    api = HfApi(token=token)

    collections = api.list_collections(owner='NexaAI', limit=99)

    repo_names = []
    print(f'[info] Found collections')
    for col in collections:
        col = api.get_collection(collection_slug=col.slug)
        print(f'  -- {col.title} with {len(col.items)} items')
        for model in col.items:
            if model.item_type == 'model':
                repo_names.append(model.item_id.split('/')[-1])

    repo_names.extend(
        ['Llama3.2-3B-NPU', 'granite-4.0-micro-GGUF',
         'sdxl-turbo-amd-npu'])  # NOTE: add manually
    repo_names.sort()
    print(f'[info] Found {len(repo_names)} repos in total:')
    for name in repo_names:
        print(f'  - {name}')

    return repo_names


def iter_files(root: Path):
    root = root.resolve()
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        yield p, rel


def upload(
    endpoint: Optional[str],
    region: str,
    ak: str,
    sk: str,
    local_dir: Path,
    bucket: str,
    prefix: str,
    workers: int,
) -> None:
    client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region,
        config=Config(s3={'addressing_style': 'virtual'}),
    )

    # delete old files
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = response.get('Contents', [])
    if objects:
        delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects]}
        print(
            f'[info] Deleting {len(delete_keys["Objects"])} objects under {bucket}/{prefix} ...'
        )
        for o in delete_keys['Objects']:
            print(f'  - {o["Key"]}')
        client.delete_objects(Bucket=bucket, Delete=delete_keys)

    files = list(iter_files(local_dir))
    if not files:
        print(f'[warn] No files found under: {local_dir}')
        return

    def _put(fpair):
        local_path, rel_key = fpair
        key = f'{prefix.rstrip("/")}/{rel_key}'
        try:
            client.upload_file(local_path, bucket, key)
        except Exception as e:
            print(
                f'[error] Failed to upload {local_path} to s3://{bucket}/{key}: {e}'
            )
        return key

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_put, f) for f in files]
        for _ in tqdm(as_completed(futures),
                      total=len(futures),
                      desc='uploading...'):
            pass


def clean(
    endpoint: Optional[str],
    region: str,
    ak: str,
    sk: str,
    bucket: str,
    prefix: str,
    models: list[str],
) -> None:
    client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region,
        config=Config(s3={'addressing_style': 'virtual'}),
    )

    # delete old files
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get('Contents', [])
        if not objects:
            continue
        delete_keys = {'Objects': []}
        for obj in objects:
            key = obj['Key']
            if any(
                    key.startswith(f'{prefix.rstrip("/")}/{m}/')
                    for m in models):
                continue
            delete_keys['Objects'].append({'Key': key})
        if delete_keys['Objects']:
            print(
                f'[info] Deleting {len(delete_keys["Objects"])} objects under {bucket}/{prefix} ...'
            )
            for o in delete_keys['Objects']:
                print(f'  - {o["Key"]}')
            client.delete_objects(Bucket=bucket, Delete=delete_keys)


def main():
    ap = argparse.ArgumentParser(
        description=
        'Download Hugging Face repos into temp folders and upload them to Volcengine TOS.'
    )
    ap.add_argument(
        'repo_names',
        nargs='*',
        help='One or more repo names belong NexaAI, e.g. "paddleocr-npu"',
    )
    ap.add_argument('--data-dir', default=Path(__file__).parent)
    ap.add_argument('--workers', type=int, default=8)
    # HF repo args...
    ap.add_argument(
        '--hf-token',
        default=os.getenv('HF_TOKEN', 'REDACTED_HF_TOKEN'),
    )
    # S3 args...
    ap.add_argument('--s3-ak',
                    default=os.getenv('S3_ACCESS_KEY', 'REDACTED_AWS_ACCESS_KEY'))
    ap.add_argument(
        '--s3-sk',
        default=os.getenv('S3_SECRET_KEY',
                          'REDACTED_AWS_SECRET_KEY'),
    )
    # TOS args...
    ap.add_argument(
        '--tos-ak',
        default=os.getenv('TOS_ACCESS_KEY',
                          'REDACTED_VOLC_ACCESS_KEY'),
    )
    ap.add_argument(
        '--tos-sk',
        default=os.getenv(
            'TOS_SECRET_KEY',
            'T0RWak9EaGpNelV4WW1aaU5EQTJPRGxoT0dVNE0ySm1ZMk00WWpSak5XSQ==',
        ),
    )

    args = ap.parse_args()

    if not args.repo_names:
        print(
            f'[info] No repository names provided; retrieving all repositories in Hugging Face collections.'
        )
        args.repo_names = get_repo_in_collections(args.hf_token)
        if not args.repo_names:
            print(f'[info] No repo found in collections, exit.')
            return

    print(f'[info] Processing {len(args.repo_names)} repository(ies)')

    data_dir = Path(args.data_dir).resolve()

    for i, repo_name in enumerate(args.repo_names, 1):
        print(
            f'[info] [{i}/{len(args.repo_names)}] Processing repo: {repo_name}'
        )
        repo_name = Path(repo_name).name

        # Check if update is needed

        server_sha = check_update('NexaAI/' + repo_name,
                                  args.hf_token,
                                  data_dir=data_dir)
        if server_sha is None:
            print(f'[info] No update found, skipping: {repo_name}')
            continue

        # Download the repo

        local_dir = (Path(args.data_dir) / repo_name).resolve()
        print(f'[info] Downloading HF repo {repo_name} -> {local_dir}')
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(
            repo_id='NexaAI/' + repo_name,
            local_dir=local_dir,
            token=args.hf_token,
            max_workers=args.workers,
        )
        shutil.rmtree(local_dir / '.cache', ignore_errors=True)  # clean cache
        print(f'[info] Download completed: {snapshot_path}')

        # Upload to TOS and S3

        print(f'[info] Uploading to TOS')
        upload(
            'https://tos-s3-cn-beijing.volces.com',
            'cn-beijing',
            args.tos_ak,
            args.tos_sk,
            local_dir,
            'nexa-model-hub-bucket',
            f'model/{repo_name}/',
            args.workers,
        )

        print(f'[info] Uploading to S3')
        upload(
            None,
            'us-west-1',
            args.s3_ak,
            args.s3_sk,
            local_dir,
            'nexa-model-hub-bucket',
            f'public/nexa_sdk/huggingface-models/{repo_name}/',
            args.workers,
        )

        # Save the new sha

        with open(data_dir / f'{repo_name}.sha', 'w') as f:
            f.write(server_sha)
        shutil.rmtree(local_dir, ignore_errors=True)  # clean model
        print(f'[info] [{i}/{len(args.repo_names)}] Upload finished.')

    print(
        f'[info] All {len(args.repo_names)} repositories processed successfully!'
    )

    print(f'[info] Cleaning old models in TOS')
    clean(
        'https://tos-s3-cn-beijing.volces.com',
        'cn-beijing',
        args.tos_ak,
        args.tos_sk,
        'nexa-model-hub-bucket',
        'model/',
        args.repo_names,
    )
    print(f'[info] Cleaning old models in S3')
    clean(
        None,
        'us-west-1',
        args.s3_ak,
        args.s3_sk,
        'nexa-model-hub-bucket',
        'public/nexa_sdk/huggingface-models/',
        args.repo_names,
    )


def ticker(interval: int):
    next_time = time.time()
    while True:
        print(f'[info] Ticker wake up, start syncing...')

        try:
            main()
        except Exception as e:
            print(f'[error] Ticker failed: {e}')

        print(f'[info] Ticker done, sleeping...')
        next_time += interval
        while time.time() > next_time:
            print(f'[warn] {next_time} skipped')
            next_time += interval
        time.sleep(next_time - time.time())


if __name__ == '__main__':
    ticker(3600)  # every hour
