import os
import platform
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

LIB_DIR = Path('geniex/geniex_sdk/lib')
ARCHIVE = 'geniex-bridge.zip'


def detect_platform():
    version = os.getenv('GENIEX_BRIDGE_VERSION', '').lstrip('v')
    if not version:
        with open('geniex/_version.py', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    version = line.split('=')[1].strip().strip('\'"')
                    break
    if not version or version == '0.0.0':
        raise RuntimeError(f'Invalid version: {version}. Please ensure GENIEX_BRIDGE_VERSION is set.')

    system = platform.system()
    arch = platform.machine().lower()

    os_name = system.lower()
    arch = 'arm64'

    return version, os_name, arch


def recreate_symlinks_from_zip(lib_dir):
    """
    Recreate symlinks after ZIP extraction during wheel build.

    ZIP files store symlinks as small text files containing the target path.
    Python's zipfile.extractall() preserves them as text files, not symlinks.
    This function converts them back to proper symlinks.

    Note: When setuptools builds the wheel, it follows these symlinks and copies
    the actual library files, resulting in hard copies in the final wheel.

    Raises:
        RuntimeError: If symlink creation fails (permission issues, unsupported filesystem, etc.)
    """
    if platform.system() == 'Windows':
        # Windows DLLs don't use symlinked versioning
        return

    symlinks_created = 0

    for root, dirs, files in os.walk(lib_dir):
        root_path = Path(root)
        for filename in files:
            filepath = root_path / filename

            # Skip if it's already a symlink
            if filepath.is_symlink():
                continue

            # Check if it's a small file that might be a converted symlink
            try:
                file_size = filepath.stat().st_size
                if file_size < 100 and filepath.is_file():
                    content = filepath.read_text().strip()

                    # Check if content looks like a library filename
                    if content and (content.endswith(('.so', '.dylib')) or '.so.' in content):
                        target_path = root_path / content

                        # Verify the target exists
                        if target_path.exists():
                            try:
                                # Remove the text file and create a symlink
                                filepath.unlink()
                                filepath.symlink_to(content)
                                symlinks_created += 1
                            except OSError as e:
                                raise RuntimeError(
                                    f'Failed to create symlink {filepath} -> {content}: {e}\n'
                                    f'This may be due to:\n'
                                    f'  - Insufficient permissions\n'
                                    f'  - Filesystem does not support symlinks\n'
                                    f'  - SELinux or security policies\n'
                                    f'Please ensure your filesystem supports symlinks or install to a different location.'
                                ) from e
            except UnicodeDecodeError:
                # Not a text file, skip it
                continue

    if symlinks_created > 0:
        print(
            f'✓ Recreated {symlinks_created} symlinks',
            file=sys.stderr,
            flush=True,
        )


def download_and_extract():
    version, os_name, arch = detect_platform()
    url = (
        f'https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/geniex/v{version}/{os_name}_{arch}/{ARCHIVE}'
    )

    if LIB_DIR.exists():
        return

    print(f'Downloading {url}...', file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / ARCHIVE
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(LIB_DIR)

    # Recreate symlinks that were converted to text files by ZIP
    recreate_symlinks_from_zip(LIB_DIR)

    print('✓ Binaries installed', file=sys.stderr, flush=True)


class BuildWithBinary(build_py):
    _downloaded = False

    def run(self):
        if not os.getenv('GENIEX_SKIP_DOWNLOAD'):
            if not BuildWithBinary._downloaded:
                try:
                    download_and_extract()
                    BuildWithBinary._downloaded = True
                except Exception as e:
                    print(
                        f'ERROR: Failed to download binaries: {e}',
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

        super().run()


setup(
    packages=find_packages(),
    cmdclass={
        'build_py': BuildWithBinary,
    },
    include_package_data=True,
)
