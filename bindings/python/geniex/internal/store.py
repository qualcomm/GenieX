import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from filelock import FileLock

from .types import ModelManifest

logger = logging.getLogger(__name__)


class Store:
    _instance: Optional['Store'] = None
    _initialized = False

    def __init__(self):
        if Store._initialized:
            return

        # Get user's cache directory (OS-specific)
        self.home = Path.home() / '.cache' / 'nexa.ai' / 'geniex_sdk'

        # Create models directory structure
        (self.home / 'models').mkdir(parents=True, exist_ok=True)

        self._model_locks: dict[str, FileLock] = {}
        Store._initialized = True

    @classmethod
    def get(cls) -> 'Store':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def data_path(self) -> Path:
        return self.home

    def model_dir_path(self) -> Path:
        return self.home / 'models'

    def modelfile_path(self, name: str, filename: str) -> Path:
        return self.model_dir_path() / name / filename

    def get_manifest(self, name: str) -> Optional[ModelManifest]:
        if not (self.model_dir_path() / name / 'nexa.manifest').exists():
            return None

        try:
            with open(self.model_dir_path() / name / 'nexa.manifest', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelManifest.from_dict(data)
        except Exception as e:
            logger.warning(f'Failed to read manifest for {name}: {e}')
            return None

    def save_manifest(self, manifest: ModelManifest) -> None:
        (self.model_dir_path() / manifest.name).mkdir(parents=True, exist_ok=True)

        with open(self.model_dir_path() / manifest.name / 'nexa.manifest', 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def lock_model(self, model_name: str) -> bool:
        if not model_name:
            return False

        if model_name in self._model_locks:
            return True  # Already locked by this process

        (self.model_dir_path() / model_name).mkdir(parents=True, exist_ok=True)
        lock_path = self.model_dir_path() / f'{model_name}.lock'

        try:
            lock = FileLock(str(lock_path), timeout=0)
            lock.acquire()
            self._model_locks[model_name] = lock
            logger.debug(f'Locked model: {model_name}')
            return True
        except Exception as e:
            logger.warning(f'Failed to lock model {model_name}: {e}')
            return False

    def unlock_model(self, model_name: str) -> None:
        if not model_name:
            return

        if model_name in self._model_locks:
            try:
                lock = self._model_locks[model_name]
                lock.release()
                del self._model_locks[model_name]
                logger.debug(f'Unlocked model: {model_name}')
            except Exception as e:
                logger.warning(f'Failed to unlock model {model_name}: {e}')

    def model_exists(self, name: str) -> bool:
        manifest = self.get_manifest(name)
        return manifest is not None

    def list_models(self) -> List[ModelManifest]:
        models = []

        if not self.model_dir_path().exists():
            return models

        # Scan two-level directory structure: org/repo
        # First level: organization directories
        for org_dir in self.model_dir_path().iterdir():
            if not org_dir.is_dir():
                continue
            if org_dir.name == '.cache':
                continue
            if org_dir.name.endswith('.lock'):
                continue
            # Second level: repository directories
            try:
                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    # Combine org and repo to form model name (e.g., "org/repo")
                    model_name = f'{org_dir.name}/{repo_dir.name}'

                    # Try to read manifest
                    manifest = self.get_manifest(model_name)
                    if manifest is not None:
                        models.append(manifest)
            except Exception as e:
                logger.warning(f'Failed to read model subdirectory {org_dir.name}: {e}')
                continue

        return models

    def remove_model(self, name: str) -> bool:
        if not name:
            return False

        # Unlock the model first if it's locked
        if name in self._model_locks:
            self.unlock_model(name)

        model_dir = self.model_dir_path() / name
        lock_file = self.model_dir_path() / f'{name}.lock'

        # Remove model directory
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                logger.info(f'Removed model directory: {model_dir}')
            except Exception as e:
                logger.error(f'Failed to remove model directory {model_dir}: {e}')
                return False

        # Remove lock file if it exists
        if lock_file.exists():
            try:
                lock_file.unlink()
                logger.debug(f'Removed lock file: {lock_file}')
            except Exception as e:
                logger.warning(f'Failed to remove lock file {lock_file}: {e}')

        return True
