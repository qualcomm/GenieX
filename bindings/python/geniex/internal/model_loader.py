import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

from .model_hub import ensure_model_downloaded, is_local_path
from .store import Store
from .types import ModelFileInfo, ModelManifest

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ModelLoaderMixin(ABC):
    @classmethod
    def from_(
        cls,
        model: str,
        quant: Optional[str] = None,
        **kwargs,
    ) -> T:
        # Check if it's a local path
        if is_local_path(model):
            return cls._from_local_path(model, **kwargs)

        # It's a Huggingface repo ID
        return cls._from_huggingface_repo(model, quant, **kwargs)

    @classmethod
    def _from_local_path(cls, model_path: str, **kwargs) -> T:
        # Extract common parameters
        tokenizer_path = kwargs.pop('tokenizer_path', None)
        if tokenizer_path is None:
            tokenizer_path = model_path

        # Ensure plugin_id is set
        plugin_id = kwargs.get('plugin_id')
        if not plugin_id:
            plugin_id = cls._get_default_plugin_id()
            kwargs['plugin_id'] = plugin_id

        # Create instance with local path
        return cls._create_instance(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            **kwargs,
        )

    @classmethod
    def _from_huggingface_repo(
        cls,
        repo_id: str,
        quant_spec: Optional[str] = None,
        **kwargs,
    ) -> T:
        store = Store.get()

        # Use unified internal function to ensure consistent behavior with download_model API
        manifest = ensure_model_downloaded(
            repo_id=repo_id,
            quant_spec=quant_spec,
            token=kwargs.get('token'),
            progress_callback=kwargs.get('progress_callback'),
            model_type=cls._get_model_type(),
        )

        # Select model file based on quant_spec
        model_file = cls._select_model_file(manifest, quant_spec, repo_id)

        # Get full path to model file
        model_path = store.modelfile_path(repo_id, model_file.name)

        # Verify model file exists, if not, try to find available local files
        if not model_path.exists():
            logger.debug(
                f'Requested model file not found: {model_path}. Searching for available local quantized files...'
            )
            # Try to find available local model files
            available_file = cls._find_available_local_file(manifest, repo_id, store)
            if available_file is None:
                raise FileNotFoundError(
                    f'Model file not found: {model_path}. Model may not have been downloaded completely.'
                )
            model_file = available_file
            model_path = store.modelfile_path(repo_id, model_file.name)
            logger.debug(f'Using available local quantized file: {model_file.name} (requested quant: {quant_spec})')

        params = cls._extract_model_params_from_manifest(
            manifest=manifest,
            model_path=str(model_path),
            repo_id=repo_id,
            store=store,
            **kwargs,
        )

        # Create instance
        return cls._create_instance(**params)

    @classmethod
    def _find_available_local_file(
        cls,
        manifest: ModelManifest,
        repo_id: str,
        store: Store,
    ) -> Optional['ModelFileInfo']:
        model_dir = store.model_dir_path() / repo_id
        if not model_dir.exists():
            return None

        # Scan for .gguf files in the model directory
        available_files = {}
        for file_path in model_dir.glob('*.gguf'):
            filename = file_path.name
            # Find matching entry in manifest
            for quant, file_info in manifest.model_file.items():
                if file_info.name == filename:
                    available_files[quant] = file_info
                    break

        if not available_files:
            return None

        # Prefer common quant names in order (matching Go implementation)
        preferred_quants = ['Q8_0', 'Q4_K_M', 'Q4_0', 'q8_0', 'q4_k_m', 'q4_0', 'default', 'N/A']
        selected_quant = None
        for pref in preferred_quants:
            # Try exact match first
            if pref in available_files:
                selected_quant = pref
                break
            # Try case-insensitive match
            for quant in available_files.keys():
                if quant.upper() == pref.upper():
                    selected_quant = quant
                    break
            if selected_quant:
                break

        if selected_quant is None:
            # Use the first available quant
            selected_quant = list(available_files.keys())[0]

        return available_files[selected_quant]

    @classmethod
    def _select_model_file(
        cls,
        manifest: ModelManifest,
        quant_spec: Optional[str],
        repo_id: str,
    ) -> 'ModelFileInfo':
        if quant_spec:
            # Normalize quant_spec to uppercase for matching (matching Go implementation)
            quant_spec_upper = quant_spec.strip().upper()
            selected_quant = None

            # First, try exact match on quant keys (case-insensitive)
            # This matches Go's logic: getQuant(ggufName) == specifiedQuant
            for quant, file_info in manifest.model_file.items():
                quant_upper = quant.upper()
                if quant_upper == quant_spec_upper:
                    selected_quant = quant
                    break

            # If no exact match, try matching by extracting quant from filename
            if selected_quant is None:
                from .model_hub import get_quant

                for quant, file_info in manifest.model_file.items():
                    # Check quant key or extract from filename
                    if quant.upper() == quant_spec_upper or get_quant(file_info.name).upper() == quant_spec_upper:
                        selected_quant = quant
                        break

            if selected_quant is None:
                # List available files for better error message
                available_quants = list(manifest.model_file.keys())
                available_files = [f'{quant}: {info.name}' for quant, info in manifest.model_file.items()]
                raise ValueError(
                    f'Quant "{quant_spec}" not found in model {repo_id}. '
                    f'Available quants: {available_quants}. '
                    f'Available files: {available_files[:5]}{"..." if len(available_files) > 5 else ""}'
                )
            else:
                return manifest.model_file[selected_quant]
        else:
            # Use the first available quant
            if not manifest.model_file:
                raise ValueError(f'No model files found in manifest for {repo_id}')

            # Prefer common quant names in order (matching Go implementation)
            # Go code prefers: Q8_0, Q4_K_M, Q4_0
            preferred_quants = ['Q8_0', 'Q4_K_M', 'Q4_0', 'q8_0', 'q4_k_m', 'q4_0', 'default', 'N/A']
            selected_quant = None
            for pref in preferred_quants:
                # Try exact match first
                if pref in manifest.model_file:
                    selected_quant = pref
                    break
                # Try case-insensitive match
                for quant in manifest.model_file.keys():
                    if quant.upper() == pref.upper():
                        selected_quant = quant
                        break
                if selected_quant:
                    break

            if selected_quant is None:
                selected_quant = list(manifest.model_file.keys())[0]

            selected_file = manifest.model_file[selected_quant]
            logger.debug(f'Selected quant: {selected_quant}, file: {selected_file.name}')
            return selected_file

    @classmethod
    @abstractmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest: ModelManifest,
        model_path: str,
        repo_id: str,
        store: Store,
        **kwargs,
    ) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def _create_instance(cls, **params) -> T:
        pass

    @classmethod
    def _get_default_plugin_id(cls) -> str:
        return 'cpu_gpu'

    @classmethod
    @abstractmethod
    def _get_model_type(cls) -> str:
        """Return the model type for this class (e.g., 'vlm', 'llm', 'embedder')."""
        pass
