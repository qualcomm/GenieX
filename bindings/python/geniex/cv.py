"""
CV (Computer Vision) wrapper for geniex.
Provides high-level Python API for computer vision operations.
"""

import ctypes
import logging
from typing import List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.cv import (
    ml_CV,
    ml_CVCapabilities,
    ml_CVCreateInput,
    ml_CVInferInput,
    ml_CVInferOutput,
    ml_CVModelConfig,
)
from .geniex_sdk.error import check_error

logger = logging.getLogger(__name__)


class CV(ModelLoaderMixin):
    """Computer Vision model wrapper."""

    def __init__(
        self,
        model_name: str,
        capabilities: int = 0,
        det_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        char_dict_path: Optional[str] = None,
        qnn_model_folder_path: Optional[str] = None,
        qnn_lib_folder_path: Optional[str] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
        license_id: Optional[str] = None,
        license_key: Optional[str] = None,
    ):
        """
        Create and initialize a CV instance.

        Args:
            model_name: Name of the model.
            capabilities: CV capabilities (0=OCR, 1=Classification, 2=Segmentation, 3=Custom).
            det_model_path: Path to detection model (for OCR).
            rec_model_path: Path to recognition model (for OCR).
            char_dict_path: Path to character dictionary (for OCR).
            qnn_model_folder_path: Path to QNN model folder.
            qnn_lib_folder_path: Path to QNN library folder.
            plugin_id: Plugin to use for the model. If None, uses default.
            device_id: Device to use for the model. If None, uses default device.
            license_id: License ID for loading NPU models.
            license_key: License key for loading NPU models.
        """
        model_name_bytes = model_name.encode('utf-8')
        plugin_id_str = plugin_id if plugin_id else ''
        device_id_str = device_id if device_id else ''
        license_id_str = license_id if license_id else ''
        license_key_str = license_key if license_key else ''

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id_str.encode('utf-8'))
        license_id_buf = ctypes.create_string_buffer(license_id_str.encode('utf-8'))
        license_key_buf = ctypes.create_string_buffer(license_key_str.encode('utf-8'))

        det_model_path_buf = ctypes.create_string_buffer(det_model_path.encode('utf-8')) if det_model_path else None
        rec_model_path_buf = ctypes.create_string_buffer(rec_model_path.encode('utf-8')) if rec_model_path else None
        char_dict_path_buf = ctypes.create_string_buffer(char_dict_path.encode('utf-8')) if char_dict_path else None
        qnn_model_folder_path_buf = (
            ctypes.create_string_buffer(qnn_model_folder_path.encode('utf-8')) if qnn_model_folder_path else None
        )
        qnn_lib_folder_path_buf = (
            ctypes.create_string_buffer(qnn_lib_folder_path.encode('utf-8')) if qnn_lib_folder_path else None
        )

        # Create cv_model_config with pointers to the string buffers
        cv_model_config = ml_CVModelConfig(
            capabilities=ml_CVCapabilities(capabilities),
            det_model_path=ctypes.cast(det_model_path_buf, ctypes.c_char_p) if det_model_path_buf else None,
            rec_model_path=ctypes.cast(rec_model_path_buf, ctypes.c_char_p) if rec_model_path_buf else None,
            char_dict_path=ctypes.cast(char_dict_path_buf, ctypes.c_char_p) if char_dict_path_buf else None,
            qnn_model_folder_path=(
                ctypes.cast(qnn_model_folder_path_buf, ctypes.c_char_p) if qnn_model_folder_path_buf else None
            ),
            qnn_lib_folder_path=(
                ctypes.cast(qnn_lib_folder_path_buf, ctypes.c_char_p) if qnn_lib_folder_path_buf else None
            ),
        )

        c_input = ml_CVCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            config=cv_model_config,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
            license_id=ctypes.cast(license_id_buf, ctypes.c_char_p) if license_id_buf else None,
            license_key=ctypes.cast(license_key_buf, ctypes.c_char_p) if license_key_buf else None,
        )

        self._string_refs = [
            model_name_buf,
            plugin_id_buf,
            device_id_buf,
            license_id_buf,
            license_key_buf,
            det_model_path_buf,
            rec_model_path_buf,
            char_dict_path_buf,
            qnn_model_folder_path_buf,
            qnn_lib_folder_path_buf,
        ]

        handle_ptr = ctypes.POINTER(ml_CV)()
        error_code = _lib.ml_cv_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create CV instance')

        self._handle = handle_ptr

    def __del__(self):
        """Destroy CV instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_cv_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during CV cleanup: {e}')
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'cv'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """Extract CV-specific parameters from manifest."""
        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

        capabilities = kwargs.pop('capabilities', 0)
        det_model_path = kwargs.pop('det_model_path', None)
        rec_model_path = kwargs.pop('rec_model_path', None)
        char_dict_path = kwargs.pop('char_dict_path', None)

        # If det_model_path and rec_model_path are not provided, use model_path (matching Go behavior)
        if not det_model_path:
            det_model_path = model_path
        if not rec_model_path:
            rec_model_path = model_path

        plugin_id = manifest.plugin_id.strip() if manifest.plugin_id else None
        if not plugin_id:
            plugin_id = kwargs.pop('plugin_id', None)
        else:
            # Remove plugin_id from kwargs if manifest has it (to avoid passing it twice)
            kwargs.pop('plugin_id', None)
        if plugin_id:
            plugin_id = plugin_id.strip() if isinstance(plugin_id, str) else plugin_id
            if not plugin_id:
                plugin_id = None

        if not plugin_id:
            plugin_id = cls._get_default_plugin_id()

        device_id = manifest.device_id.strip() if manifest.device_id else None
        if not device_id:
            device_id = kwargs.pop('device_id', None)
        if device_id:
            device_id = device_id.strip() if isinstance(device_id, str) else device_id
            if not device_id:
                device_id = None

        return {
            'model_name': model_name,
            'capabilities': capabilities,
            'det_model_path': det_model_path,
            'rec_model_path': rec_model_path,
            'char_dict_path': char_dict_path,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'CV':
        """Create CV instance with given parameters."""
        return cls(**params)

    @classmethod
    def _get_default_plugin_id(cls) -> str:
        """Get default plugin ID for CV."""
        return 'cpu_gpu'

    def infer(self, input_image_path: str) -> 'CVResult':
        """
        Perform CV inference on image.

        Args:
            input_image_path: Path to the input image file.

        Returns:
            CVResult containing detection/classification results.

        Raises:
            Geniex: If inference fails.
        """
        # C interface implementation
        input_image_path_bytes = input_image_path.encode('utf-8')
        input_image_path_buf = ctypes.create_string_buffer(input_image_path_bytes)

        c_input = ml_CVInferInput(
            input_image_path=ctypes.cast(input_image_path_buf, ctypes.c_char_p),
        )

        c_output = ml_CVInferOutput()
        error_code = _lib.ml_cv_infer(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def _extract_result(self, c_output: ml_CVInferOutput) -> 'CVResult':
        """Extract CVResult from C output structure."""
        if not (c_output.results and c_output.result_count > 0):
            return CVResult(results=[])

        results = []
        for i in range(c_output.result_count):
            result = c_output.results[i]
            image_paths = []
            if result.image_paths and result.image_count > 0:
                image_paths_array = ctypes.cast(
                    result.image_paths, ctypes.POINTER(ctypes.c_char_p * result.image_count)
                ).contents
                for j in range(result.image_count):
                    img_path_ptr = image_paths_array[j]
                    if img_path_ptr:
                        image_paths.append(ctypes.string_at(img_path_ptr).decode('utf-8'))

            text = ''
            if result.text:
                text = ctypes.string_at(result.text).decode('utf-8')

            embedding = []
            if result.embedding and result.embedding_dim > 0:
                embedding_array = ctypes.cast(
                    result.embedding, ctypes.POINTER(ctypes.c_float * result.embedding_dim)
                ).contents
                embedding = [float(embedding_array[j]) for j in range(result.embedding_dim)]

            bbox = BoundingBox(
                x=float(result.bbox.x),
                y=float(result.bbox.y),
                width=float(result.bbox.width),
                height=float(result.bbox.height),
            )

            results.append(
                CVResultItem(
                    image_paths=image_paths,
                    class_id=int(result.class_id),
                    confidence=float(result.confidence),
                    bbox=bbox,
                    text=text,
                    embedding=embedding,
                    embedding_dim=int(result.embedding_dim),
                )
            )

        return CVResult(results=results)


class BoundingBox:
    """Bounding box structure."""

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f'BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class CVResultItem:
    """Single CV result item."""

    def __init__(
        self,
        image_paths: List[str],
        class_id: int,
        confidence: float,
        bbox: Optional[BoundingBox],
        text: str,
        embedding: List[float],
        embedding_dim: int,
    ):
        self.image_paths = image_paths
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox
        self.text = text
        self.embedding = embedding
        self.embedding_dim = embedding_dim

    def __repr__(self) -> str:
        text_preview = self.text[:30] if self.text else None
        return f'CVResultItem(class_id={self.class_id}, confidence={self.confidence:.2f}, text={text_preview})'


class CVResult:
    """Result of CV inference."""

    def __init__(self, results: List[CVResultItem]):
        self.results = results

    def __repr__(self) -> str:
        return f'CVResult(results={len(self.results)})'
