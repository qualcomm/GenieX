from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class ProgressInfo(TypedDict):
    total_downloaded: int  # Bytes downloaded
    total_size: int  # Total bytes to download
    percentage: float  # Progress percentage (0-100)
    files_active: int  # Number of files currently downloading
    files_total: int  # Total number of files
    known_total: bool  # Whether total size is known


class SpeedInfo(TypedDict):
    bytes_per_second: float  # Download speed in bytes/sec
    formatted: str  # Human readable speed (e.g., "1.2 MB/s")


class FormattingInfo(TypedDict):
    downloaded: str  # Human readable downloaded size
    total_size: str  # Human readable total size


class TimingInfo(TypedDict):
    elapsed_seconds: Optional[float]  # Time since download started
    eta_seconds: Optional[float]  # Estimated time remaining
    start_time: Optional[float]  # Download start timestamp


class DownloadProgressInfo(TypedDict):
    status: str  # 'idle', 'downloading', 'completed', 'error'
    error_message: Optional[str]  # Only present if status is 'error'
    progress: ProgressInfo
    speed: SpeedInfo
    formatting: FormattingInfo
    timing: TimingInfo


@dataclass
class ModelFileInfo:
    name: str
    downloaded: bool = False
    size: int = 0

    def to_dict(self) -> dict:
        return {
            'Name': self.name,
            'Downloaded': self.downloaded,
            'Size': self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelFileInfo':
        if not data:
            return cls(name='')
        return cls(
            name=data.get('Name') or '',
            downloaded=data.get('Downloaded') or False,
            size=data.get('Size') or 0,
        )


@dataclass
class ModelManifest:
    name: str  # OrgName/RepoName format
    model_name: str  # Model architecture name like "qwen3-4b", "yolov12", etc.
    model_type: str  # "llm", "vlm", "embedder", etc.
    plugin_id: str = ''
    device_id: str = ''
    min_sdk_version: str = ''
    model_file: Dict[str, ModelFileInfo] = field(default_factory=dict)  # quant -> modelfile
    mmproj_file: ModelFileInfo = field(default_factory=lambda: ModelFileInfo(''))
    tokenizer_file: ModelFileInfo = field(default_factory=lambda: ModelFileInfo(''))
    extra_files: List[ModelFileInfo] = field(default_factory=list)
    # Extra fields (not part of core manifest schema, but stored in manifest.json)
    pipeline_tag: Optional[str] = None  # Extra: Pipeline tag from HuggingFace model info
    download_time: Optional[str] = None  # Extra: ISO format timestamp of download
    avatar_url: Optional[str] = None  # Extra: Avatar URL for the model author

    def to_dict(self) -> dict:
        result = {
            'Name': self.name,
            'ModelName': self.model_name,
            'ModelType': self.model_type,
            'PluginId': self.plugin_id,
            'DeviceId': self.device_id,
            'MinSDKVersion': self.min_sdk_version,
            'ModelFile': {k: v.to_dict() for k, v in self.model_file.items()},
            'MMProjFile': self.mmproj_file.to_dict(),
            'TokenizerFile': self.tokenizer_file.to_dict(),
            'ExtraFiles': [f.to_dict() for f in self.extra_files],
            # Extra fields (flattened in manifest.json, using snake_case)
            'pipeline_tag': self.pipeline_tag,
            'download_time': self.download_time,
            'avatar_url': self.avatar_url,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelManifest':
        # Use `or` to handle None values (when key exists but value is null)
        model_file_data = data.get('ModelFile') or {}
        mmproj_file_data = data.get('MMProjFile') or {}
        tokenizer_file_data = data.get('TokenizerFile') or {}
        extra_files_data = data.get('ExtraFiles') or []

        return cls(
            name=data.get('Name') or '',
            model_name=data.get('ModelName') or '',
            model_type=data.get('ModelType') or '',
            plugin_id=data.get('PluginId') or '',
            device_id=data.get('DeviceId') or '',
            min_sdk_version=data.get('MinSDKVersion') or '',
            model_file={k: ModelFileInfo.from_dict(v) for k, v in model_file_data.items()},
            mmproj_file=ModelFileInfo.from_dict(mmproj_file_data),
            tokenizer_file=ModelFileInfo.from_dict(tokenizer_file_data),
            extra_files=[ModelFileInfo.from_dict(f) for f in extra_files_data],
            # Extra fields (flattened in manifest.json, using snake_case)
            pipeline_tag=data.get('pipeline_tag'),
            download_time=data.get('download_time'),
            avatar_url=data.get('avatar_url'),
        )

    def get_size(self) -> int:
        total = 0
        for file_info in self.model_file.values():
            if file_info.downloaded:
                total += file_info.size
        if self.mmproj_file.downloaded:
            total += self.mmproj_file.size
        if self.tokenizer_file.downloaded:
            total += self.tokenizer_file.size
        for file_info in self.extra_files:
            if file_info.downloaded:
                total += file_info.size
        return total


@dataclass
class MMProjInfo:
    """Data class for mmproj file information."""

    mmproj_path: Optional[str] = None
    size: int = 0


@dataclass
class ModelInfo:
    """Data class representing a model with all its metadata."""

    repo_id: str  # Model repository ID (e.g., "microsoft/phi-3-mini-4k-instruct")
    model_name: str  # Model architecture name
    model_type: str  # Model type (e.g., "llm", "vlm", "embedder")
    files: List[str]  # List of filenames (not paths)
    local_path: str  # Local path where the model is stored
    size_bytes: int  # Total size in bytes
    file_count: int  # Count of all files
    full_repo_download_complete: bool = True  # True if no incomplete downloads detected
    plugin_id: Optional[str] = None  # Plugin ID from nexa.manifest
    device_id: Optional[str] = None  # Device ID from nexa.manifest
    min_sdk_version: str = ''  # Minimum SDK version required
    pipeline_tag: Optional[str] = None  # Pipeline tag from HuggingFace model info (extra field)
    download_time: Optional[str] = None  # ISO format timestamp of download (extra field)
    avatar_url: Optional[str] = None  # Avatar URL for the model author (extra field)
    mmproj_info: Optional[MMProjInfo] = None  # mmproj file information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            'repo_id': self.repo_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'files': self.files,
            'local_path': self.local_path,
            'size_bytes': self.size_bytes,
            'file_count': self.file_count,
            'full_repo_download_complete': self.full_repo_download_complete,
            'plugin_id': self.plugin_id,
            'device_id': self.device_id,
            'min_sdk_version': self.min_sdk_version,
            'pipeline_tag': self.pipeline_tag,
            'download_time': self.download_time,
            'avatar_url': self.avatar_url,
            'mmproj_info': (
                {
                    'mmproj_path': self.mmproj_info.mmproj_path,
                    'size': self.mmproj_info.size,
                }
                if self.mmproj_info
                else None
            ),
        }
        return result

    @classmethod
    def from_manifest(
        cls,
        manifest: ModelManifest,
        local_path: str = '',
        **kwargs,
    ) -> 'ModelInfo':
        """
        Create ModelInfo from a ModelManifest.

        Args:
            manifest: The ModelManifest to convert
            local_path: Local path where the model is stored
            **kwargs: Extra fields to override manifest values:
                - pipeline_tag: Pipeline tag from HuggingFace
                - download_time: ISO format timestamp of download
                - avatar_url: Avatar URL for the model author
        """
        # Collect all filenames
        files: List[str] = []
        for file_info in manifest.model_file.values():
            if file_info.downloaded and file_info.name:
                files.append(file_info.name)
        if manifest.mmproj_file.downloaded and manifest.mmproj_file.name:
            files.append(manifest.mmproj_file.name)
        if manifest.tokenizer_file.downloaded and manifest.tokenizer_file.name:
            files.append(manifest.tokenizer_file.name)
        for file_info in manifest.extra_files:
            if file_info.downloaded and file_info.name:
                files.append(file_info.name)

        # Check if all files are fully downloaded
        full_repo_download_complete = True
        for file_info in manifest.model_file.values():
            if file_info.name and not file_info.downloaded:
                full_repo_download_complete = False
                break

        # Get mmproj info if available
        mmproj_info = None
        if manifest.mmproj_file.name and manifest.mmproj_file.downloaded:
            mmproj_info = MMProjInfo(
                mmproj_path=manifest.mmproj_file.name,
                size=manifest.mmproj_file.size,
            )

        # Use extra fields from kwargs, fallback to manifest values
        final_pipeline_tag = kwargs.get('pipeline_tag') or manifest.pipeline_tag
        final_download_time = kwargs.get('download_time') or manifest.download_time
        final_avatar_url = kwargs.get('avatar_url') or manifest.avatar_url

        return cls(
            repo_id=manifest.name,
            model_name=manifest.model_name,
            model_type=manifest.model_type,
            files=files,
            local_path=local_path,
            size_bytes=manifest.get_size(),
            file_count=len(files),
            full_repo_download_complete=full_repo_download_complete,
            plugin_id=manifest.plugin_id or None,
            device_id=manifest.device_id or None,
            min_sdk_version=manifest.min_sdk_version,
            pipeline_tag=final_pipeline_tag,
            download_time=final_download_time,
            avatar_url=final_avatar_url,
            mmproj_info=mmproj_info,
        )
