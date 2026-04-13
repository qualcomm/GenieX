"""
Progress tracking utilities for downloads with tqdm integration.

This module provides custom progress tracking classes that can monitor
download progress with callback support and customizable display options.
"""

import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from .types import DownloadProgressInfo

logger = logging.getLogger(__name__)


class DownloadProgressTracker:
    """Progress tracker for HuggingFace downloads with callback support."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[DownloadProgressInfo], None]] = None,
        show_progress: bool = True,
    ):
        self.progress_data: Dict[str, Dict[str, Any]] = {}
        self.total_repo_size = 0
        self.repo_file_count = 0
        self.original_tqdm_update = None
        self.original_tqdm_init = None
        self.original_tqdm_display = None
        self.original_tqdm_write = None
        self.is_tracking = False

        # Callback function
        self.progress_callback = progress_callback

        # Progress display
        self.show_progress = show_progress
        self.last_display_length = 0

        # Speed tracking
        self.last_downloaded = None
        self.last_time = None
        self.speed_history: List[float] = []
        self.max_speed_history = 10

        # Download status
        self.download_status = 'idle'
        self.error_message: Optional[str] = None
        self.download_start_time: Optional[float] = None

    def set_repo_info(self, total_size: int, file_count: int):
        """Set the total repository size and file count before download."""
        self.total_repo_size = total_size
        self.repo_file_count = file_count

    def register_tqdm(self, tqdm_instance):
        """Register a tqdm instance for monitoring."""
        tqdm_id = str(id(tqdm_instance))
        self.progress_data[tqdm_id] = {
            'current': 0,
            'total': getattr(tqdm_instance, 'total', 0) or 0,
            'desc': getattr(tqdm_instance, 'desc', 'Unknown'),
            'tqdm_obj': tqdm_instance,
        }
        self._trigger_callback()

    def update_progress(self, tqdm_instance, n=1):
        """Update progress for a tqdm instance."""
        tqdm_id = str(id(tqdm_instance))
        if tqdm_id in self.progress_data:
            self.progress_data[tqdm_id]['current'] = getattr(tqdm_instance, 'n', 0)
            self.progress_data[tqdm_id]['total'] = getattr(tqdm_instance, 'total', 0) or 0
            self._trigger_callback()

    def calculate_speed(self, current_downloaded: int) -> float:
        """Calculate download speed in bytes per second."""
        current_time = time.time()

        if self.last_time is not None and self.last_downloaded is not None:
            time_diff = current_time - self.last_time
            if time_diff > 0.1:
                bytes_diff = current_downloaded - self.last_downloaded
                if bytes_diff >= 0:
                    speed = bytes_diff / time_diff
                    self.speed_history.append(speed)
                    if len(self.speed_history) > self.max_speed_history:
                        self.speed_history.pop(0)
                    self.last_downloaded = current_downloaded
                    self.last_time = current_time
        else:
            self.last_downloaded = current_downloaded
            self.last_time = current_time

        if self.speed_history:
            return sum(self.speed_history) / len(self.speed_history)
        return 0.0

    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f'{bytes_value:.1f} {unit}'
            bytes_value /= 1024.0
        return f'{bytes_value:.1f} TB'

    def format_speed(self, speed: float) -> str:
        """Format speed to human readable string."""
        if speed == 0:
            return '0 B/s'
        for unit in ['B/s', 'KB/s', 'MB/s', 'GB/s']:
            if speed < 1024.0:
                return f'{speed:.1f} {unit}'
            speed /= 1024.0
        return f'{speed:.1f} TB/s'

    def get_progress_data(self) -> DownloadProgressInfo:
        """Get current progress data."""
        total_downloaded = 0
        active_file_count = 0
        total_file_sizes = 0

        for data in self.progress_data.values():
            if data['total'] > 0:
                total_downloaded += data['current']
                total_file_sizes += data['total']
                active_file_count += 1

        speed = self.calculate_speed(total_downloaded)

        if self.total_repo_size > 0:
            total_size = self.total_repo_size
        elif total_file_sizes > 0:
            total_size = total_file_sizes
        else:
            total_size = 0

        file_count = self.repo_file_count if self.repo_file_count > 0 else active_file_count

        if total_size > 0:
            percentage = min((total_downloaded / total_size * 100), 100.0)
        else:
            percentage = 0

        eta_seconds: Optional[float] = None
        if speed > 0 and total_size > total_downloaded:
            eta_seconds = (total_size - total_downloaded) / speed

        elapsed_seconds: Optional[float] = None
        if self.download_start_time:
            elapsed_seconds = time.time() - self.download_start_time

        return {
            'status': self.download_status,
            'error_message': self.error_message,
            'progress': {
                'total_downloaded': total_downloaded,
                'total_size': total_size,
                'percentage': round(percentage, 2),
                'files_active': active_file_count,
                'files_total': file_count,
                'known_total': total_size > 0,
            },
            'speed': {
                'bytes_per_second': speed,
                'formatted': self.format_speed(speed),
            },
            'formatting': {
                'downloaded': self.format_bytes(total_downloaded),
                'total_size': self.format_bytes(total_size),
            },
            'timing': {
                'elapsed_seconds': elapsed_seconds,
                'eta_seconds': eta_seconds,
                'start_time': self.download_start_time,
            },
        }

    def _display_progress_bar(self, progress_data: DownloadProgressInfo):
        """Display a custom unified progress bar."""
        if not self.show_progress:
            return

        if self.last_display_length > 0:
            print('\r' + ' ' * self.last_display_length, end='\r')

        progress_info = progress_data.get('progress', {})
        speed_info = progress_data.get('speed', {})
        formatting_info = progress_data.get('formatting', {})

        percentage = progress_info.get('percentage', 0)
        downloaded = formatting_info.get('downloaded', '0 B')
        total_size_raw = progress_info.get('total_size', 0)
        total_size = formatting_info.get('total_size', 'Unknown')
        speed = speed_info.get('formatted', '0 B/s')
        known_total = progress_info.get('known_total', False)

        bar_width = 30
        if known_total and total_size_raw > 0:
            filled_width = int(bar_width * min(percentage, 100) / 100)
            bar = '#' * filled_width + '-' * (bar_width - filled_width)
        else:
            animation_pos = int(time.time() * 2) % bar_width
            bar = '-' * animation_pos + '#' + '-' * (bar_width - animation_pos - 1)

        status = progress_data.get('status', 'unknown')
        if status == 'downloading':
            if known_total:
                progress_line = f'[{bar}] {percentage:.1f}% | {downloaded}/{total_size} | {speed}'
            else:
                progress_line = f'[{bar}] {downloaded} | {speed} | Calculating size...'
        elif status == 'completed':
            progress_line = f'[{bar}] 100.0% | {downloaded} | Complete!'
        elif status == 'error':
            progress_line = f'Error: {progress_data.get("error_message", "Unknown error")}'
        else:
            progress_line = 'Starting download...'

        print(progress_line, end='', flush=True)
        self.last_display_length = len(progress_line)

    def _clear_progress_bar(self):
        """Clear the progress bar display."""
        if self.show_progress and self.last_display_length > 0:
            print('\r' + ' ' * self.last_display_length, end='\r')
            print()
            self.last_display_length = 0

    def _trigger_callback(self):
        """Trigger the progress callback if one is set."""
        progress_data = self.get_progress_data()

        if self.progress_callback:
            try:
                self.progress_callback(progress_data)
            except Exception as e:
                logger.debug(f'Error in progress callback: {e}')

        if self.progress_callback and self.show_progress:
            self._display_progress_bar(progress_data)

    def start_tracking(self):
        """Start progress tracking (monkey patch tqdm)."""
        if self.is_tracking:
            return

        self.original_tqdm_update = tqdm.update
        self.original_tqdm_init = tqdm.__init__
        self.original_tqdm_display = getattr(tqdm, 'display', None)
        self.original_tqdm_write = getattr(tqdm, 'write', None)

        tracker = self

        def patched_init(self_tqdm, *args, **kwargs):
            kwargs['file'] = open(os.devnull, 'w')
            kwargs['disable'] = False
            kwargs['leave'] = False
            kwargs.pop('name', None)
            result = tracker.original_tqdm_init(self_tqdm, *args, **kwargs)
            tracker.register_tqdm(self_tqdm)
            return result

        def patched_update(self_tqdm, n=1):
            result = tracker.original_tqdm_update(self_tqdm, n)
            tracker.update_progress(self_tqdm, n)
            return result

        def patched_display(self_tqdm, msg=None, pos=None):
            pass

        def patched_write(self_tqdm, s, file=None, end='\n', nolock=False):
            pass

        tqdm.__init__ = patched_init
        tqdm.update = patched_update
        tqdm.display = patched_display
        tqdm.write = patched_write

        self.is_tracking = True
        self.download_status = 'downloading'
        self.download_start_time = time.time()
        self._trigger_callback()

    def stop_tracking(self):
        """Stop progress tracking and restore original tqdm."""
        if not self.is_tracking:
            return

        if self.original_tqdm_update:
            tqdm.update = self.original_tqdm_update
        if self.original_tqdm_init:
            tqdm.__init__ = self.original_tqdm_init
        if self.original_tqdm_display:
            tqdm.display = self.original_tqdm_display
        if self.original_tqdm_write:
            tqdm.write = self.original_tqdm_write

        for data in self.progress_data.values():
            if 'tqdm_obj' in data and hasattr(data['tqdm_obj'], 'fp'):
                try:
                    fp = data['tqdm_obj'].fp
                    if fp and fp != sys.stdout and fp != sys.stderr and not fp.closed:
                        fp.close()
                except Exception:
                    pass

        self.is_tracking = False
        if self.download_status == 'downloading':
            self.download_status = 'completed'

        self._trigger_callback()
        self._clear_progress_bar()

    def set_error(self, error_message: str):
        """Set error status and trigger callback."""
        self.download_status = 'error'
        self.error_message = error_message
        self._trigger_callback()
        self._clear_progress_bar()
