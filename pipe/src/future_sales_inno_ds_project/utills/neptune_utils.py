# pipe/src/utils/neptune_utils.py
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def safe_upload_image(run, key: str, path: str):
    try:
        file_path = Path(path)
        if file_path.exists():
            run[key].upload(str(file_path.resolve()))
        else:
            logger.warning(f"Image not found: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to upload image to Neptune: {e}")


def safe_track_file(run, key: str, path: str):
    try:
        file_path = Path(path)
        if file_path.exists():
            relative_path = file_path.relative_to(Path.cwd())
            run[key].track_files(str(relative_path))
        else:
            logger.warning(f"File not found: {path}")
    except Exception as e:
        logger.warning(f"Failed to track {path} in Neptune: {e}")
