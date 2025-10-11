"""File handling utilities for Arbitrium Framework.

This module provides centralized file handling utilities to ensure consistent
error handling, proper path management, and atomic operations where needed.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import aiofiles

from arbitrium.logging import get_contextual_logger

# Import exceptions for proper error typing
from .exceptions import FileSystemError

logger = get_contextual_logger("arbitrium.utils.file")


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure that the specified directory exists, creating it if necessary.

    Uses proper error handling and race condition avoidance.

    Args:
        directory_path: Path to the directory to ensure exists

    Returns:
        True if directory exists/was created, False on error
    """
    try:
        # Normalize the path
        directory = Path(directory_path).resolve()

        # If it already exists and is a directory, we're done
        if directory.exists() and directory.is_dir():
            return True

        # If it exists but isn't a directory, that's a problem
        if directory.exists() and not directory.is_dir():
            logger.error(f"Path exists but is not a directory: {directory}")
            return False

        # Create the directory and any necessary parent directories
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
        return True

    except PermissionError as e:
        logger.error(
            f"Permission error creating directory {directory_path}: {e!s}"
        )
        return False
    except OSError as e:
        logger.error(f"OS error creating directory {directory_path}: {e!s}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error creating directory {directory_path}: {e!s}"
        )
        return False


def generate_unique_filename(
    base_path: str, prefix: str, extension: str = ".md"
) -> str:
    """Generate a unique filename with high-precision timestamp.

    Args:
        base_path: Base directory path
        prefix: Filename prefix
        extension: File extension (including the dot)

    Returns:
        Absolute path to a unique filename
    """
    # Get high-precision timestamp (microseconds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Combine components
    filename = f"{prefix}_{timestamp}{extension}"

    # Join with base path
    return os.path.join(base_path, filename)


async def safe_write_async(
    file_path: str | Path,
    content: str,
    mode: str = "w",
    encoding: str = "utf-8",
    atomic: bool = True,
    backup: bool = False,
) -> bool:
    """Async version of safe_write. Write content to a file with robust error handling."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if backup and path.exists():
            backup_path = f"{path}.bak"
            shutil.copy2(path, backup_path)

        if atomic:
            fd, temp_path = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            os.close(fd)  # Close immediately to avoid Windows lock issues
            try:
                async with aiofiles.open(temp_path, mode=mode, encoding=encoding) as temp_file:  # type: ignore[call-overload]
                    await temp_file.write(content)
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: os.replace(temp_path, path)
                )
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
        else:
            async with aiofiles.open(path, mode=mode, encoding=encoding) as f:  # type: ignore[call-overload]
                await f.write(content)
        return True
    except (UnicodeEncodeError, OSError) as e:
        logger.error(f"Error writing to {file_path}: {e}")
        raise FileSystemError(
            f"Error writing to file: {e}", file_path=str(file_path)
        ) from e


async def safe_read_async(
    file_path: str | Path, encoding: str = "utf-8"
) -> tuple[bool, str]:
    """Async version of safe_read. Read a file with robust error handling.

    Args:
        file_path: Path to the file to read
        encoding: File encoding to use

    Returns:
        Tuple of (success, content)
        - success: True if read was successful, False otherwise
        - content: File content if successful, error message otherwise
    """
    try:
        # Convert to Path object for consistent handling
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            return False, error_msg

        # Check if file is readable
        if not os.access(path, os.R_OK):
            error_msg = f"File not readable: {path}"
            logger.error(error_msg)
            return False, error_msg

        # Read the file asynchronously
        async with aiofiles.open(path, "r", encoding=encoding) as f:
            content = await f.read()

        return True, content

    except UnicodeDecodeError as e:
        # Try with a more lenient encoding as a fallback
        try:
            async with aiofiles.open(
                path, "r", encoding="utf-8", errors="replace"
            ) as f:
                content = await f.read()
            logger.warning(
                f"Read {file_path} with encoding fallback due to: {e!s}"
            )
            return True, content
        except Exception as fallback_e:
            error_msg = f"Failed to read {file_path} even with encoding fallback: {fallback_e!s}"
            logger.error(error_msg)
            return False, error_msg
    except PermissionError as e:
        error_msg = f"Permission error reading {file_path}: {e!s}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to read {file_path}: {e!s}"
        logger.error(error_msg)
        return False, error_msg
