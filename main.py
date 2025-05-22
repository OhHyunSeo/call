# Standard library imports
import os
import asyncio
import argparse
import logging
from typing import List

# Related third-party imports
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# Local imports
from src.audio.utils import Formatter
from src.audio.metrics import SilenceStats
from src.audio.error import DialogueDetecting
from src.audio.alignment import ForcedAligner
from src.audio.effect import DemucsVocalSeparator
from src.audio.preprocessing import SpeechEnhancement
from src.audio.io import SpeakerTimestampReader, TranscriptWriter
from src.audio.analysis import WordSpeakerMapper, SentenceSpeakerMapper, Audio
from src.audio.processing import AudioProcessor, Transcriber, PunctuationRestorer
from src.text.utils import Annotator
from src.text.llm import LLMOrchestrator, LLMResultHandler
from src.utils.utils import Cleaner, Watcher
from src.db.manager import Database
from src.pipeline import AsyncPipeline, MultiCallProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")


async def process_async(audio_file_path: str):
    """
    Process an audio file using the asynchronous pipeline.
    
    Parameters
    ----------
    audio_file_path : str
        Path to the audio file to process
    """
    try:
        logger.info(f"Processing file: {audio_file_path}")
        pipeline = AsyncPipeline()
        
        # Process the audio file
        result = await pipeline.process(audio_file_path)
        
        # Clean up resources
        await pipeline.cleanup()
        
        logger.info(f"Completed processing: {audio_file_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {audio_file_path}: {e}")
        return None


async def process_batch_async(audio_file_paths: List[str], max_concurrent: int = 3):
    """
    Process multiple audio files concurrently.
    
    Parameters
    ----------
    audio_file_paths : List[str]
        List of paths to audio files
    max_concurrent : int
        Maximum number of files to process concurrently
    """
    processor = MultiCallProcessor(max_concurrent=max_concurrent)
    results = await processor.process_batch(audio_file_paths)
    logger.info(f"Completed batch processing of {len(audio_file_paths)} files")
    return results


def process(audio_file_path: str):
    """
    Synchronous wrapper for processing a single audio file.
    
    Parameters
    ----------
    audio_file_path : str
        Path to the audio file to process
    """
    return asyncio.run(process_async(audio_file_path))


def process_batch(audio_file_paths: List[str], max_concurrent: int = 3):
    """
    Synchronous wrapper for processing multiple audio files.
    
    Parameters
    ----------
    audio_file_paths : List[str]
        List of paths to audio files
    max_concurrent : int
        Maximum number of files to process concurrently
    """
    return asyncio.run(process_batch_async(audio_file_paths, max_concurrent))


class FileHandler(FileSystemEventHandler):
    """
    Watchdog handler for processing new audio files.
    
    This handler detects when new audio files are added to a
    directory and processes them automatically.
    """
    
    def __init__(self, callback):
        """
        Initialize the file handler.
        
        Parameters
        ----------
        callback : callable
            Function to call when a new file is detected
        """
        self.callback = callback
        self.extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    
    def on_created(self, event):
        """React to file creation events."""
        if not event.is_directory and self._is_audio_file(event.src_path):
            logger.info(f"New audio file detected: {event.src_path}")
            self.callback(event.src_path)
    
    def on_moved(self, event):
        """React to file move events."""
        if not event.is_directory and self._is_audio_file(event.dest_path):
            logger.info(f"Audio file moved to: {event.dest_path}")
            self.callback(event.dest_path)
    
    def _is_audio_file(self, path):
        """Check if the file is an audio file based on extension."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.extensions


def watch_directory(directory_path: str):
    """
    Watch a directory for new audio files and process them.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to watch
    """
    logger.info(f"Starting directory watch on: {directory_path}")
    
    # Create observer
    observer = PollingObserver()
    handler = FileHandler(process)
    observer.schedule(handler, directory_path, recursive=False)
    observer.start()
    
    try:
        while True:
            asyncio.run(asyncio.sleep(1))
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()


def cleanup_temp():
    """Clean up temporary files."""
    cleaner = Cleaner()
    temp_dir = ".temp"
    if os.path.exists(temp_dir):
        cleaner.clean_directory(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing pipeline")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Parser for processing a single file
    file_parser = subparsers.add_parser("file", help="Process a single audio file")
    file_parser.add_argument("path", type=str, help="Path to the audio file")
    
    # Parser for processing multiple files
    batch_parser = subparsers.add_parser("batch", help="Process multiple audio files")
    batch_parser.add_argument("paths", type=str, nargs="+", help="Paths to audio files")
    batch_parser.add_argument("--max-concurrent", type=int, default=3, 
                             help="Maximum number of files to process concurrently")
    
    # Parser for watching a directory
    watch_parser = subparsers.add_parser("watch", help="Watch a directory for new audio files")
    watch_parser.add_argument("directory", type=str, help="Directory to watch")
    
    # Parser for cleaning temporary files
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up temporary files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Clean up any existing temporary files
    cleanup_temp()
    
    # Execute the appropriate function based on the mode
    if args.mode == "file":
        process(args.path)
    elif args.mode == "batch":
        process_batch(args.paths, args.max_concurrent)
    elif args.mode == "watch":
        watch_directory(args.directory)
    elif args.mode == "cleanup":
        cleanup_temp()
    else:
        parser.print_help()