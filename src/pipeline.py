# Standard library imports
import os
import asyncio
import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Callable

# Related third-party imports
import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

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
from src.utils.utils import Cleaner
from src.db.manager import Database
from src.utils.gpu_utils import setup_gpu_streams, optimize_gpu_memory, stream_manager
from src.utils.monitor import start_monitoring, stop_monitoring, record_stage_start, record_stage_end, record_queue_depth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline")


class AsyncPipeline:
    """
    An asynchronous pipeline for audio processing and analysis.
    
    This pipeline breaks down the audio processing into stages that run concurrently:
    1. VAD and chunking
    2. Speech enhancement and vocal separation
    3. Diarization
    4. ASR (transcription)
    5. Punctuation restoration and sentence mapping
    6. LLM-based analysis
    
    Each stage communicates with the next via asyncio Queues, enabling streaming
    processing where a chunk can move to the next stage as soon as it's ready.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", chunk_duration: float = 5.0):
        """
        Initialize the pipeline with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        chunk_duration : float
            Duration of audio chunks in seconds
        """
        self.config = OmegaConf.load(config_path)
        self.chunk_duration = chunk_duration
        self.temp_dir = ".temp"
        self.device = self.config.runtime.device
        self.compute_type = self.config.runtime.compute_type
        
        # Set up thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        # Set up process pool for heavier CPU-bound tasks
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        
        # Initialize queues for inter-stage communication
        self.queues = {
            "vad_to_enhance": asyncio.Queue(maxsize=2),
            "enhance_to_diar": asyncio.Queue(maxsize=2),
            "diar_to_asr": asyncio.Queue(maxsize=2),
            "asr_to_punct": asyncio.Queue(maxsize=2),
            "punct_to_analysis": asyncio.Queue(maxsize=2)
        }
        
        # Pipeline metadata
        self.total_chunks = 0
        self.active_tasks = set()
        
        # Initialize DB connection
        self.db = Database(".db/Callytics.sqlite")
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set up GPU optimization
        if torch.cuda.is_available():
            # Set up GPU streams for different pipeline stages
            setup_gpu_streams()
            # Apply memory optimizations
            optimize_gpu_memory()
            
            # Set environment variables for CUDA
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.config.runtime.cuda_alloc_conf
            
            logger.info(f"GPU optimizations enabled on device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU available, running in CPU mode")
        
        # Start performance monitoring
        start_monitoring()
    
    async def run_in_thread_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run a CPU-bound function in the thread pool."""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, 
            lambda: func(*args, **kwargs)
        )
    
    async def run_in_process_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run a CPU-intensive function in the process pool."""
        return await asyncio.get_event_loop().run_in_executor(
            self.process_pool, 
            lambda: func(*args, **kwargs)
        )
    
    async def run_gpu_task(self, stream_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Run a GPU-bound task on a specific CUDA stream.
        
        Parameters
        ----------
        stream_name : str
            Name of the CUDA stream to use
        func : Callable
            Function to run
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Any
            Result of the function
        """
        if not torch.cuda.is_available():
            return await self.run_in_thread_executor(func, *args, **kwargs)
        
        return await self.run_in_thread_executor(
            lambda: stream_manager.run_on_stream(stream_name, func, *args, **kwargs)
        )
    
    async def stage_vad_and_chunk(self, audio_path: str, queue_out: asyncio.Queue) -> None:
        """
        First stage: VAD detection and chunking.
        
        Detects speech in audio and splits it into chunks for parallel processing.
        """
        try:
            logger.info(f"Starting VAD and chunking for {audio_path}")
            task_id = f"vad_{os.path.basename(audio_path)}"
            record_stage_start("vad", task_id)
            
            # Check if audio contains dialogue
            dialogue_detector = DialogueDetecting(delete_original=False)
            has_dialogue = await self.run_in_thread_executor(
                dialogue_detector.process, audio_path
            )
            
            if not has_dialogue:
                logger.info(f"No dialogue detected in {audio_path}")
                await queue_out.put(None)  # Signal end of processing
                record_stage_end("vad", task_id)
                return
            
            # Initialize audio processor
            processor = AudioProcessor(audio_path, self.temp_dir)
            
            # Get audio duration
            duration = await self.run_in_thread_executor(processor.get_duration)
            
            # Calculate number of chunks
            self.total_chunks = int((duration + self.chunk_duration - 1) // self.chunk_duration)
            logger.info(f"Audio will be processed in {self.total_chunks} chunks")
            
            # Split audio into chunks
            chunk_paths = await self.run_in_thread_executor(
                processor.split_audio, self.chunk_duration
            )
            
            # Send chunks to the next stage
            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"Sending chunk {i+1}/{len(chunk_paths)} to enhancement stage")
                await queue_out.put((i, chunk_path))
                
                # Record queue depth metrics
                record_queue_depth("vad_to_enhance", queue_out.qsize())
            
            # Signal end of chunks
            await queue_out.put(None)
            
            # Record stage completion
            record_stage_end("vad", task_id)
            
        except Exception as e:
            logger.error(f"Error in VAD stage: {e}")
            await queue_out.put(None)
            record_stage_end("vad", task_id)
            raise
    
    async def stage_enhance_and_separate(self, queue_in: asyncio.Queue, queue_out: asyncio.Queue) -> None:
        """
        Second stage: Speech enhancement and vocal separation.
        
        Enhances audio quality and separates vocals from background noise.
        """
        try:
            # Initialize components
            enhancer = SpeechEnhancement(config_path="config/config.yaml", output_dir=self.temp_dir)
            separator = DemucsVocalSeparator()
            
            while True:
                # Monitor queue depth
                record_queue_depth("enhance_to_diar", queue_out.qsize())
                record_queue_depth("vad_to_enhance", queue_in.qsize())
                
                item = await queue_in.get()
                
                # Check for end signal
                if item is None:
                    logger.info("Enhancement stage complete")
                    await queue_out.put(None)
                    break
                
                chunk_idx, chunk_path = item
                task_id = f"enhance_{chunk_idx}"
                record_stage_start("enhance", task_id)
                logger.info(f"Enhancing chunk {chunk_idx}")
                
                # Enhance audio (CPU-bound)
                enhanced_path = await self.run_in_thread_executor(
                    enhancer.enhance_audio,
                    chunk_path,
                    os.path.join(self.temp_dir, f"enhanced_{chunk_idx}.wav"),
                    0.0001,
                    True
                )
                
                # Separate vocals (GPU-bound)
                vocal_path = await self.run_gpu_task(
                    "demucs",
                    separator.separate_vocals,
                    enhanced_path, 
                    self.temp_dir
                )
                
                # Send to next stage
                await queue_out.put((chunk_idx, vocal_path))
                
                # Record stage completion
                record_stage_end("enhance", task_id)
                
        except Exception as e:
            logger.error(f"Error in enhancement stage: {e}")
            await queue_out.put(None)
            raise
    
    async def stage_diarization(self, queue_in: asyncio.Queue, queue_out: asyncio.Queue) -> None:
        """
        Third stage: Speaker diarization.
        
        Identifies different speakers in the audio.
        """
        try:
            # Load NeMo configuration
            cfg = OmegaConf.load("config/nemo/diar_infer_telephonic.yaml")
            cfg.diarizer.out_dir = self.temp_dir
            
            # Initialize diarizer (done once)
            msdd_model = NeuralDiarizer(cfg=cfg)
            
            while True:
                # Monitor queue depth
                record_queue_depth("diar_to_asr", queue_out.qsize())
                record_queue_depth("enhance_to_diar", queue_in.qsize())
                
                item = await queue_in.get()
                
                # Check for end signal
                if item is None:
                    logger.info("Diarization stage complete")
                    await queue_out.put(None)
                    break
                
                chunk_idx, vocal_path = item
                task_id = f"diar_{chunk_idx}"
                record_stage_start("diarization", task_id)
                logger.info(f"Diarizing chunk {chunk_idx}")
                
                # Convert to mono (required for diarization)
                processor = AudioProcessor(vocal_path, self.temp_dir)
                mono_path = await self.run_in_thread_executor(processor.convert_to_mono)
                
                # Create manifest for this chunk
                manifest_path = os.path.join(self.temp_dir, f"manifest_{chunk_idx}.json")
                await self.run_in_thread_executor(
                    processor.create_manifest, manifest_path
                )
                
                # Run diarization (GPU-bound)
                cfg.diarizer.manifest_filepath = manifest_path
                diar_output_dir = os.path.join(self.temp_dir, f"diar_{chunk_idx}")
                cfg.diarizer.out_dir = diar_output_dir
                
                # Ensure output directory exists
                os.makedirs(diar_output_dir, exist_ok=True)
                os.makedirs(os.path.join(diar_output_dir, "pred_rttms"), exist_ok=True)
                
                # Run diarization with GPU stream
                await self.run_gpu_task("diarization", msdd_model.diarize)
                
                # Get RTTM file path
                rttm_path = os.path.join(diar_output_dir, "pred_rttms", "mono_file.rttm")
                
                # Send to next stage
                await queue_out.put((chunk_idx, vocal_path, mono_path, rttm_path))
                
                # Record stage completion
                record_stage_end("diarization", task_id)
                
        except Exception as e:
            logger.error(f"Error in diarization stage: {e}")
            await queue_out.put(None)
            raise
    
    async def stage_transcription(self, queue_in: asyncio.Queue, queue_out: asyncio.Queue) -> None:
        """
        Fourth stage: ASR transcription.
        
        Transcribes speech to text and aligns words with timestamps.
        """
        try:
            # Initialize components
            transcriber = Transcriber(
                model_name=self.config.asr.model,
                device=self.device,
                compute_type=self.compute_type
            )
            aligner = ForcedAligner(device=self.device)
            
            while True:
                # Monitor queue depth
                record_queue_depth("asr_to_punct", queue_out.qsize())
                record_queue_depth("diar_to_asr", queue_in.qsize())
                
                item = await queue_in.get()
                
                # Check for end signal
                if item is None:
                    logger.info("Transcription stage complete")
                    await queue_out.put(None)
                    break
                
                chunk_idx, vocal_path, mono_path, rttm_path = item
                task_id = f"asr_{chunk_idx}"
                record_stage_start("asr", task_id)
                logger.info(f"Transcribing chunk {chunk_idx}")
                
                # Run ASR (GPU-bound)
                transcript, info = await self.run_gpu_task(
                    "asr",
                    transcriber.transcribe, 
                    vocal_path
                )
                
                # Skip empty transcripts
                if not transcript.strip():
                    logger.info(f"Empty transcript for chunk {chunk_idx}, skipping")
                    record_stage_end("asr", task_id)
                    continue
                
                # Detect language
                detected_language = info["language"]
                
                # Forced alignment to get word timestamps (GPU-bound)
                word_timestamps = await self.run_gpu_task(
                    "alignment",
                    aligner.align, 
                    vocal_path, 
                    transcript, 
                    detected_language
                )
                
                # Read speaker timestamps
                speaker_reader = SpeakerTimestampReader(rttm_path)
                speaker_ts = await self.run_in_thread_executor(
                    speaker_reader.read_speaker_timestamps
                )
                
                # Send to next stage
                await queue_out.put((
                    chunk_idx, 
                    word_timestamps, 
                    speaker_ts, 
                    detected_language
                ))
                
                # Record stage completion
                record_stage_end("asr", task_id)
                
        except Exception as e:
            logger.error(f"Error in transcription stage: {e}")
            await queue_out.put(None)
            raise
    
    async def stage_punctuation_and_mapping(self, queue_in: asyncio.Queue, queue_out: asyncio.Queue) -> None:
        """
        Fifth stage: Punctuation restoration and speaker mapping.
        
        Restores punctuation to transcripts and maps words to speakers.
        """
        try:
            # Results container for all chunks (to maintain order)
            chunks_results = {}
            
            while True:
                # Monitor queue depth
                record_queue_depth("punct_to_analysis", queue_out.qsize())
                record_queue_depth("asr_to_punct", queue_in.qsize())
                
                item = await queue_in.get()
                
                # Check for end signal
                if item is None:
                    logger.info("Processing remaining chunks in punctuation stage")
                    
                    # Process any remaining chunks
                    if chunks_results:
                        combined_ssm = self.combine_chunks_results(chunks_results)
                        await queue_out.put(combined_ssm)
                    
                    await queue_out.put(None)
                    break
                
                chunk_idx, word_timestamps, speaker_ts, detected_language = item
                task_id = f"punct_{chunk_idx}"
                record_stage_start("punctuation", task_id)
                logger.info(f"Processing punctuation for chunk {chunk_idx}")
                
                # Map words to speakers
                word_speaker_mapper = WordSpeakerMapper(word_timestamps, speaker_ts)
                wsm = await self.run_in_thread_executor(
                    word_speaker_mapper.get_words_speaker_mapping
                )
                
                # Restore punctuation
                punct_restorer = PunctuationRestorer(language=detected_language)
                wsm = await self.run_in_thread_executor(
                    punct_restorer.restore_punctuation, wsm
                )
                
                # Realign with punctuation
                word_speaker_mapper.word_speaker_mapping = wsm
                await self.run_in_thread_executor(
                    word_speaker_mapper.realign_with_punctuation
                )
                wsm = word_speaker_mapper.word_speaker_mapping
                
                # Map sentences to speakers
                sentence_mapper = SentenceSpeakerMapper()
                ssm = await self.run_in_thread_executor(
                    sentence_mapper.get_sentences_speaker_mapping, wsm
                )
                
                # Store results by chunk index
                chunks_results[chunk_idx] = ssm
                
                # If we have all chunks, send combined results
                if len(chunks_results) == self.total_chunks:
                    combined_ssm = self.combine_chunks_results(chunks_results)
                    await queue_out.put(combined_ssm)
                    chunks_results = {}  # Clear for next batch
                
                # Record stage completion
                record_stage_end("punctuation", task_id)
                
        except Exception as e:
            logger.error(f"Error in punctuation stage: {e}")
            await queue_out.put(None)
            raise
    
    async def stage_analysis(self, queue_in: asyncio.Queue) -> Dict:
        """
        Final stage: LLM-based analysis.
        
        Performs various analyses on the transcribed text using LLMs.
        """
        try:
            # Initialize components
            llm_handler = LLMOrchestrator(
                config_path="config/config.yaml",
                prompt_config_path="config/prompt.yaml",
                model_id="openai"
            )
            llm_result_handler = LLMResultHandler()
            
            # Monitor queue depth
            record_queue_depth("punct_to_analysis", queue_in.qsize())
            
            # Get the complete transcript
            ssm = await queue_in.get()
            
            if ssm is None:
                logger.info("No transcript to analyze")
                return {}
            
            task_id = "analysis"
            record_stage_start("analysis", task_id)
            logger.info("Starting LLM analysis")
            
            # Run parallel analysis tasks
            classification_task = llm_handler.generate("Classification", ssm)
            sentiment_task = llm_handler.generate("SentimentAnalysis", ssm)
            profanity_task = llm_handler.generate("ProfanityWordDetection", ssm)
            summary_task = llm_handler.generate("Summary", ssm)
            conflict_task = llm_handler.generate("ConflictDetection", ssm)
            
            # Fetch topics from DB
            topics = await self.run_in_thread_executor(
                self.db.fetch, "src/db/sql/TopicFetch.sql"
            )
            
            # Run topic detection with existing topics
            topic_task = llm_handler.generate(
                "TopicDetection",
                user_input=ssm,
                system_input=topics
            )
            
            # Wait for all tasks to complete - LLM tasks run in parallel
            results = await asyncio.gather(
                classification_task,
                sentiment_task,
                profanity_task,
                summary_task,
                conflict_task,
                topic_task
            )
            
            # Process results
            speaker_roles = results[0]
            sentiment_results = results[1]
            profane_results = results[2]
            summary_result = results[3]
            conflict_result = results[4]
            topic_result = results[5]
            
            # Validate speaker roles and fall back if needed
            ssm = llm_result_handler.validate_and_fallback(speaker_roles, ssm)
            llm_result_handler.log_result(ssm, speaker_roles)
            
            # Add indices to sentences for annotation
            formatter = Formatter()
            ssm_with_indices = formatter.add_indices_to_ssm(ssm)
            
            # Create annotator and add all analysis results
            annotator = Annotator(ssm_with_indices)
            annotator.add_sentiment(sentiment_results)
            annotator.add_profanity(profane_results)
            annotator.add_summary(summary_result)
            annotator.add_conflict(conflict_result)
            annotator.add_topic(topic_result)
            
            # Finalize the output
            final_output = annotator.finalize()
            
            # Write transcript files
            writer = TranscriptWriter()
            await self.run_in_thread_executor(
                writer.write_transcript, 
                ssm, 
                os.path.join(self.temp_dir, "output.txt")
            )
            await self.run_in_thread_executor(
                writer.write_srt,
                ssm,
                os.path.join(self.temp_dir, "output.srt")
            )
            
            # Store in database
            await self.run_in_thread_executor(
                self.db.insert,
                "src/db/sql/UtteranceInsert.sql", 
                final_output
            )
            
            # Record stage completion
            record_stage_end("analysis", task_id)
            logger.info("Analysis complete")
            return final_output
            
        except Exception as e:
            logger.error(f"Error in analysis stage: {e}")
            return {}
    
    def combine_chunks_results(self, chunks_results: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Combine results from multiple chunks into a single list.
        
        Parameters
        ----------
        chunks_results : Dict[int, List[Dict]]
            Results from each chunk, keyed by chunk index
            
        Returns
        -------
        List[Dict]
            Combined results in proper sequence
        """
        combined = []
        # Sort by chunk index to maintain proper sequence
        for idx in sorted(chunks_results.keys()):
            chunk_ssm = chunks_results[idx]
            
            # Adjust timestamps for chunk position
            chunk_offset = idx * self.chunk_duration * 1000  # ms
            for sentence in chunk_ssm:
                sentence["start_time"] += chunk_offset
                sentence["end_time"] += chunk_offset
                
            combined.extend(chunk_ssm)
            
        return combined
    
    async def process(self, audio_path: str) -> Dict:
        """
        Process an audio file through the entire pipeline.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file
            
        Returns
        -------
        Dict
            Analysis results
        """
        # Clear queues
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
        
        # Start all pipeline stages
        tasks = [
            self.stage_vad_and_chunk(audio_path, self.queues["vad_to_enhance"]),
            self.stage_enhance_and_separate(self.queues["vad_to_enhance"], self.queues["enhance_to_diar"]),
            self.stage_diarization(self.queues["enhance_to_diar"], self.queues["diar_to_asr"]),
            self.stage_transcription(self.queues["diar_to_asr"], self.queues["asr_to_punct"]),
            self.stage_punctuation_and_mapping(self.queues["asr_to_punct"], self.queues["punct_to_analysis"]),
            self.stage_analysis(self.queues["punct_to_analysis"])
        ]
        
        # Track tasks
        self.active_tasks = set(tasks[:-1])  # All except the last one
        
        # The last task returns the result
        results = await asyncio.gather(*tasks)
        
        # The result is from the last task (analysis stage)
        return results[-1]
    
    async def cleanup(self):
        """Clean up resources."""
        self.thread_pool.shutdown()
        self.process_pool.shutdown()
        
        # Clean up temp files
        cleaner = Cleaner()
        await self.run_in_thread_executor(cleaner.clean_directory, self.temp_dir)
        
        # Synchronize all GPU streams
        if torch.cuda.is_available():
            stream_manager.synchronize_all()
            torch.cuda.empty_cache()


class MultiCallProcessor:
    """
    Process multiple audio files in parallel.
    
    This class creates multiple pipeline instances to process
    several audio files concurrently.
    """
    
    def __init__(self, max_concurrent: int = 3):
        """
        Initialize the multi-call processor.
        
        Parameters
        ----------
        max_concurrent : int
            Maximum number of calls to process concurrently
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_call(self, audio_path: str) -> Dict:
        """
        Process a single call with throttling.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file
            
        Returns
        -------
        Dict
            Analysis results
        """
        async with self.semaphore:
            pipeline = AsyncPipeline()
            try:
                return await pipeline.process(audio_path)
            finally:
                await pipeline.cleanup()
    
    async def process_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Process a batch of calls concurrently.
        
        Parameters
        ----------
        audio_paths : List[str]
            Paths to audio files
            
        Returns
        -------
        List[Dict]
            Analysis results for each audio
        """
        tasks = [self.process_call(path) for path in audio_paths]
        return await asyncio.gather(*tasks) 