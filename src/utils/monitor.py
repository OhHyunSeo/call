# Standard library imports
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline_monitor")


class MetricsCollector:
    """
    Collects and maintains performance metrics for the pipeline.
    
    This class tracks metrics such as processing time, queue depths,
    and throughput for different stages of the pipeline.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the metrics collector.
        
        Parameters
        ----------
        max_history : int
            Maximum number of historical metrics to keep
        """
        self.max_history = max_history
        self.stage_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.queue_depths: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.throughputs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.start_times: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.active_tasks: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
    
    def start_timer(self, stage: str, task_id: str) -> None:
        """
        Start timing a specific task in a pipeline stage.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        task_id : str
            Unique identifier for the task
        """
        with self.lock:
            self.start_times[stage][task_id] = time.time()
            self.active_tasks[stage] += 1
    
    def stop_timer(self, stage: str, task_id: str) -> float:
        """
        Stop timing a task and record the elapsed time.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        task_id : str
            Unique identifier for the task
            
        Returns
        -------
        float
            Elapsed time in seconds
        """
        with self.lock:
            if stage not in self.start_times or task_id not in self.start_times[stage]:
                logger.warning(f"No start time found for {stage}:{task_id}")
                return 0.0
                
            elapsed = time.time() - self.start_times[stage][task_id]
            self.stage_timings[stage].append(elapsed)
            del self.start_times[stage][task_id]
            
            if self.active_tasks[stage] > 0:
                self.active_tasks[stage] -= 1
                
            return elapsed
    
    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """
        Record the current depth of a queue.
        
        Parameters
        ----------
        queue_name : str
            Name of the queue
        depth : int
            Current depth of the queue
        """
        with self.lock:
            self.queue_depths[queue_name].append((time.time(), depth))
    
    def record_throughput(self, stage: str, items_processed: int, time_window: float) -> None:
        """
        Record throughput for a stage.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        items_processed : int
            Number of items processed
        time_window : float
            Time window in seconds
        """
        with self.lock:
            throughput = items_processed / time_window if time_window > 0 else 0
            self.throughputs[stage].append((time.time(), throughput))
    
    def get_stage_metrics(self, stage: str) -> Dict[str, Any]:
        """
        Get metrics for a specific pipeline stage.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        with self.lock:
            timings = list(self.stage_timings[stage])
            if not timings:
                return {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "p95": 0.0,
                    "active_tasks": self.active_tasks[stage]
                }
                
            timings.sort()
            p95_idx = int(len(timings) * 0.95)
            return {
                "min": min(timings),
                "max": max(timings),
                "avg": sum(timings) / len(timings),
                "p95": timings[p95_idx] if p95_idx < len(timings) else timings[-1],
                "active_tasks": self.active_tasks[stage]
            }
    
    def get_queue_metrics(self, queue_name: str) -> Dict[str, Any]:
        """
        Get metrics for a specific queue.
        
        Parameters
        ----------
        queue_name : str
            Name of the queue
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        with self.lock:
            depths = [d for _, d in self.queue_depths[queue_name]]
            if not depths:
                return {
                    "current": 0,
                    "min": 0,
                    "max": 0,
                    "avg": 0.0
                }
                
            return {
                "current": depths[-1] if depths else 0,
                "min": min(depths),
                "max": max(depths),
                "avg": sum(depths) / len(depths)
            }
    
    def get_throughput_metrics(self, stage: str) -> Dict[str, Any]:
        """
        Get throughput metrics for a specific stage.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        with self.lock:
            throughputs = [t for _, t in self.throughputs[stage]]
            if not throughputs:
                return {
                    "current": 0.0,
                    "avg": 0.0
                }
                
            return {
                "current": throughputs[-1] if throughputs else 0.0,
                "avg": sum(throughputs) / len(throughputs)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics for all stages and queues.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of all metrics
        """
        with self.lock:
            result = {
                "timestamp": datetime.now().isoformat(),
                "stages": {},
                "queues": {},
                "throughputs": {}
            }
            
            for stage in self.stage_timings.keys():
                result["stages"][stage] = self.get_stage_metrics(stage)
                
            for queue in self.queue_depths.keys():
                result["queues"][queue] = self.get_queue_metrics(queue)
                
            for stage in self.throughputs.keys():
                result["throughputs"][stage] = self.get_throughput_metrics(stage)
                
            return result
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.stage_timings.clear()
            self.queue_depths.clear()
            self.throughputs.clear()
            self.start_times.clear()
            self.active_tasks.clear()


class PipelineMonitor:
    """
    Monitors and reports on pipeline performance.
    
    This class periodically collects metrics about the pipeline
    and can report them for monitoring purposes.
    """
    
    def __init__(self, interval: float = 5.0):
        """
        Initialize the pipeline monitor.
        
        Parameters
        ----------
        interval : float
            Interval in seconds between metrics collections
        """
        self.interval = interval
        self.collector = MetricsCollector()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Pipeline monitoring started")
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Pipeline monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background thread that periodically collects and logs metrics."""
        while self.running:
            try:
                metrics = self.collector.get_all_metrics()
                self._log_metrics(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log the collected metrics.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metrics to log
        """
        # Log stage timing metrics
        for stage, stage_metrics in metrics["stages"].items():
            if stage_metrics["avg"] > 0:
                logger.info(
                    f"Stage {stage}: avg={stage_metrics['avg']:.3f}s, "
                    f"p95={stage_metrics['p95']:.3f}s, "
                    f"active={stage_metrics['active_tasks']}"
                )
        
        # Log queue depth metrics
        for queue, queue_metrics in metrics["queues"].items():
            if queue_metrics["current"] > 0:
                logger.info(
                    f"Queue {queue}: current={queue_metrics['current']}, "
                    f"max={queue_metrics['max']}"
                )
        
        # Log throughput metrics
        for stage, throughput_metrics in metrics["throughputs"].items():
            if throughput_metrics["current"] > 0:
                logger.info(
                    f"Throughput {stage}: "
                    f"current={throughput_metrics['current']:.2f} items/s"
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of current metrics
        """
        return self.collector.get_all_metrics()
    
    def record_stage_start(self, stage: str, task_id: str) -> None:
        """
        Record the start of a stage for a specific task.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        task_id : str
            Unique identifier for the task
        """
        self.collector.start_timer(stage, task_id)
    
    def record_stage_end(self, stage: str, task_id: str) -> float:
        """
        Record the end of a stage for a specific task.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        task_id : str
            Unique identifier for the task
            
        Returns
        -------
        float
            Elapsed time in seconds
        """
        return self.collector.stop_timer(stage, task_id)
    
    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """
        Record the current depth of a queue.
        
        Parameters
        ----------
        queue_name : str
            Name of the queue
        depth : int
            Current depth of the queue
        """
        self.collector.record_queue_depth(queue_name, depth)
    
    def record_throughput(self, stage: str, items_processed: int, time_window: float) -> None:
        """
        Record throughput for a stage.
        
        Parameters
        ----------
        stage : str
            Name of the pipeline stage
        items_processed : int
            Number of items processed
        time_window : float
            Time window in seconds
        """
        self.collector.record_throughput(stage, items_processed, time_window)


# Create a global monitor instance
pipeline_monitor = PipelineMonitor()


def start_monitoring() -> None:
    """Start pipeline monitoring."""
    pipeline_monitor.start()


def stop_monitoring() -> None:
    """Stop pipeline monitoring."""
    pipeline_monitor.stop()


def get_metrics() -> Dict[str, Any]:
    """
    Get current pipeline metrics.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of current metrics
    """
    return pipeline_monitor.get_metrics()


# Convenience functions for recording metrics
def record_stage_start(stage: str, task_id: str) -> None:
    """Record stage start time."""
    pipeline_monitor.record_stage_start(stage, task_id)


def record_stage_end(stage: str, task_id: str) -> float:
    """Record stage end time and return elapsed time."""
    return pipeline_monitor.record_stage_end(stage, task_id)


def record_queue_depth(queue_name: str, depth: int) -> None:
    """Record queue depth."""
    pipeline_monitor.record_queue_depth(queue_name, depth)


def record_throughput(stage: str, items_processed: int, time_window: float) -> None:
    """Record throughput for a stage."""
    pipeline_monitor.record_throughput(stage, items_processed, time_window) 