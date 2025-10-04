#!/usr/bin/env python3
"""
Configurable GPU Occupier - Indefinitely occupy GPUs until killed
A cleaner, more configurable version based on occ.sh
"""

import argparse
import configparser
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class GPUOccupierConfig:
    """Configuration class for GPU Occupier with sensible defaults"""

    def __init__(self, config_file: Optional[str] = None):
        # Default configuration
        self.defaults = {
            # GPU Settings
            'gpu_list': None,  # None means auto-detect all GPUs
            'target_utilization': 85,  # Target GPU utilization percentage
            'min_utilization_threshold': 20,  # Below this, increase workload
            'min_power_threshold': 100,  # Below this (watts), increase workload

            # Computation Settings
            'initial_matrix_size': 128,  # Starting matrix size for computations
            'matrix_increment': 16,    # How much to increase matrix size
            'max_matrix_size': 2048,   # Maximum matrix size
            'iterations_per_cycle': 600,  # Matrix operations per cycle
            'cycle_sleep': 0.1,        # Sleep between cycles (seconds)

            # Monitoring Settings
            'monitor_interval': 1,     # How often to check GPU stats (seconds)
            'smoothing_window': 5,     # Sliding window for smoothed averages
            'log_level': 'INFO',       # Logging level
            'log_file': None,          # Log file path (None = auto-generate)

            # Adaptive Behavior
            'enable_adaptive': True,   # Enable adaptive workload adjustment
            'utilization_tolerance': 5,  # ±% tolerance for target utilization
            'adjustment_factor': 1.1,  # Factor for workload adjustments
        }

        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

        # Set up logging
        self._setup_logging()

    def _load_from_file(self, config_file: str):
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        config.read(config_file)

        for section_name, section in config.items():
            if section_name == 'DEFAULT':
                continue
            for key, value in section.items():
                if key in self.defaults:
                    # Type conversion based on default type
                    default_type = type(self.defaults[key])
                    if default_type == bool:
                        self.defaults[key] = config.getboolean(section_name, key)
                    elif default_type == int:
                        self.defaults[key] = config.getint(section_name, key)
                    elif default_type == float:
                        self.defaults[key] = config.getfloat(section_name, key)
                    else:
                        self.defaults[key] = value

    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = self.defaults['log_file']
        if log_file is None:
            hostname = os.uname().nodename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'gpu_occupier_{hostname}_{timestamp}.log'

        log_level = getattr(logging, self.defaults['log_level'].upper())

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info(f"GPU Occupier started - Log file: {log_file}")

    def __getattr__(self, name):
        """Allow accessing config values as attributes"""
        if name in self.defaults:
            return self.defaults[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_gpu_list(self) -> List[int]:
        """Get list of GPU indices to use"""
        if self.gpu_list is None:
            # Auto-detect GPUs
            return self._detect_gpus()
        elif isinstance(self.gpu_list, str):
            # Parse comma-separated list
            return [int(x.strip()) for x in self.gpu_list.split(',')]
        else:
            return self.gpu_list

    def _detect_gpus(self) -> List[int]:
        """Auto-detect available GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_indices = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
                return gpu_indices if gpu_indices else [0]
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            logging.warning("Failed to auto-detect GPUs, defaulting to GPU 0")

        return [0]  # Default fallback


class GPUMonitor:
    """GPU monitoring and statistics collection"""

    def __init__(self, config: GPUOccupierConfig):
        self.config = config
        self.gpu_list = config.get_gpu_list()
        self.num_gpus = len(self.gpu_list)

        # Initialize sliding windows for smoothed averages
        self.utilization_history = deque(maxlen=config.smoothing_window)
        self.power_history = deque(maxlen=config.smoothing_window)

        logging.info(f"Monitoring {self.num_gpus} GPUs: {self.gpu_list}")

    def get_gpu_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current GPU utilization and power consumption"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,power.draw', '--format=csv'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                logging.error(f"nvidia-smi failed: {result.stderr}")
                return {}

            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            gpu_stats = {}

            reader = csv.reader(lines[1:], skipinitialspace=True)
            for row in reader:
                if len(row) >= 3:
                    gpu_idx = int(row[0])
                    utilization = float(row[1].replace('%', '').strip())
                    power_str = row[2].replace('W', '').strip()
                    power = float(power_str) if power_str != 'N/A' else 0.0

                    gpu_stats[gpu_idx] = {
                        'utilization': utilization,
                        'power': power
                    }

            return gpu_stats

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logging.error(f"Failed to get GPU stats: {e}")
            return {}

    def get_smoothed_averages(self) -> Tuple[float, float]:
        """Get smoothed average utilization and power consumption"""
        if not self.utilization_history or not self.power_history:
            return 0.0, 0.0

        avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
        avg_power = sum(self.power_history) / len(self.power_history)

        return avg_utilization, avg_power

    def update_stats(self) -> Optional[Tuple[float, float]]:
        """Update statistics and return smoothed averages"""
        gpu_stats = self.get_gpu_stats()
        if not gpu_stats:
            return None

        # Filter stats for our target GPUs
        target_stats = {gpu_id: stats for gpu_id, stats in gpu_stats.items() if gpu_id in self.gpu_list}

        if not target_stats:
            logging.warning(f"No stats found for target GPUs: {self.gpu_list}")
            return None

        # Calculate averages for target GPUs
        total_utilization = sum(stats['utilization'] for stats in target_stats.values())
        total_power = sum(stats['power'] for stats in target_stats.values())

        avg_utilization = total_utilization / len(target_stats)
        avg_power = total_power / len(target_stats)

        # Update sliding windows
        self.utilization_history.append(avg_utilization)
        self.power_history.append(avg_power)

        return self.get_smoothed_averages()


class GPUWorker:
    """Worker process to occupy a single GPU"""

    def __init__(self, gpu_id: int, config: GPUOccupierConfig, queue: Queue):
        self.gpu_id = gpu_id
        self.config = config
        self.queue = queue
        self.device = torch.device(f'cuda:{gpu_id}')

        # Workload parameters
        self.matrix_size = config.initial_matrix_size
        self.running = True

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logging.info(f"GPU Worker {gpu_id} initialized with matrix size {self.matrix_size}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"GPU Worker {self.gpu_id} received signal {signum}, shutting down...")
        self.running = False

    def adjust_workload(self, command: str):
        """Adjust computational workload based on command"""
        if command == "increase":
            if self.matrix_size < self.config.max_matrix_size:
                old_size = self.matrix_size
                self.matrix_size = min(
                    self.matrix_size + self.config.matrix_increment,
                    self.config.max_matrix_size
                )
                logging.info(f"GPU {self.gpu_id}: Increased matrix size {old_size} -> {self.matrix_size}")

        elif command == "decrease":
            if self.matrix_size > self.config.initial_matrix_size:
                old_size = self.matrix_size
                self.matrix_size = max(
                    self.matrix_size - self.config.matrix_increment,
                    self.config.initial_matrix_size
                )
                logging.info(f"GPU {self.gpu_id}: Decreased matrix size {old_size} -> {self.matrix_size}")

        elif command == "stop":
            logging.info(f"GPU Worker {self.gpu_id} received stop command")
            self.running = False

    def run_computation_cycle(self):
        """Run one cycle of matrix computations"""
        try:
            for _ in range(self.config.iterations_per_cycle):
                if not self.running:
                    break

                # Create random matrices
                matrix_a = torch.randn(self.matrix_size, self.matrix_size, device=self.device)
                matrix_b = torch.randn(self.matrix_size, self.matrix_size, device=self.device)

                # Matrix multiplication
                result = torch.matmul(matrix_a, matrix_b)

                # Additional operations to increase computational load
                result = result * torch.sin(result) + torch.cos(result)

                # Ensure computation completes
                torch.cuda.synchronize()

            # Small sleep between cycles
            if self.config.cycle_sleep > 0:
                time.sleep(self.config.cycle_sleep)

        except Exception as e:
            logging.error(f"GPU Worker {self.gpu_id} computation error: {e}")

    def run(self):
        """Main worker loop"""
        logging.info(f"GPU Worker {self.gpu_id} starting...")

        try:
            while self.running:
                # Check for commands from monitor
                if not self.queue.empty():
                    command = self.queue.get()
                    logging.info(f"GPU Worker {self.gpu_id} received command: {command}")
                    self.adjust_workload(command)

                # Run computation cycle
                self.run_computation_cycle()

            logging.info(f"GPU Worker {self.gpu_id} finished")

        except Exception as e:
            logging.error(f"GPU Worker {self.gpu_id} fatal error: {e}", exc_info=True)


class GPUOccupier:
    """Main GPU Occupier orchestrator"""

    def __init__(self, config: GPUOccupierConfig):
        self.config = config
        self.monitor = GPUMonitor(config)
        self.workers = []
        self.queues = []
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logging.info("GPU Occupier initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"GPU Occupier received signal {signum}, initiating shutdown...")
        self.running = False
        self.shutdown()

    def start_workers(self):
        """Start GPU worker processes"""
        gpu_list = self.config.get_gpu_list()

        for gpu_id in gpu_list:
            queue = Queue()
            self.queues.append(queue)

            worker = Process(target=self._worker_wrapper, args=(gpu_id, queue))
            worker.start()
            self.workers.append(worker)

            logging.info(f"Started worker for GPU {gpu_id}")

    def _worker_wrapper(self, gpu_id: int, queue: Queue):
        """Wrapper for worker process"""
        try:
            worker = GPUWorker(gpu_id, self.config, queue)
            worker.run()
        except Exception as e:
            logging.error(f"Worker {gpu_id} failed: {e}")

    def monitor_and_adjust(self):
        """Monitor GPUs and send adjustment commands to workers"""
        last_command = None

        while self.running:
            try:
                # Update GPU statistics
                stats = self.monitor.update_stats()
                if stats is None:
                    time.sleep(self.config.monitor_interval)
                    continue

                smoothed_utilization, smoothed_power = stats

                # Determine adjustment command
                command = self._determine_command(smoothed_utilization, smoothed_power)

                # Send command to all workers if changed
                if command != last_command:
                    for queue in self.queues:
                        try:
                            queue.put(command)
                        except:
                            pass  # Queue might be full

                    logging.info(f"Command: {command} (Util: {smoothed_utilization:.1f}%, Power: {smoothed_power:.1f}W)")
                    last_command = command

                time.sleep(self.config.monitor_interval)

            except Exception as e:
                logging.error(f"Monitor error: {e}")
                time.sleep(self.config.monitor_interval)

    def _determine_command(self, utilization: float, power: float) -> str:
        """Determine adjustment command based on current metrics"""
        if not self.config.enable_adaptive:
            return "maintain"

        target = self.config.target_utilization
        tolerance = self.config.utilization_tolerance

        if utilization < self.config.min_utilization_threshold or power < self.config.min_power_threshold:
            return "increase"
        elif utilization < target - tolerance:
            return "increase"
        elif utilization > target + tolerance:
            return "decrease"
        else:
            return "maintain"

    def run(self):
        """Main execution loop"""
        logging.info("Starting GPU occupation...")

        try:
            # Start worker processes
            self.start_workers()

            # Start monitoring and adjustment
            self.monitor_and_adjust()

        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
        except Exception as e:
            logging.error(f"Fatal error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        if not self.running:
            return

        self.running = False
        logging.info("Shutting down GPU Occupier...")

        # Send stop commands to all workers
        for queue in self.queues:
            try:
                queue.put("stop")
            except:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            try:
                worker.join(timeout=5)
                if worker.is_alive():
                    logging.warning(f"Force terminating worker {worker.pid}")
                    worker.terminate()
                    worker.join(timeout=2)
            except:
                pass

        logging.info("GPU Occupier shutdown complete")


def create_default_config(config_file: str):
    """Create a default configuration file"""
    config = configparser.ConfigParser()

    config['gpu'] = {
        '# GPU Settings': '',
        'gpu_list': '# Comma-separated list of GPU indices (empty = auto-detect)',
        'target_utilization': '85',
        'min_utilization_threshold': '20',
        'min_power_threshold': '100'
    }

    config['computation'] = {
        '# Computation Settings': '',
        'initial_matrix_size': '128',
        'matrix_increment': '16',
        'max_matrix_size': '2048',
        'iterations_per_cycle': '600',
        'cycle_sleep': '0.1'
    }

    config['monitoring'] = {
        '# Monitoring Settings': '',
        'monitor_interval': '1',
        'smoothing_window': '5',
        'log_level': 'INFO',
        'log_file': '# Leave empty for auto-generated filename'
    }

    config['adaptive'] = {
        '# Adaptive Behavior': '',
        'enable_adaptive': 'true',
        'utilization_tolerance': '5',
        'adjustment_factor': '1.1'
    }

    with open(config_file, 'w') as f:
        config.write(f)

    print(f"Created default configuration file: {config_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Configurable GPU Occupier - Indefinitely occupy GPUs until killed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto-detected GPUs
  python gpu_occupier.py

  # Use specific GPUs
  python gpu_occupier.py --gpus 0,1,2,3

  # Use configuration file
  python gpu_occupier.py --config my_config.ini

  # Create default configuration file
  python gpu_occupier.py --create-config
        """
    )

    parser.add_argument('--gpus', type=str, help='Comma-separated list of GPU indices to use')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file and exit')
    parser.add_argument('--target-util', type=int, help='Target GPU utilization percentage')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    args = parser.parse_args()

    if args.create_config:
        config_file = args.config or 'gpu_occupier.ini'
        create_default_config(config_file)
        return

    # Initialize configuration
    config = GPUOccupierConfig(args.config)

    # Override with command line arguments
    if args.gpus:
        config.defaults['gpu_list'] = args.gpus
    if args.target_util:
        config.defaults['target_utilization'] = args.target_util
    if args.log_level:
        config.defaults['log_level'] = args.log_level

    # Create and run occupier
    occupier = GPUOccupier(config)
    occupier.run()


if __name__ == "__main__":
    main()