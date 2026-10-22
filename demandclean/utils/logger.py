"""
Logging Utilities
=================

Unified logging management and training history recording.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


class DemandCleanLogger:
    """
    DemandClean logger manager

    Provides:
        - Console and file logging output
        - Training history recording
        - JSON-formatted history export
    """

    def __init__(self,
                 config_or_name = "demandclean",
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 to_file: bool = True,
                 to_console: bool = True):
        """
        Initialize the logger manager.

        Args:
            config_or_name: Configuration object or logger name string
            log_dir: Directory for log files
            level: Log level
            to_file: Whether to output to a file
            to_console: Whether to output to the console
        """
        # Handle config object input
        if hasattr(config_or_name, 'save_path'):
            # A configuration object was passed in
            config = config_or_name
            name = "demandclean"
            log_dir = log_dir or config.save_path
        else:
            # A string name was passed in
            name = str(config_or_name)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # Log format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
        )

        # Console output
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File output
        self.log_file = None
        if to_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(log_dir, f'demandclean_{timestamp}.log')
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Training history
        self.history: Dict[str, List] = {
            'episode': [],
            'score': [],
            'reward': [],
            'epsilon': [],
            'no_action': [],
            'repair_value': [],
            'delete': [],
            'replace_nearby': []
        }

        # Best-model tracking
        self.best_score = float('-inf')
        self.best_episode = 0

    def info(self, msg: str) -> None:
        """Log at INFO level."""
        self.logger.info(msg)

    def log_info(self, msg: str) -> None:
        """Log at INFO level (alias)."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log at WARNING level."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log at ERROR level."""
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """Log at DEBUG level."""
        self.logger.debug(msg)

    def log_episode(self,
                    episode: int,
                    score: float,
                    reward: float,
                    epsilon: float,
                    action_counts: Dict[str, int]) -> None:
        """
        Record the result of a single training episode.

        Args:
            episode: Episode index
            score: Model score (accuracy or negative MSE)
            reward: Cumulative reward
            epsilon: Current exploration rate
            action_counts: Action count statistics
        """
        self.history['episode'].append(episode)
        self.history['score'].append(score)
        self.history['reward'].append(reward)
        self.history['epsilon'].append(epsilon)

        for action in ['no_action', 'repair_value', 'delete', 'replace_nearby']:
            self.history[action].append(action_counts.get(action, 0))

        # Update best-so-far
        if score > self.best_score:
            self.best_score = score
            self.best_episode = episode

    def log_training_start(self, config: Any) -> None:
        """Record the start of training."""
        self.info("=" * 60)
        self.info("DemandClean training started")
        self.info("=" * 60)
        self.info(f"Task type: {config.task_type.value}")
        self.info(f"Model type: {config.model_type.value}")
        self.info(f"Agent type: {config.agent_type.value}")
        self.info(f"Training episodes: {config.n_episodes}")
        self.info(f"Truth budget: [{config.min_truth_budget}, {config.max_truth_budget}]")
        self.info("-" * 60)

    def log_training_end(self) -> None:
        """Record the end of training."""
        self.info("-" * 60)
        self.info("Training completed!")
        self.info(f"Best score: {self.best_score:.4f} (Episode {self.best_episode})")
        self.info("=" * 60)

    def log_inference(self,
                      action_counts: Dict[str, int],
                      repair_log: List[Dict]) -> None:
        """
        Record inference results.

        Args:
            action_counts: Action counts
            repair_log: Repair log entries
        """
        self.info("=" * 50)
        self.info("Inference completed")
        self.info("=" * 50)
        self.info(f"Action statistics:")
        self.info(f"  No-op: {action_counts.get('no_action', 0)}")
        self.info(f"  Truth repair: {action_counts.get('repair_value', 0)}")
        self.info(f"  Delete: {action_counts.get('delete', 0)}")
        self.info(f"  Replace with nearby: {action_counts.get('replace_nearby', 0)}")
        self.info(f"Truth values consumed: {len(repair_log)}")
        self.info("-" * 50)

    def log_two_phase_plan(self, repair_plan: List[Dict]) -> None:
        """Record a two-phase inference plan."""
        self.info("=" * 50)
        self.info("Two-phase inference - Phase 1: repair plan")
        self.info("=" * 50)
        self.info(f"Positions requiring truth repair: {len(repair_plan)}")
        for i, item in enumerate(repair_plan[:10]):  # Show at most 10 entries
            self.info(f"  [{i+1}] Position ({item['idx']}, {item['col']}): "
                     f"estimated value={item.get('estimated_value', 'N/A'):.4f}")
        if len(repair_plan) > 10:
            self.info(f"  ... and {len(repair_plan) - 10} more")
        self.info("-" * 50)

    def log_two_phase_execute(self, repair_count: int) -> None:
        """Record two-phase inference execution."""
        self.info("=" * 50)
        self.info("Two-phase inference - Phase 2: execute repairs")
        self.info("=" * 50)
        self.info(f"Repaired: {repair_count} positions")
        self.info("-" * 50)

    def save_history(self, path: str) -> None:
        """
        Save training history to a JSON file.

        Args:
            path: Destination path
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        self.info(f"Training history saved to: {path}")

    def load_history(self, path: str) -> None:
        """
        Load training history from disk.

        Args:
            path: History file path
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        self.info(f"Training history loaded from: {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a training summary."""
        if not self.history['episode']:
            return {}

        return {
            'total_episodes': len(self.history['episode']),
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'final_score': self.history['score'][-1],
            'final_epsilon': self.history['epsilon'][-1],
            'avg_score_last_50': sum(self.history['score'][-50:]) / min(50, len(self.history['score'])),
        }
