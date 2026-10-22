"""
Single-phase inference
======================

Runs data cleaning with a trained agent in a single pass.
Requires ground-truth values for repairs.
"""

import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import BaseAgent
from ..core.environments import CleaningEnv
from ..core.state import (
    StateExtractor, ClassificationStateExtractor,
    RegressionStateExtractor, ClusteringStateExtractor,
)
from ..models import ModelAdapter, create_model_adapter

# Agent type -> algorithm name
_AGENT_ALGO_NAME = {
    AgentType.SINGLE_STAGE: 'DQN (Single Stage)',
    AgentType.DUELING_SINGLE_STAGE: 'Dueling DQN (Single Stage)',
    AgentType.TWO_STAGE: 'Double DQN (Two Stage)',
    AgentType.DUELING_TWO_STAGE: 'Dueling Double DQN (Two Stage)',
}


class SinglePhaseInference:
    """
    Single-phase inference.

    Runs data cleaning with a trained agent in a single pass.
    Clean data is required to fetch ground-truth repair values.
    """

    def __init__(self,
                 agent: BaseAgent,
                 config: DemandCleanConfig):
        """
        Initialize the inference engine.

        Args:
            agent: trained agent
            config: configuration object
        """
        self.agent = agent
        self.config = config

        # Build the model adapter
        self.model_adapter = create_model_adapter(config.model_type, config.task_type)

        # Build the state extractor
        self.state_extractor = self._create_state_extractor()

        # Reference to the environment used for inference (used to fetch decision_log)
        self._env: Optional[CleaningEnv] = None

    def _create_state_extractor(self) -> StateExtractor:
        """Build the state extractor."""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def clean(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              X_clean: np.ndarray,
              detected_errors: Dict[str, List],
              verbose: bool = True,
              y_clean: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[Dict]]:
        """
        Clean data using detected errors.

        Args:
            X_dirty: dirty data matrix
            y: label vector
            X_clean: clean data (used to fetch ground-truth repairs)
            detected_errors: detected errors
            verbose: whether to print details
            y_clean: clean label vector (used for label-noise repairs)

        Returns:
            (X_clean_result, y_clean_result, keep_mask, action_counts, repair_log)
        """
        # Build the error list (with ground-truth repairs)
        error_list = self._build_error_list(detected_errors, X_clean, y_clean)

        n_missing = len(detected_errors.get('missing', []))
        n_semantic = len(detected_errors.get('semantic', []))
        n_syntactic = len(detected_errors.get('syntactic', []))
        n_label = len(detected_errors.get('label_noise', []))
        total_errors = len(error_list)

        if verbose:
            algo_name = _AGENT_ALGO_NAME.get(self.config.agent_type, self.config.agent_type.value)
            print(f"\n{'='*60}")
            print(f"Single-phase inference")
            print(f"{'='*60}")
            print(f"  Algorithm: {algo_name}")
            print(f"  Task type: {self.config.task_type.value}")
            print(f"  Downstream model: {self.config.model_type.value}")
            print(f"  Detected errors: {total_errors}"
                  f" (missing={n_missing}, semantic={n_semantic},"
                  f" syntactic={n_syntactic}, label={n_label})")

        # Build the environment
        env = CleaningEnv(
            X_dirty, y, error_list,
            self.model_adapter, self.state_extractor, self.config
        )
        self._env = env

        # Switch to inference mode
        self.agent.epsilon = 0
        state = env.reset()

        # Progress-bar parameters
        progress_total = 20
        progress_step = max(1, total_errors // progress_total)
        processed = 0

        if verbose:
            sys.stdout.write(f"\n  Inference progress: [")
            sys.stdout.flush()

        # Run inference
        while True:
            if self.config.agent_type in (AgentType.TWO_STAGE, AgentType.DUELING_TWO_STAGE):
                final_action, _, _ = self.agent.act(state, training=False)
            else:
                final_action = self.agent.act(state, training=False)

            next_state, _, done, _ = env.step(final_action)
            state = next_state
            processed += 1

            # Update progress bar
            if verbose and processed % progress_step == 0:
                sys.stdout.write("=")
                sys.stdout.flush()

            if done:
                break

        if verbose:
            # Finish progress bar
            bars_printed = processed // progress_step
            remaining = progress_total - bars_printed
            sys.stdout.write("=" * remaining + f"] {processed}/{total_errors}\n")
            sys.stdout.flush()

        X_result, y_result, keep_mask = env.get_cleaned_data()
        action_counts = env.get_action_counts()
        repair_log = env.get_repair_log()

        if verbose:
            env.print_decision_summary()

        return X_result, y_result, keep_mask, action_counts, repair_log

    def get_decision_log(self) -> List[Dict]:
        """Return the full decision log produced during inference."""
        if self._env is None:
            return []
        return self._env.get_decision_log()

    def _build_error_list(self,
                          detected_errors: Dict[str, List],
                          X_clean: np.ndarray,
                          y_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """Convert detected errors into the format expected by the environment."""
        error_list = []

        # Missing errors (type=0)
        for item in detected_errors.get('missing', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': true_val
            })

        # Semantic errors (type=1)
        for item in detected_errors.get('semantic', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': true_val
            })

        # Syntactic errors (type=2)
        for item in detected_errors.get('syntactic', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': true_val
            })

        # Label noise errors (type=3, col=-1)
        for item in detected_errors.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                # Prefer y_clean as the source of truth
                if y_clean is not None and idx < len(y_clean):
                    repair_value = y_clean[idx]
                elif len(item) > 2:
                    repair_value = item[2]  # estimated_val
                else:
                    repair_value = float('nan')
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': repair_value
                })

        return error_list

    def get_stats(self) -> Dict[str, Any]:
        """Return inference stats."""
        return {
            'agent_type': self.config.agent_type.value,
            'task_type': self.config.task_type.value,
            'model_type': self.config.model_type.value
        }
