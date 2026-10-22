"""
Two-phase inference
===================

Phase 1: produce a repair plan without requiring ground truths.
Phase 2: execute the plan after the user supplies ground truths.
"""

import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import BaseAgent
from ..core.environments import TwoPhaseCleaningEnv
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


class TwoPhaseInference:
    """
    Two-phase inference.

    Phase 1 (plan): predict every action; repair_value is added to the plan while
        other actions execute immediately.
    Phase 2 (execute): after the user supplies ground truths, execute the planned repairs.
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

        # Two-phase environment
        self._env: Optional[TwoPhaseCleaningEnv] = None
        self._repair_plan: List[Dict] = []

    def _create_state_extractor(self) -> StateExtractor:
        """Build the state extractor."""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def plan(self,
             X_dirty: np.ndarray,
             y: np.ndarray,
             detected_errors: Dict[str, List],
             verbose: bool = True,
             save_csv_path: Optional[str] = None) -> List[Dict]:
        """
        Phase 1: build the repair plan.

        Ground truths are not needed; returns the list of positions that need repair.

        Args:
            X_dirty: dirty data matrix
            y: label vector
            detected_errors: detected errors
            verbose: whether to print details
            save_csv_path: optional path to save the repair-plan CSV

        Returns:
            repair_plan: positions that require ground-truth repair
                [{'idx', 'col', 'error_type', 'estimated_value', 'current_dirty_value'}, ...]
        """
        # Build the error list (no ground truth required)
        error_list = self._build_error_list_no_truth(detected_errors)

        n_missing = len(detected_errors.get('missing', []))
        n_semantic = len(detected_errors.get('semantic', []))
        n_syntactic = len(detected_errors.get('syntactic', []))
        n_label = len(detected_errors.get('label_noise', []))
        total_errors = len(error_list)

        if verbose:
            algo_name = _AGENT_ALGO_NAME.get(self.config.agent_type, self.config.agent_type.value)
            print(f"\n{'='*60}")
            print(f"Two-phase inference - Phase 1 (Plan)")
            print(f"{'='*60}")
            print(f"  Algorithm: {algo_name}")
            print(f"  Task type: {self.config.task_type.value}")
            print(f"  Downstream model: {self.config.model_type.value}")
            print(f"  Detected errors: {total_errors}"
                  f" (missing={n_missing}, semantic={n_semantic},"
                  f" syntactic={n_syntactic}, label={n_label})")

        # Build the two-phase environment
        self._env = TwoPhaseCleaningEnv(
            X_dirty, y, error_list,
            self.model_adapter, self.state_extractor, self.config
        )

        # Switch to inference mode
        self.agent.epsilon = 0
        state = self._env.reset()

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

            next_state, _, done, _ = self._env.step(final_action)
            state = next_state
            processed += 1

            # Update progress bar
            if verbose and processed % progress_step == 0:
                sys.stdout.write("=")
                sys.stdout.flush()

            if done:
                break

        if verbose:
            bars_printed = processed // progress_step
            remaining = progress_total - bars_printed
            sys.stdout.write("=" * remaining + f"] {processed}/{total_errors}\n")
            sys.stdout.flush()

        self._repair_plan = self._env.get_repair_plan()

        if verbose:
            self._env.print_decision_summary()
            print(f"\n  User must supply {len(self._repair_plan)} ground-truth values")

        # Optionally save the plan to CSV
        if save_csv_path:
            self._env.save_plan_csv(save_csv_path)
            if verbose:
                print(f"  Repair plan saved: {save_csv_path}")

        return self._repair_plan

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """
        Return the positions that need ground-truth repairs.

        Returns:
            [(idx, col), ...] - positions the user must supply truths for.
        """
        if self._env is None:
            return []
        return self._env.get_plan_positions()

    def execute(self,
                X_dirty: np.ndarray,
                true_values: Dict[Tuple[int, int], float],
                verbose: bool = True,
                y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase 2: execute the repairs.

        Args:
            X_dirty: original dirty data
            true_values: ground-truth dict {(idx, col): value}
            verbose: whether to print details
            y_dirty: original dirty labels (required for label repairs)

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        if self._env is None:
            raise ValueError("Call plan() first to build the repair plan")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Two-phase inference - Phase 2 (Execute)")
            print(f"{'='*60}")
            print(f"  Ground-truth values provided: {len(true_values)}")
            print(f"  Planned repairs: {len(self._repair_plan)}")

        # Fetch keep_mask and labels from the environment for consistency
        _, y_from_env, keep_mask = self._env.get_cleaned_data()

        # Execute repairs (pass y_dirty to enable label repairs)
        X_result, y_result = self._env.execute_repair_plan(
            X_dirty, true_values, y_dirty=y_dirty
        )

        # If y_dirty was not provided, fall back to the environment's y
        if y_result is None:
            y_result = y_from_env

        if verbose:
            matched = sum(1 for item in self._repair_plan
                          if (item['idx'], item['col']) in true_values)
            print(f"\n  Execution results:")
            print(f"    Ground truths matched: {matched} / {len(self._repair_plan)}")
            print(f"    Rows deleted: {int((~keep_mask).sum())}")
            print(f"    Final row count: {int(keep_mask.sum())}")

        return X_result, y_result, keep_mask

    def clean_with_reference(self,
                             X_dirty: np.ndarray,
                             y: np.ndarray,
                             X_clean: np.ndarray,
                             detected_errors: Dict[str, List],
                             verbose: bool = True,
                             y_clean: Optional[np.ndarray] = None,
                             save_csv_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[Dict]]:
        """
        Two-phase cleaning with reference data.

        Convenience method that pulls ground truths from X_clean / y_clean.

        Args:
            X_dirty: dirty data matrix
            y: label vector
            X_clean: clean data (used to fetch ground truths)
            detected_errors: detected errors
            verbose: whether to print details
            y_clean: clean label vector (used for label-noise repairs)
            save_csv_path: optional path to save the repair-plan CSV

        Returns:
            (X_clean_result, y_clean_result, keep_mask, action_counts, repair_plan)
        """
        # Phase 1
        repair_plan = self.plan(X_dirty, y, detected_errors, verbose,
                                save_csv_path=save_csv_path)

        # Pull ground truths from X_clean / y_clean
        true_values = {}
        for item in repair_plan:
            idx, col = item['idx'], item['col']
            if col == -1:
                # Label noise: use y_clean
                if y_clean is not None and idx < len(y_clean):
                    true_values[(idx, col)] = y_clean[idx]
            else:
                true_values[(idx, col)] = X_clean[idx, col]

        # Phase 2
        X_result, y_result, keep_mask = self.execute(
            X_dirty, true_values, verbose, y_dirty=y
        )

        action_counts = self._env.get_action_counts() if self._env else {}

        return X_result, y_result, keep_mask, action_counts, repair_plan

    def _build_error_list_no_truth(self,
                                   detected_errors: Dict[str, List]) -> List[Dict]:
        """Convert detected errors into the environment format (no ground truth)."""
        error_list = []

        # Missing errors (type=0)
        for item in detected_errors.get('missing', []):
            idx, col = item[0], item[1]
            estimated_val = item[2] if len(item) > 2 else 0
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': None  # no ground truth needed
            })

        # Semantic errors (type=1)
        for item in detected_errors.get('semantic', []):
            idx, col = item[0], item[1]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': None
            })

        # Syntactic errors (type=2)
        for item in detected_errors.get('syntactic', []):
            idx, col = item[0], item[1]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': None
            })

        # Label noise errors (type=3, col=-1)
        for item in detected_errors.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': None
                })

        return error_list

    def get_stats(self) -> Dict[str, Any]:
        """Return inference stats."""
        stats = {
            'agent_type': self.config.agent_type.value,
            'task_type': self.config.task_type.value,
            'model_type': self.config.model_type.value,
            'plan_size': len(self._repair_plan)
        }

        if self._env:
            stats['action_counts'] = self._env.get_action_counts()

        return stats

    def get_decision_log(self) -> List[Dict]:
        """Return the full decision log produced during inference."""
        if self._env is None:
            return []
        return self._env.get_decision_log()
