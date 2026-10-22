"""
DemandClean high-level API
==========================

Concise interface for demand-driven data cleaning.
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Set
import os
import numpy as np

from ..config import (
    DemandCleanConfig, TaskType, ModelType, AgentType,
    DetectorMode, InferenceMode,
)
from ..core.agents import (
    BaseAgent,
    SingleStageDQNAgent, TwoStageDQNAgent,
    DuelingSingleStageAgent, DuelingTwoStageAgent,
)
from ..training import Trainer
from ..inference import SinglePhaseInference, TwoPhaseInference
from ..detectors import (
    ErrorInjector, AutoDetector, RahaBasedDetector, OracleDetector,
    parse_rules_file, load_rules, extract_fd_pairs, rules_to_dict,
)
from ..utils.model_io import ModelIO
from ..utils.logger import DemandCleanLogger


class DemandClean:
    """
    DemandClean data cleaning system.

    Example:
    ```python
    from demandclean import DemandClean

    # Create an instance
    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        max_truth_budget=50
    )

    # Train (does not require clean data)
    dc.fit(X_dirty, y, semantic_errors=[(10, 1), (25, 1)])

    # Single-phase inference
    X_clean, y_clean, stats = dc.clean(X_dirty, y, X_clean_ref)

    # Two-phase inference
    plan = dc.plan(X_dirty, y)
    X_clean = dc.execute(X_dirty, true_values)
    ```
    """

    def __init__(self,
                 task_type: Optional[str] = None,
                 model_type: Optional[str] = None,
                 agent_type: Optional[str] = None,
                 detector_mode: Optional[str] = None,
                 inference_mode: Optional[str] = None,
                 training_mode: Optional[str] = None,
                 n_episodes: Optional[int] = None,
                 repair_lambda: Optional[float] = None,
                 min_truth_budget: Optional[int] = None,
                 max_truth_budget: Optional[int] = None,
                 rules_path: Optional[str] = None,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 column_names: Optional[List[str]] = None,
                 dirty_csv_path: Optional[str] = None,
                 clean_csv_path: Optional[str] = None,
                 csv_columns: Optional[List[str]] = None,
                 label_col: Optional[str] = None,
                 save_path: Optional[str] = None,
                 apply_raha_truth: Optional[bool] = None,
                 count_raha_cost: Optional[bool] = None,
                 **kwargs):
        """
        Initialize DemandClean.

        Args:
            task_type: task type ('classification' or 'regression')
            model_type: model type ('svm', 'random_forest', 'xgboost', 'linear', 'ridge')
            agent_type: agent type ('single', 'two_stage', 'dueling_single', 'dueling_two_stage')
            detector_mode: detector mode ('auto' or 'oracle')
            inference_mode: inference mode ('single_phase' or 'two_phase')
            training_mode: training mode ('clean_base' or 'self_supervised')
            n_episodes: number of training episodes
            repair_lambda: repair cost coefficient
            min_truth_budget: minimum number of ground-truth values required
            max_truth_budget: maximum number of ground-truth values allowed
            rules_path: path to FD rules file
            fd_rules: parsed list of FD rules
            column_names: feature column names (excluding index/label)
            dirty_csv_path: path to raw dirty CSV (required by RAHA in auto mode)
            clean_csv_path: path to raw clean CSV (required by RAHA in auto mode)
            csv_columns: all column names of the raw CSV (incl. index/label, for RAHA column mapping)
            label_col: label column name
            save_path: output save path
            apply_raha_truth: whether to apply RAHA-labeled ground-truth to the data for repair
            count_raha_cost: whether to include RAHA labeling cost in the total truth cost
            **kwargs: other config parameters
        """
        # ============================================================
        # Build config_kwargs: only pass non-None parameters.
        # All default values are centrally defined in DemandCleanConfig (config.py).
        # ============================================================

        # Enum mapping tables
        _task_type_map = {
            'classification': TaskType.CLASSIFICATION,
            'regression': TaskType.REGRESSION,
            'clustering': TaskType.CLUSTERING,
        }
        _model_type_map = {
            'svm': ModelType.SVM,
            'random_forest': ModelType.RANDOM_FOREST,
            'xgboost': ModelType.XGBOOST,
            'linear': ModelType.LINEAR,
            'ridge': ModelType.RIDGE,
            'xgboost_reg': ModelType.XGBOOST_REG,
            'kmeans': ModelType.KMEANS,
        }
        _agent_type_map = {
            'single_stage': AgentType.SINGLE_STAGE,
            'single': AgentType.SINGLE_STAGE,
            'two_stage': AgentType.TWO_STAGE,
            'dueling_single': AgentType.DUELING_SINGLE_STAGE,
            'dueling_single_stage': AgentType.DUELING_SINGLE_STAGE,
            'dueling_two_stage': AgentType.DUELING_TWO_STAGE,
        }
        _detector_mode_map = {
            'auto': DetectorMode.AUTO,
            'oracle': DetectorMode.ORACLE,
        }
        _inference_mode_map = {
            'single_phase': InferenceMode.SINGLE_PHASE,
            'two_phase': InferenceMode.TWO_PHASE,
        }

        config_kwargs: Dict[str, Any] = {}

        # Enum-typed parameters: convert and add only when non-None
        if task_type is not None:
            config_kwargs['task_type'] = _task_type_map.get(
                task_type.lower(), TaskType.CLASSIFICATION)
        if model_type is not None:
            config_kwargs['model_type'] = _model_type_map.get(
                model_type.lower(), ModelType.SVM)
        if agent_type is not None:
            config_kwargs['agent_type'] = _agent_type_map.get(
                agent_type.lower(), AgentType.SINGLE_STAGE)
        if detector_mode is not None:
            config_kwargs['detector_mode'] = _detector_mode_map.get(
                detector_mode.lower(), DetectorMode.AUTO)
        if inference_mode is not None:
            config_kwargs['inference_mode'] = _inference_mode_map.get(
                inference_mode.lower(), InferenceMode.SINGLE_PHASE)

        # Scalar parameters: add only when non-None
        _optional_params = {
            'training_mode': training_mode,
            'n_episodes': n_episodes,
            'repair_lambda': repair_lambda,
            'min_truth_budget': min_truth_budget,
            'max_truth_budget': max_truth_budget,
            'rules_path': rules_path,
            'fd_rules': fd_rules,
            'column_names': column_names,
            'label_col': label_col,
            'save_path': save_path,
            'apply_raha_truth': apply_raha_truth,
            'count_raha_cost': count_raha_cost,
        }
        for key, val in _optional_params.items():
            if val is not None:
                config_kwargs[key] = val

        # Encoding helpers (passed via **kwargs; added only when non-None)
        _encoding_params = {
            'label_encoders': kwargs.pop('encoding_label_encoders', None),
            'scaler': kwargs.pop('encoding_scaler', None),
            'categorical_cols': kwargs.pop('encoding_categorical_cols', None),
            'dirty_df': kwargs.pop('encoding_dirty_df', None),
            'clean_df': kwargs.pop('encoding_clean_df', None),
        }
        for key, val in _encoding_params.items():
            if val is not None:
                config_kwargs[key] = val

        # Extract AutoDetector-only parameters (not passed to DemandCleanConfig)
        self._disable_raha = kwargs.pop('disable_raha', False)

        # Remaining kwargs are forwarded directly (backward compatibility)
        config_kwargs.update(kwargs)

        self.config = DemandCleanConfig(**config_kwargs)

        # Store CSV path parameters (used by RAHA detection in auto mode)
        self.dirty_csv_path = dirty_csv_path
        self.clean_csv_path = clean_csv_path
        self.csv_columns = csv_columns
        self.label_col = label_col

        # Parse FD rules if rules_path is provided and fd_rules is not given directly
        if rules_path and not fd_rules:
            self._parse_rules(rules_path)

        # Components
        self.trainer = Trainer(self.config)
        self.detector: Optional[Union[AutoDetector, OracleDetector]] = None
        self.agent: Optional[BaseAgent] = None
        self.logger = DemandCleanLogger(self.config)
        self.model_io = ModelIO()

        # Inference engines (lazy-initialized)
        self._single_phase_inference: Optional[SinglePhaseInference] = None
        self._two_phase_inference: Optional[TwoPhaseInference] = None

        # State
        self._is_fitted = False

        # Detection cache: avoid redetecting the same data across fit() and clean()/plan()
        self._detected_cache: Optional[Dict[str, List]] = None
        self._detected_cache_fingerprint: Optional[int] = None

    def _parse_rules(self, rules_path: str):
        """Parse FD rules and rich rules from the rules file."""
        try:
            parsed = load_rules(rules_path)
            # Extract FD pairs
            fd_pairs = extract_fd_pairs(parsed)
            if fd_pairs:
                self.config.fd_rules = fd_pairs
            # Rich rules (DOMAIN/CFD/DC)
            rich_dict = rules_to_dict(parsed)
            if rich_dict.get('has_rich_rules'):
                self.config.rich_rules = rich_dict
        except Exception as e:
            print(f"  [Warning] Failed to parse rules file: {e}")

    def fit(self,
            X_dirty: np.ndarray,
            y: np.ndarray,
            X_clean: Optional[np.ndarray] = None,
            y_clean: Optional[np.ndarray] = None,
            semantic_errors: Optional[List[Tuple[int, int]]] = None,
            n_episodes: Optional[int] = None,
            verbose: bool = True,
            resume_from: Optional[str] = None,
            prev_history: Optional[Dict[str, List]] = None,
            X_clean_val: Optional[np.ndarray] = None,
            y_clean_val: Optional[np.ndarray] = None,
            ) -> 'DemandClean':
        """
        Train the model.

        Supports pre-repairing dirty data using the RAHA-labeled rows (labeling_budget rows):
        when X_clean/y_clean are provided and the detector is in Auto mode, RAHA is run first
        to obtain labeled_tuples, and the corresponding rows in the dirty data are replaced
        with clean values to improve training data quality.

        Args:
            X_dirty: dirty feature matrix
            y: label vector
            X_clean: clean data (optional, used for pre-repair of RAHA-labeled rows)
            y_clean: clean labels (optional, used for pre-repair of RAHA-labeled row labels)
            semantic_errors: list of semantic error positions [(row, col), ...]
            n_episodes: number of training episodes (defaults to config)
            verbose: whether to print detailed information
            resume_from: path to a model for resumed training (None = train from scratch)
            prev_history: previous training history (concatenated when resuming)
            X_clean_val: clean validation features (Oracle mode reward signal)
            y_clean_val: clean validation labels (Oracle mode reward signal)

        Returns:
            self
        """
        if verbose:
            self.logger.log_info("=" * 50)
            self.logger.log_info("DemandClean training started")
            self.logger.log_info(f"  Detector mode: {self.config.detector_mode.value}")
            self.logger.log_info(f"  Training mode: {self.config.training_mode}")
            self.logger.log_info(f"  Agent type: {self.config.agent_type.value}")
            self.logger.log_info(f"  Inference mode: {self.config.inference_mode.value}")
            self.logger.log_info(f"  FD rules: {len(self.config.fd_rules or [])}")
            self.logger.log_info(f"  Rich rules: {'yes' if self.config.rich_rules else 'no'}")
            self.logger.log_info("=" * 50)

        # 1. Create the detector
        if verbose:
            self.logger.log_info("\n[Step 1] Initializing error detector...")

        if self.config.is_oracle:
            self.detector = OracleDetector(
                column_names=self.config.column_names
            )
            self.detector.fit(verbose=verbose)
        else:
            self.detector = self._build_auto_detector(X_dirty, verbose=verbose)

        # 2. Detect errors (training needs detected_errors for self_supervised mode)
        detected_errors = None

        if isinstance(self.detector, AutoDetector) and X_clean is not None:
            if not self.detector.labeled_tuples:
                # Auto mode + X_clean provided: run RAHA to obtain labeled_tuples
                if verbose:
                    self.logger.log_info("\n[Step 1.5] Running RAHA detection to collect labeled rows...")
                task_type_str = self.config.task_type.value
                detected_errors = self.detector.detect(
                    X_dirty, y_dirty=y, task_type=task_type_str,
                    semantic_positions=semantic_errors, verbose=verbose
                )
                if detected_errors:
                    for key in ['missing', 'semantic', 'syntactic', 'label_noise']:
                        if key not in detected_errors:
                            detected_errors[key] = []
                    # Cache detection results for later clean()/plan() reuse
                    self._detected_cache = detected_errors
                    self._detected_cache_fingerprint = self._data_fingerprint(X_dirty, y)

            # Pre-repair RAHA-labeled rows (controlled by apply_raha_truth)
            if self.config.apply_raha_truth:
                X_dirty, y = self._prefix_labeled_rows(
                    X_dirty, y, X_clean, y_clean,
                    detected_errors=detected_errors, verbose=verbose
                )
            elif verbose:
                self.logger.log_info(
                    "  [Skip pre-repair] apply_raha_truth=False, RAHA labels used only for detection"
                )

        # In self_supervised mode, every detector needs detected_errors
        if self.config.training_mode == 'self_supervised' and detected_errors is None:
            if verbose:
                self.logger.log_info("\n[Step 1.5] self_supervised mode: running detector to obtain error distribution...")
            detected_errors = self.detect_errors(
                X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
                semantic_errors=semantic_errors, verbose=verbose
            )

        # 3. Resume training: load existing model (compatible with two-stage model file naming)
        start_episode = 0
        resume_agent = None
        if resume_from and self.model_io.agent_model_exists(resume_from):
            if verbose:
                self.logger.log_info(f"\n[Step 1.8] Loading model to resume training: {resume_from}")
            # Create an agent matching the config via Trainer, then load weights
            resume_agent = self.trainer._create_agent()
            resume_agent.load(resume_from)
            start_episode = resume_agent.total_episodes
            if verbose:
                self.logger.log_info(
                    f"  Already trained {start_episode} episodes, "
                    f"best_score={resume_agent.best_score:.4f}, "
                    f"epsilon={resume_agent.epsilon:.4f}"
                )

        # 4. Train the DQN agent
        if verbose:
            self.logger.log_info("\n[Step 2] Training DQN agent...")

        self.agent, history = self.trainer.train(
            X_dirty, y,
            n_episodes=n_episodes,
            verbose=verbose,
            detected_errors=detected_errors,
            start_episode=start_episode,
            prev_history=prev_history,
            agent=resume_agent,
            X_clean_val=X_clean_val,
            y_clean_val=y_clean_val,
        )

        self._is_fitted = True

        if verbose:
            self.logger.log_info("\nTraining finished.")

        return self

    def clean(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              X_clean: np.ndarray,
              y_clean: Optional[np.ndarray] = None,
              semantic_errors: Optional[List[Tuple[int, int]]] = None,
              pre_detected: Optional[Dict[str, List]] = None,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Single-phase inference: directly clean the data.

        Args:
            X_dirty: dirty feature matrix
            y: label vector (dirty labels)
            X_clean: clean data (used to obtain ground-truth repairs)
            y_clean: clean label vector (used for label-noise detection and repair, optional)
            semantic_errors: list of semantic error positions [(row, col), ...]
            pre_detected: externally provided detection results, bypassing the internal detector
                (used in ablation studies for a fair comparison with baselines)
            verbose: whether to print detailed information

        Returns:
            (X_clean_result, y_clean_result, stats)
        """
        self._check_fitted()

        if verbose:
            self.logger.log_info("\n" + "=" * 50)
            self.logger.log_info("Single-phase inference")
            self.logger.log_info("=" * 50)

        if pre_detected is not None:
            # Use externally provided detection results (bypass OracleDetector)
            detected = pre_detected
        else:
            # Detect errors (including label noise)
            detected = self.detect_errors(
                X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
                semantic_errors=semantic_errors, verbose=verbose
            )

        # Pre-repair RAHA-labeled rows and drop fixed cells from `detected` (gated by switch)
        if self.config.apply_raha_truth:
            X_dirty, y, detected = self._prefix_labeled_rows_for_inference(
                X_dirty, y, X_clean, y_clean, detected, verbose=verbose
            )

        # Create inference engine
        if self._single_phase_inference is None:
            self._single_phase_inference = SinglePhaseInference(self.agent, self.config)

        # Run cleaning
        X_result, y_result, keep_mask, action_counts, repair_log = \
            self._single_phase_inference.clean(
                X_dirty, y, X_clean, detected, verbose, y_clean=y_clean
            )

        stats = {
            'action_counts': action_counts,
            'repair_log': repair_log,
            'keep_mask': keep_mask,
            'truth_cost': action_counts['repair_value'],
            'deleted_count': action_counts['delete']
        }

        return X_result, y_result, stats

    def plan(self,
             X_dirty: np.ndarray,
             y: np.ndarray,
             X_clean: Optional[np.ndarray] = None,
             y_clean: Optional[np.ndarray] = None,
             semantic_errors: Optional[List[Tuple[int, int]]] = None,
             verbose: bool = True) -> List[Dict]:
        """
        Two-phase inference - Phase 1: generate the repair plan.

        Does not require ground truth; returns the list of positions to repair.
        X_clean must be provided in Oracle mode for error detection.

        Args:
            X_dirty: dirty feature matrix
            y: label vector (dirty labels)
            X_clean: clean data (required in Oracle mode)
            y_clean: clean label vector (used for label-noise detection, optional)
            semantic_errors: list of semantic error positions
            verbose: whether to print detailed information

        Returns:
            repair_plan: list of positions that require ground-truth repair
        """
        self._check_fitted()

        if verbose:
            self.logger.log_info("\n" + "=" * 50)
            self.logger.log_info("Two-phase inference - Phase 1 (Plan)")
            self.logger.log_info("=" * 50)

        # Detect errors (incl. label noise; Oracle mode requires X_clean / y_clean)
        detected = self.detect_errors(
            X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
            semantic_errors=semantic_errors, verbose=verbose
        )

        # Create inference engine
        if self._two_phase_inference is None:
            self._two_phase_inference = TwoPhaseInference(self.agent, self.config)

        # Generate plan
        repair_plan = self._two_phase_inference.plan(X_dirty, y, detected, verbose)

        return repair_plan

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """Return the list of positions that require ground-truth values."""
        if self._two_phase_inference is None:
            return []
        return self._two_phase_inference.get_plan_positions()

    def get_action_counts(self) -> Dict[str, int]:
        """
        Return the action distribution after two-phase inference.

        Single-phase inference already exposes it via stats['action_counts']
        from clean(); execute() only returns (X, y, keep_mask), so this
        accessor fills that gap.

        Returns:
            {'no_action': int, 'repair_value': int, 'delete': int,
             'replace_nearby': int}; empty dict if two-phase inference has
            not run yet.
        """
        if self._two_phase_inference is None:
            return {}
        return self._two_phase_inference.get_stats().get('action_counts', {})

    def execute(self,
                X_dirty: np.ndarray,
                true_values: Dict[Tuple[int, int], float],
                verbose: bool = True,
                y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-phase inference - Phase 2: execute the repairs.

        Args:
            X_dirty: original dirty data
            true_values: ground-truth value dict {(idx, col): value}
            verbose: whether to print detailed information
            y_dirty: original dirty labels (required when repairing labels)

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        if self._two_phase_inference is None:
            raise ValueError("Call plan() first to generate the repair plan")

        return self._two_phase_inference.execute(X_dirty, true_values, verbose, y_dirty=y_dirty)

    @staticmethod
    def _data_fingerprint(X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Compute a data fingerprint used to check detection-cache hits.

        Based on a hash of the shape plus head/tail/middle row samples. O(1) with
        negligible collision probability.
        """
        parts = [X.shape]
        # Head / tail rows
        if len(X) > 0:
            parts.append(X[0].tobytes())
            parts.append(X[-1].tobytes())
        if len(X) > 2:
            parts.append(X[len(X) // 2].tobytes())
        if y is not None:
            parts.append(y.shape)
            if len(y) > 0:
                parts.append(y[0])
                parts.append(y[-1])
        return hash(tuple(str(p) for p in parts))

    def _build_auto_detector(self, X_dirty: np.ndarray, verbose: bool = True) -> AutoDetector:
        """Build and fit the hybrid error detector.

        The detector combines the rule library of the dataset, RAHA with a calibration seed
        set of 20 tuples, and semantic and statistical checks fitted on the dirty rows that
        have no missing value. Training and inference from a checkpoint use this detector.
        """
        detector = AutoDetector(
            dirty_csv_path=self.dirty_csv_path,
            clean_csv_path=self.clean_csv_path,
            dataset_name=os.path.basename(os.path.dirname(self.dirty_csv_path))
                if self.dirty_csv_path else "data",
            label_col=self.label_col,
            csv_columns=self.csv_columns,
            column_names=self.config.column_names,
            fd_rules=self.config.fd_rules,
            labeling_budget=20,
            rules_path=self.config.rules_path,
            disable_raha=self._disable_raha,
        )
        complete_rows = X_dirty[~np.isnan(X_dirty).any(axis=1)]
        if len(complete_rows) > 0:
            detector.fit(complete_rows, verbose=verbose)
        else:
            detector.fit(verbose=verbose)
        return detector

    def detect_errors(self,
                      X_dirty: np.ndarray,
                      X_clean: Optional[np.ndarray] = None,
                      y_dirty: Optional[np.ndarray] = None,
                      y_clean: Optional[np.ndarray] = None,
                      semantic_errors: Optional[List[Tuple[int, int]]] = None,
                      verbose: bool = True) -> Dict[str, List]:
        """
        Detect errors.

        In Oracle mode, directly compares dirty vs. clean (features + labels).
        In Auto mode, uses RAHA (on the raw CSV) + FD + Confident Learning.

        Args:
            X_dirty: dirty data
            X_clean: clean data (required in Oracle mode, optional in Auto mode)
            y_dirty: dirty label vector
            y_clean: clean label vector (used for label-noise detection in Oracle mode)
            semantic_errors: list of semantic error positions (used in Auto mode when no FD rules)
            verbose: whether to print detailed information

        Returns:
            detected: {'missing': [...], 'semantic': [...], 'syntactic': [...], 'label_noise': [...]}
        """
        # Cache-hit check: avoid redetecting the same data
        fingerprint = self._data_fingerprint(X_dirty, y_dirty)
        if (self._detected_cache is not None
                and self._detected_cache_fingerprint == fingerprint):
            if verbose:
                total = sum(len(v) for v in self._detected_cache.values())
                print(f"  [Detection cache hit] Reusing previous results ({total} cells)")
            # Return a shallow copy per list so callers (e.g. _prefix_labeled_rows_for_inference)
            # cannot mutate the cached entries.
            return {k: list(v) for k, v in self._detected_cache.items()}

        if self.detector is None:
            if self.config.is_oracle:
                self.detector = OracleDetector(column_names=self.config.column_names)
            else:
                # A policy loaded from a checkpoint: build the detector of a training run
                self.detector = self._build_auto_detector(X_dirty, verbose=verbose)

        if isinstance(self.detector, OracleDetector):
            if X_clean is None:
                raise ValueError("X_clean is required in Oracle mode")
            detected = self.detector.detect(
                X_dirty, X_clean,
                y_dirty=y_dirty, y_clean=y_clean,
                verbose=verbose
            )
        else:
            task_type_str = self.config.task_type.value if hasattr(self.config.task_type, 'value') else str(self.config.task_type)
            detected = self.detector.detect(
                X_dirty,
                y_dirty=y_dirty,
                task_type=task_type_str,
                semantic_positions=semantic_errors,
                verbose=verbose
            )

        # Ensure every error-type list exists
        for key in ['missing', 'semantic', 'syntactic', 'label_noise']:
            if key not in detected:
                detected[key] = []

        # Write to cache
        self._detected_cache = detected
        self._detected_cache_fingerprint = fingerprint

        return detected

    def save(self,
             model_path: str,
             detector_path: Optional[str] = None) -> None:
        """
        Save the model.

        Args:
            model_path: path to save the agent model
            detector_path: path to save the detector (optional)
        """
        self._check_fitted()

        # Save the agent
        self.model_io.save_agent(self.agent, model_path)

        # Save the detector
        if detector_path and self.detector:
            self.detector.save(detector_path)

        self.logger.log_info(f"Model saved: {model_path}")

    def load(self,
             model_path: str,
             detector_path: Optional[str] = None) -> 'DemandClean':
        """
        Load the model.

        Args:
            model_path: path to the agent model
            detector_path: path to the detector (optional)

        Returns:
            self
        """
        # Load the agent
        _AGENT_CLASS_MAP = {
            AgentType.SINGLE_STAGE: SingleStageDQNAgent,
            AgentType.TWO_STAGE: TwoStageDQNAgent,
            AgentType.DUELING_SINGLE_STAGE: DuelingSingleStageAgent,
            AgentType.DUELING_TWO_STAGE: DuelingTwoStageAgent,
        }
        agent_cls = _AGENT_CLASS_MAP.get(self.config.agent_type, SingleStageDQNAgent)
        self.agent = self.model_io.load_agent(agent_cls, model_path)
        self._adopt_checkpoint_state_size()

        # Load the detector
        if detector_path and os.path.exists(detector_path):
            self.detector = AutoDetector.load(detector_path)

        self._is_fitted = True
        self.logger.log_info(f"Model loaded: {model_path}")

        return self

    def _adopt_checkpoint_state_size(self) -> None:
        """Match the environment state layout to the loaded checkpoint.

        A checkpoint whose first layer takes LEGACY_STATE_SIZE (8) inputs was
        trained with the earlier state builder (the har policy of the reported
        run is one). Setting config.state_size to 8 makes the
        environments reproduce that state builder: no global features, the
        init model fit on all rows, and feature importance kept at its init
        value. The two-stage agent also switches to the earlier head-to-action
        layout.
        """
        from ..core.state.state_extractor import LEGACY_STATE_SIZE
        ckpt_size = int(getattr(self.agent, 'state_size', self.config.state_size))
        if ckpt_size == self.config.state_size:
            return
        if ckpt_size != LEGACY_STATE_SIZE:
            raise ValueError(
                f"checkpoint state_size={ckpt_size} matches neither the current "
                f"layout ({self.config.state_size}) nor the legacy layout ({LEGACY_STATE_SIZE})")
        self.config.state_size = ckpt_size
        if hasattr(self.agent, 'legacy_action_map'):
            self.agent.legacy_action_map = True
        msg = (f"[legacy state] checkpoint expects {ckpt_size} inputs: using the 8-dim "
               f"per-error state without global features"
               + (" and the earlier two-stage action map" if hasattr(self.agent, 'legacy_action_map') else ""))
        print(msg)
        self.logger.log_info(msg)

    def _get_labeled_tuples(self) -> Set[int]:
        """Return the set of row indices labeled by RAHA."""
        if isinstance(self.detector, AutoDetector) and self.detector.labeled_tuples:
            return self.detector.labeled_tuples
        return set()

    def _prefix_labeled_rows(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: Optional[np.ndarray],
        y_clean: Optional[np.ndarray],
        detected_errors: Optional[Dict[str, List]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-repair dirty data using clean values from RAHA-labeled rows (training phase).

        Only cells detected as errors within labeled rows are replaced, not the whole row.

        Returns:
            (X_dirty_prefixed, y_prefixed) , copies; the originals are not modified.
        """
        labeled_indices = self._get_labeled_tuples()
        if not labeled_indices or X_clean is None:
            return X_dirty, y

        # Extract all error cells (idx, col) from detected_errors
        error_cells: Set[Tuple[int, int]] = set()
        label_error_rows: Set[int] = set()
        if detected_errors:
            for key in ['missing', 'semantic', 'syntactic']:
                for item in detected_errors.get(key, []):
                    error_cells.add((item[0], item[1]))
            for item in detected_errors.get('label_noise', []):
                label_error_rows.add(item[0])

        X_out = X_dirty.copy()
        y_out = y.copy()
        fixed_feature_cells = 0
        fixed_label_cells = 0

        for idx in sorted(labeled_indices):
            if idx >= len(X_out) or idx >= len(X_clean):
                continue
            # Only replace feature cells in this row that are detected as errors
            for col in range(X_out.shape[1]):
                if (idx, col) in error_cells:
                    X_out[idx, col] = X_clean[idx, col]
                    fixed_feature_cells += 1
            # Label error: only replace rows detected as label noise
            if idx in label_error_rows and y_clean is not None and idx < len(y_clean):
                y_out[idx] = y_clean[idx]
                fixed_label_cells += 1

        total_fixed = fixed_feature_cells + fixed_label_cells
        if verbose and total_fixed > 0:
            self.logger.log_info(
                f"  [Pre-repair] Using RAHA-labeled rows, repaired {fixed_feature_cells} feature cells"
                f" + {fixed_label_cells} labels"
            )
            self.logger.log_info(
                f"  Labeled rows: {sorted(labeled_indices)[:10]}"
                + (f"... ({len(labeled_indices)} rows in total)" if len(labeled_indices) > 10 else "")
            )

        return X_out, y_out

    def _prefix_labeled_rows_for_inference(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: Optional[np.ndarray],
        y_clean: Optional[np.ndarray],
        detected: Dict[str, List],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """
        Pre-repair dirty data using clean values from RAHA-labeled rows (inference phase).
        Only cells detected as errors within labeled rows are replaced, and the repaired
        entries are removed from detected_errors.

        Returns:
            (X_dirty_prefixed, y_prefixed, detected_filtered)
        """
        labeled_indices = self._get_labeled_tuples()
        if not labeled_indices or X_clean is None:
            return X_dirty, y, detected

        # Extract all error cells (idx, col) and label-error rows from `detected`
        error_cells: Set[Tuple[int, int]] = set()
        label_error_rows: Set[int] = set()
        for key in ['missing', 'semantic', 'syntactic']:
            for item in detected.get(key, []):
                error_cells.add((item[0], item[1]))
        for item in detected.get('label_noise', []):
            label_error_rows.add(item[0])

        X_out = X_dirty.copy()
        y_out = y.copy()
        fixed_feature_cells: Set[Tuple[int, int]] = set()
        fixed_label_rows: Set[int] = set()

        for idx in sorted(labeled_indices):
            if idx >= len(X_out) or idx >= len(X_clean):
                continue
            # Only replace feature cells in this row that are detected as errors
            for col in range(X_out.shape[1]):
                if (idx, col) in error_cells:
                    X_out[idx, col] = X_clean[idx, col]
                    fixed_feature_cells.add((idx, col))
            # Label error
            if idx in label_error_rows and y_clean is not None and idx < len(y_clean):
                y_out[idx] = y_clean[idx]
                fixed_label_rows.add(idx)

        # Remove repaired cells from detected_errors
        removed_counts = {}
        detected_filtered = {}
        for key in ['missing', 'semantic', 'syntactic']:
            original = detected.get(key, [])
            filtered = [e for e in original if (e[0], e[1]) not in fixed_feature_cells]
            detected_filtered[key] = filtered
            removed = len(original) - len(filtered)
            if removed > 0:
                removed_counts[key] = removed
        # Label noise: remove repaired rows
        original_label = detected.get('label_noise', [])
        filtered_label = [e for e in original_label if e[0] not in fixed_label_rows]
        detected_filtered['label_noise'] = filtered_label
        removed_label = len(original_label) - len(filtered_label)
        if removed_label > 0:
            removed_counts['label_noise'] = removed_label

        total_fixed = len(fixed_feature_cells) + len(fixed_label_rows)
        if verbose and total_fixed > 0:
            self.logger.log_info(
                f"  [Pre-repair] Using RAHA-labeled rows, repaired {len(fixed_feature_cells)} feature cells"
                f" + {len(fixed_label_rows)} labels"
            )
            if removed_counts:
                parts = [f"{k}={v}" for k, v in removed_counts.items()]
                self.logger.log_info(
                    f"  Removed repaired errors: {', '.join(parts)}"
                )

        return X_out, y_out, detected_filtered

    def _check_fitted(self) -> None:
        """Check whether the model has been trained."""
        if not self._is_fitted or self.agent is None:
            raise ValueError("Model not trained or loaded; call fit() or load() first")

    def get_config(self) -> DemandCleanConfig:
        """Return the configuration."""
        return self.config

    def get_training_history(self) -> Dict[str, List]:
        """Return the training history."""
        return self.trainer.get_history()

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted
