"""
Trainer
=======

DQN agent trainer for two-stage and single-stage agents.

Training modes:
  - clean_base (default): a base table built with the DeleteFix strategy, errors injected for training
  - self_supervised: training on the full dirty table, detected errors keep their type, extra errors injected into undetected cells
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Set
import os
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import (
    BaseAgent, SingleStageDQNAgent, TwoStageDQNAgent,
    DuelingSingleStageAgent, DuelingTwoStageAgent,
)
from ..core.environments import CleaningEnv
from ..core.environments.value_estimation import ValueEstimator
from ..core.state import (
    StateExtractor, ClassificationStateExtractor,
    RegressionStateExtractor, ClusteringStateExtractor,
)
from ..models import ModelAdapter, create_model_adapter
from ..detectors import ErrorInjector
from ..detectors.error_injector import LabelErrorPattern, analyze_label_error_pattern
from ..utils.logger import DemandCleanLogger
from ..utils.model_io import ModelIO


class Trainer:
    """
    DQN agent trainer.

    Two training modes:
    - clean_base: learn the policy on the NaN-free dirty table with self-injected errors
    - self_supervised: train on the full dirty table with detected and newly injected errors
    """

    def __init__(self, config: DemandCleanConfig):
        """
        Initialise the trainer.

        Args:
            config: configuration object
        """
        self.config = config
        self.logger = DemandCleanLogger(config)
        self.model_io = ModelIO()

        # Model adapter
        model_kwargs = config.model_kwargs or {}
        self.model_adapter = create_model_adapter(config.model_type, config.task_type, **model_kwargs)

        # State extractor
        self.state_extractor = self._create_state_extractor()

        # Agent
        self.agent: Optional[BaseAgent] = None

        # Training history
        self.history: Dict[str, List] = {}

    def _create_state_extractor(self) -> StateExtractor:
        """Create the state extractor."""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def _create_agent(self) -> BaseAgent:
        """Create the agent."""
        at = self.config.agent_type
        common = dict(
            state_size=self.config.state_size,
            memory_size=self.config.memory_size,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            epsilon_min=self.config.epsilon_min,
            epsilon_decay=self.config.epsilon_decay,
            learning_rate=self.config.learning_rate,
        )

        if at == AgentType.DUELING_TWO_STAGE:
            return DuelingTwoStageAgent(**common)
        elif at == AgentType.DUELING_SINGLE_STAGE:
            return DuelingSingleStageAgent(action_size=4, **common)
        elif at == AgentType.TWO_STAGE:
            return TwoStageDQNAgent(**common)
        else:
            return SingleStageDQNAgent(action_size=4, **common)

    # ====================================================================
    # Public interface
    # ====================================================================

    def train(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              n_episodes: Optional[int] = None,
              verbose: bool = True,
              detected_errors: Optional[Dict[str, List]] = None,
              start_episode: int = 0,
              prev_history: Optional[Dict[str, List]] = None,
              agent: Optional[BaseAgent] = None,
              X_clean_val: Optional[np.ndarray] = None,
              y_clean_val: Optional[np.ndarray] = None,
              ) -> Tuple[BaseAgent, Dict[str, List]]:
        """
        Train the DQN agent.

        Dispatches on config.training_mode:
        - "clean_base": self-injected errors on the NaN-free dirty table
        - "self_supervised": detected errors plus newly injected errors on the full dirty table

        Args:
            X_dirty: dirty data matrix
            y: label vector
            n_episodes: number of training episodes (configuration value by default)
            verbose: print details
            detected_errors: detector output (needed by self_supervised)
                {'missing': [...], 'semantic': [...], 'syntactic': [...], 'label_noise': [...]}
            start_episode: first episode when resuming (0 trains from scratch)
            prev_history: earlier training history (concatenated when resuming)
            agent: existing agent passed in when resuming
            X_clean_val: clean validation features (reward signal with a clean validation split)
            y_clean_val: clean validation labels (reward signal with a clean validation split)

        Returns:
            (agent, history)
        """
        if (self.config.training_mode == 'self_supervised'
                and detected_errors is not None):
            return self._train_self_supervised(
                X_dirty, y, n_episodes, verbose, detected_errors,
                start_episode=start_episode,
                prev_history=prev_history,
                agent=agent)
        else:
            return self._train_clean_base(
                X_dirty, y, n_episodes, verbose, detected_errors,
                start_episode=start_episode,
                prev_history=prev_history,
                agent=agent,
                X_clean_val=X_clean_val,
                y_clean_val=y_clean_val)

    # ====================================================================
    # Training mode 1: clean_base (default)
    # ====================================================================

    def _train_clean_base(self,
                          X_dirty: np.ndarray,
                          y: np.ndarray,
                          n_episodes: Optional[int] = None,
                          verbose: bool = True,
                          detected_errors: Optional[Dict[str, List]] = None,
                          start_episode: int = 0,
                          prev_history: Optional[Dict[str, List]] = None,
                          agent: Optional[BaseAgent] = None,
                          X_clean_val: Optional[np.ndarray] = None,
                          y_clean_val: Optional[np.ndarray] = None,
                          ) -> Tuple[BaseAgent, Dict[str, List]]:
        """clean_base training mode.

        X_base = DeleteFix(dirty, detected_errors). Each episode injects errors and uses the
        pre-injection values as pseudo ground truth.

        With a clean validation split (use_clean_validation=True):
          - X_base_inject is still built from the dirty data (used for error injection)
          - CleaningEnv.X_base uses X_clean_val (the clean validation split gives the reward)
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            mode_label = "Resuming training" if start_episode > 0 else "Starting training"
            self.logger.log_info(f"{mode_label} ({self.config.agent_type.value} Agent, mode=clean_base)")
            self.logger.log_info(f"  Task type: {self.config.task_type.value}")
            self.logger.log_info(f"  Model type: {self.config.model_type.value}")
            self.logger.log_info(f"  Input data: {len(X_dirty)} rows")
            if start_episode > 0:
                self.logger.log_info(f"  Resume from episode: {start_episode}")

        # 1. Base data (DeleteFix or VE-Fill chosen automatically)
        if getattr(self.config, 'auto_select_base', True) and detected_errors is not None:
            X_base, y_base, strategy = self._select_best_base(
                X_dirty, y, detected_errors, verbose)
        else:
            X_base, y_base = self._build_base_data(
                X_dirty, y, detected_errors=detected_errors,
                fill_nan=True, verbose=verbose)

        # 2. Label error pattern (conditional label injection)
        label_pattern, label_rate_range = self._analyze_label_injection(
            detected_errors, y, verbose)

        # 3. Adaptive injection rate ranges (from the detection result)
        adaptive_ranges = self._compute_adaptive_rate_ranges(
            detected_errors, X_dirty, y, verbose)

        # 4. Agent (an agent passed in is reused when resuming)
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent()

        error_injector = ErrorInjector(
            X_base, y_base,
            fd_rules=self.config.fd_rules,
            column_names=self.config.column_names,
            rich_rules=self.config.rich_rules,
            label_encoders=self.config.label_encoders,
            scaler=self.config.scaler,
            categorical_cols=self.config.categorical_cols,
            dirty_df=self.config.dirty_df,
            label_col=self.config.label_col,
        )

        if verbose:
            stats = error_injector.get_stats()
            self.logger.log_info(
                f"  ErrorInjector: rich_rules={stats['has_rich_rules']}, "
                f"domain={stats['n_domain_rules']}, cfd={stats['n_cfd_rules']}, "
                f"fd={stats['n_fd_rules']}")

        # Clean validation split: target of the reward evaluation
        if self.config.use_clean_validation and X_clean_val is not None:
            X_eval_target, y_eval_target = X_clean_val, y_clean_val
            if verbose:
                self.logger.log_info(
                    f"  Clean validation split: reward uses the clean validation rows ({len(X_clean_val)} rows)")
        else:
            X_eval_target, y_eval_target = X_base, y_base

        # 5. Training loop
        return self._run_training_loop(
            error_injector, n_episodes, verbose,
            label_pattern, label_rate_range,
            inject_mode='clean_base',
            adaptive_ranges=adaptive_ranges,
            start_episode=start_episode,
            prev_history=prev_history,
            X_base=X_eval_target, y_base=y_eval_target,
        )

    # ====================================================================
    # Training mode 2: self_supervised (full dirty table)
    # ====================================================================

    def _train_self_supervised(self,
                               X_dirty: np.ndarray,
                               y: np.ndarray,
                               n_episodes: Optional[int] = None,
                               verbose: bool = True,
                               detected_errors: Optional[Dict[str, List]] = None,
                               start_episode: int = 0,
                               prev_history: Optional[Dict[str, List]] = None,
                               agent: Optional[BaseAgent] = None,
                               ) -> Tuple[BaseAgent, Dict[str, List]]:
        """self_supervised training mode.

          - errors in detected_errors form a fixed error_list (repair_value from column means or KNN)
          - label errors get repair_value=None and favour delete during training
          - undetected cells receive newly injected errors with the pre-injection value as pseudo ground truth
          - each episode merges the fixed list and the newly injected list into one CleaningEnv
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            mode_label = "Resuming training" if start_episode > 0 else "Starting training"
            self.logger.log_info(f"{mode_label} ({self.config.agent_type.value} Agent, mode=self_supervised)")
            self.logger.log_info(f"  Task type: {self.config.task_type.value}")
            self.logger.log_info(f"  Model type: {self.config.model_type.value}")
            self.logger.log_info(f"  Input data: {len(X_dirty)} rows")
            if start_episode > 0:
                self.logger.log_info(f"  Resume from episode: {start_episode}")

        # 1. Fixed list of detected errors
        fixed_error_list, detected_cells = self._build_detected_error_list(
            detected_errors, X_dirty, y, verbose)

        # 2. Label error pattern
        label_pattern, label_rate_range = self._analyze_label_injection(
            detected_errors, y, verbose)

        # 3. Adaptive injection rate ranges (from the detection result)
        adaptive_ranges = self._compute_adaptive_rate_ranges(
            detected_errors, X_dirty, y, verbose)

        # 4. Agent (an agent passed in is reused when resuming)
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent()

        # 5. ErrorInjector statistics from the DeleteFix base
        X_base_for_stats, y_base_for_stats = self._build_base_data(
            X_dirty, y, detected_errors=detected_errors,
            fill_nan=True, verbose=verbose)

        error_injector = ErrorInjector(
            X_base_for_stats, y_base_for_stats,
            fd_rules=self.config.fd_rules,
            column_names=self.config.column_names,
            rich_rules=self.config.rich_rules,
            # Encoders (re-encode after injection in CSV space)
            label_encoders=self.config.label_encoders,
            scaler=self.config.scaler,
            categorical_cols=self.config.categorical_cols,
            dirty_df=self.config.dirty_df,
            label_col=self.config.label_col,
        )

        if verbose:
            self.logger.log_info(
                f"  Fixed error list: {len(fixed_error_list)} errors, "
                f"detected cells: {len(detected_cells)}")

        # 6. Training loop
        return self._run_training_loop(
            error_injector, n_episodes, verbose,
            label_pattern, label_rate_range,
            inject_mode='self_supervised',
            X_dirty_full=X_dirty,
            y_dirty_full=y,
            fixed_error_list=fixed_error_list,
            detected_cells=detected_cells,
            adaptive_ranges=adaptive_ranges,
            start_episode=start_episode,
            prev_history=prev_history,
            X_base=X_base_for_stats, y_base=y_base_for_stats,
        )

    # ====================================================================
    # Shared training loop
    # ====================================================================

    def _run_training_loop(self,
                           error_injector: ErrorInjector,
                           n_episodes: int,
                           verbose: bool,
                           label_pattern: Optional[LabelErrorPattern],
                           label_rate_range: Tuple[float, float],
                           inject_mode: str = 'clean_base',
                           X_dirty_full: Optional[np.ndarray] = None,
                           y_dirty_full: Optional[np.ndarray] = None,
                           fixed_error_list: Optional[List[Dict]] = None,
                           detected_cells: Optional[Set[Tuple[int, int]]] = None,
                           adaptive_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                           start_episode: int = 0,
                           prev_history: Optional[Dict[str, List]] = None,
                           X_base: Optional[np.ndarray] = None,
                           y_base: Optional[np.ndarray] = None,
                           ) -> Tuple[BaseAgent, Dict[str, List]]:
        """Shared training loop (resume support and uncertainty logging)."""

        # Training history
        history_keys = [
            'episode', 'score', 'reward', 'epsilon',
            'no_action', 'repair_value', 'delete', 'replace_nearby',
            'q_std', 'action_probs',
        ]
        self.history = {k: [] for k in history_keys}

        # Resume: append to the earlier history
        if prev_history:
            for k in self.history:
                self.history[k] = list(prev_history.get(k, []))

        # Best-model tracking
        # When resuming, the old best_score is not restored. It was computed on the encoding, data
        # and injection of the earlier run and is not comparable, so new episodes could never exceed it,
        # best_weights would stay None and the best model would never be restored.
        best_score = -float('inf') if self.config.task_type == TaskType.REGRESSION else 0
        best_episode = start_episode
        best_cost = float('inf')
        best_actions = None
        best_weights = None

        if verbose:
            self.logger.log_info(f"  Training DQN ({n_episodes} episodes, first episode={start_episode})...")

        end_episode = start_episode + n_episodes

        # Shared ValueEstimator (indices are not rebuilt in every episode)
        shared_ve = ValueEstimator(self.config)

        for episode in range(start_episode, end_episode):
            # Adaptive error rates (from the detection result, undetected types are not injected)
            if adaptive_ranges is not None:
                missing_rate = np.random.uniform(*adaptive_ranges['missing'])
                semantic_rate = np.random.uniform(*adaptive_ranges['semantic'])
                syntactic_rate = np.random.uniform(*adaptive_ranges['syntactic'])
                label_rate = np.random.uniform(*adaptive_ranges['label'])
            else:
                # Fallback: fixed configuration ranges when there is no detection result
                missing_rate = np.random.uniform(*self.config.missing_rate_range)
                semantic_rate = np.random.uniform(*self.config.semantic_rate_range)
                syntactic_rate = np.random.uniform(*self.config.syntactic_rate_range)
                label_rate = np.random.uniform(*label_rate_range)

            if inject_mode == 'self_supervised' and X_dirty_full is not None:
                # Self-supervised: extra errors in undetected cells of the dirty table
                X_injected, y_injected, new_injected = error_injector.inject_on_dirty(
                    X_dirty_full, y_dirty_full, detected_cells or set(),
                    missing_rate, semantic_rate, syntactic_rate,
                    label_rate, label_pattern,
                )
                # Merge the fixed error list and the newly injected list
                new_error_list = error_injector.build_error_list(new_injected)
                error_list = (fixed_error_list or []) + new_error_list
            else:
                # clean_base: inject into the NaN-free base data
                X_injected, y_injected, injected = error_injector.inject_errors(
                    missing_rate, semantic_rate, syntactic_rate,
                    label_rate, label_pattern,
                )
                error_list = error_injector.build_error_list(injected)

            # Shaping decay factor: 1.0 during warmup, then linear decay to min_weight
            warmup_ratio = self.config.shaping_warmup_ratio
            min_weight = self.config.shaping_min_weight
            warmup_end = start_episode + int(n_episodes * warmup_ratio)
            if episode < warmup_end:
                shaping_weight = 1.0
            else:
                decay_progress = (episode - warmup_end) / max(start_episode + n_episodes - warmup_end, 1)
                shaping_weight = max(min_weight, 1.0 - (1.0 - min_weight) * decay_progress)

            # Environment (shares the ValueEstimator)
            env = CleaningEnv(
                X_injected, y_injected, error_list,
                self.model_adapter, self.state_extractor, self.config,
                X_base=X_base, y_base=y_base,
                value_estimator=shared_ve,
                shaping_weight=shaping_weight,
            )

            # Print the adaptive evaluation interval in the first episode
            if episode == start_episode and verbose:
                n_errs = len(error_list)
                N = env._reward_eval_interval
                evals_per_ep = (n_errs // N + 1) if n_errs > 0 else 0
                self.logger.log_info(
                    f"  Reward evaluation interval: N={N} "
                    f"(errors={n_errs}, about {evals_per_ep} evaluations per episode)"
                )

            state = env.reset()
            total_reward = 0

            # Q values of this episode (uncertainty statistics)
            q_values_this_episode = []

            # Episode
            while True:
                if self.config.agent_type in (AgentType.TWO_STAGE, AgentType.DUELING_TWO_STAGE):
                    # Collect Q values
                    try:
                        q_vals = self.agent.get_q_values(state)
                        q_values_this_episode.append(q_vals)
                    except Exception:
                        pass

                    final_action, stage1_action, stage2_action = self.agent.act(state, training=True)
                    next_state, reward, done, info = env.step(final_action)

                    # Stage-wise reward routing
                    stage1_reward = info.get('stage1_reward', reward)
                    self.agent.remember_stage1(state, stage1_action, stage1_reward, next_state, done)

                    if stage2_action is not None:
                        stage2_reward = info.get('stage2_reward', reward)
                        if stage2_reward is not None:
                            self.agent.remember_stage2(state, stage2_action, stage2_reward, next_state, done)
                else:
                    # Collect Q values
                    try:
                        q_vals = self.agent.get_q_values(state)
                        q_values_this_episode.append(q_vals)
                    except Exception:
                        pass

                    action = self.agent.act(state, training=True)
                    next_state, reward, done, _ = env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                if done:
                    break

            # Experience replay
            self.agent.replay(batch_size=self.config.batch_size)

            # Target network update
            if (episode + 1) % self.config.target_update_freq == 0:
                self.agent.update_target_model()

            # History (reuses the score already evaluated in _calculate_final_reward)
            final_score = getattr(env, 'last_episode_score', None)
            if final_score is None:
                final_score = env._evaluate_score(env.X_current, env.y_current)
            self.history['episode'].append(episode + 1)
            self.history['score'].append(final_score)
            self.history['reward'].append(total_reward)
            self.history['epsilon'].append(self.agent.epsilon)
            self.history['no_action'].append(env.action_counts['no_action'])
            self.history['repair_value'].append(env.action_counts['repair_value'])
            self.history['delete'].append(env.action_counts['delete'])
            self.history['replace_nearby'].append(env.action_counts['replace_nearby'])

            # Uncertainty statistics
            if q_values_this_episode:
                q_arr = np.array(q_values_this_episode)
                q_std = float(np.mean(np.std(q_arr, axis=1)))
                # Action probabilities: mean of softmax(Q)
                q_mean = np.mean(q_arr, axis=0)
                q_exp = np.exp(q_mean - np.max(q_mean))
                action_probs = (q_exp / q_exp.sum()).tolist()
            else:
                q_std = 0.0
                action_probs = []
            self.history['q_std'].append(q_std)
            self.history['action_probs'].append(action_probs)

            self.logger.log_episode(
                episode + 1, final_score, total_reward,
                self.agent.epsilon, env.action_counts
            )

            # Best-model tracking
            current_cost = env.action_counts['repair_value']
            is_better = False

            if self.config.task_type == TaskType.REGRESSION:
                if final_score > best_score or (final_score == best_score and current_cost <= best_cost):
                    is_better = True
            else:
                if final_score > best_score or (final_score == best_score and current_cost <= best_cost):
                    is_better = True

            if is_better:
                best_score = final_score
                best_cost = current_cost
                best_episode = episode + 1
                best_actions = env.action_counts.copy()
                best_weights = self.agent.get_weights()

            # Progress
            if verbose and (episode + 1) % self.config.log_interval == 0:
                self.logger.log_info(
                    f"  Episode {episode + 1}/{end_episode}: "
                    f"Score={final_score:.4f}, Best={best_score:.4f}, Eps={self.agent.epsilon:.3f}"
                )
                self.logger.log_info(f"    Actions: {env.action_counts}")

        # Restore the best model weights
        if best_weights is not None:
            self.agent.set_weights(best_weights)
            if verbose:
                self.logger.log_info(
                    f"\nRestored the best model weights (episode {best_episode}, "
                    f"Score={best_score:.4f}, Cost={best_cost})"
                )

        # Resume metadata of the agent
        self.agent.total_episodes = end_episode
        self.agent.best_score = best_score
        self.agent.best_episode = best_episode

        if verbose:
            self.logger.log_info(f"\nTraining finished. Best score: {best_score:.4f} ({end_episode} episodes)")
            if best_actions:
                self.logger.log_info(f"  Action counts of the best episode: {best_actions}")

        return self.agent, self.history

    # ====================================================================
    # Helpers
    # ====================================================================

    def _analyze_label_injection(self,
                                  detected_errors: Optional[Dict[str, List]],
                                  y_dirty: np.ndarray,
                                  verbose: bool,
                                  ) -> Tuple[Optional[LabelErrorPattern], Tuple[float, float]]:
        """Decide whether label errors are injected.

        Label errors are not injected when the detector found none.

        Returns:
            (label_pattern, label_rate_range)
        """
        if detected_errors is None:
            return None, (0.0, 0.0)

        detected_label = detected_errors.get('label_noise', [])
        if not detected_label:
            if verbose:
                self.logger.log_info("  Label injection: the detector found no label errors, no injection")
            return None, (0.0, 0.0)

        # Label error pattern
        task_type_str = self.config.task_type.value if hasattr(self.config.task_type, 'value') else str(self.config.task_type)
        pattern = analyze_label_error_pattern(detected_label, y_dirty, task_type=task_type_str)

        if verbose:
            self.logger.log_info(
                f"  Label injection: {len(detected_label)} label errors detected, "
                f"rate={pattern.error_rate:.3f}, symmetric={pattern.is_symmetric}")

        return pattern, self.config.label_rate_range

    def _compute_adaptive_rate_ranges(
        self,
        detected_errors: Optional[Dict[str, List]],
        X_dirty: np.ndarray,
        y: np.ndarray,
        verbose: bool,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Adaptive injection rate ranges from the detection result.

          - an error type that was not detected gets rate_range = (0, 0) and is not injected
          - a detected error type is centred on its detection rate with a spread of 20%

        Returns:
            None without a detection result (configuration defaults apply), otherwise the range dict
        """
        if detected_errors is None:
            return None

        n_rows, n_features = X_dirty.shape
        n_cells = n_rows * n_features

        type_map = {
            'missing':   ('missing',     n_cells),
            'semantic':  ('semantic',    n_cells),
            'syntactic': ('syntactic',   n_cells),
            'label':     ('label_noise', n_rows),
        }

        result = {}
        for out_key, (det_key, denominator) in type_map.items():
            n_detected = len(detected_errors.get(det_key, []))
            if n_detected == 0:
                # With DC/FD/CFD rules but no semantic error detected in the raw data, use a minimum rate,
                # because the rules can cover violations that appear only after injection
                if out_key == 'semantic' and self._has_semantic_rules():
                    result[out_key] = (0.005, 0.02)
                else:
                    result[out_key] = (0.0, 0.0)
            else:
                rate = n_detected / max(denominator, 1)
                lo = max(rate * 0.8, 0.005)
                hi = max(rate * 1.2, lo + 0.005)
                result[out_key] = (round(lo, 6), round(hi, 6))

        if verbose:
            self.logger.log_info("  Adaptive injection rate ranges (from the detection result):")
            for k, (lo, hi) in result.items():
                status = "not injected" if lo == 0.0 else f"({lo:.4f}, {hi:.4f})"
                self.logger.log_info(f"    {k}: {status}")

        return result

    def _has_semantic_rules(self) -> bool:
        """Return whether the configuration holds semantic rules (FD, CFD or DC).

        Used by _compute_adaptive_rate_ranges(): without semantic errors in the raw data but with
        rules, a minimum injection rate still covers the violations the rules can detect.

        Returns:
            True if there is an FD, CFD or DC rule
        """
        # FD rules
        if self.config.fd_rules:
            return True
        # CFD and DC rules in rich_rules
        rr = self.config.rich_rules
        if rr and rr.get('has_rich_rules'):
            if rr.get('cfd_rules'):
                return True
            if rr.get('dc_rules'):
                return True
        return False

    def _build_base_data(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]] = None,
        fill_nan: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the base data (DeleteFix strategy).

        With detected_errors only the rows with detected errors are deleted and the other rows are kept.
        Without a detection result the NaN-free strategy is used.

        Args:
            X_dirty: raw dirty data
            y: label vector
            detected_errors: detection result (None falls back to the NaN-free strategy)
            fill_nan: fill residual NaN (missing values in error-free rows get the column mean)
            verbose: print log lines

        Returns:
            (X_base, y_base)
        """
        if detected_errors is not None:
            # DeleteFix: delete only the rows with detected errors
            error_row_set = set()
            for err_type in ('missing', 'semantic', 'syntactic', 'label_noise'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        error_row_set.add(int(item[0]))
                    elif isinstance(item, dict) and 'idx' in item:
                        error_row_set.add(int(item['idx']))

            n_total = len(X_dirty)
            min_keep = max(10, int(n_total * 0.2))  # Keep at least 20%

            if len(error_row_set) <= n_total - min_keep:
                keep_mask = np.array([i not in error_row_set for i in range(n_total)])
            else:
                # Over the limit: delete rows in descending order of error count
                from collections import Counter
                row_error_count = Counter()
                for err_type in ('label_noise', 'missing', 'syntactic', 'semantic'):
                    for item in detected_errors.get(err_type, []):
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            row_error_count[int(item[0])] += 1
                        elif isinstance(item, dict) and 'idx' in item:
                            row_error_count[int(item['idx'])] += 1
                max_delete = n_total - min_keep
                sorted_rows = sorted(row_error_count.keys(),
                                     key=lambda r: row_error_count[r], reverse=True)
                delete_set = set(sorted_rows[:max_delete])
                keep_mask = np.array([i not in delete_set for i in range(n_total)])

            X_base = X_dirty[keep_mask].copy()
            y_base = y[keep_mask].copy()
            n_deleted = n_total - int(keep_mask.sum())

            # Fill residual NaN (cell-wise ValueEstimator estimates)
            n_residual_nan = 0
            if fill_nan:
                col_means = np.nanmean(X_base, axis=0)
                # Index map: X_base[i] -> original row of dirty_df
                keep_indices = np.where(keep_mask)[0]
                # Cell-wise ValueEstimator estimates
                from demandclean.core.environments.value_estimation import ValueEstimator
                ve = ValueEstimator(self.config)
                for j in range(X_base.shape[1]):
                    nan_mask_j = np.isnan(X_base[:, j])
                    n_residual_nan += int(nan_mask_j.sum())
                    for i in np.where(nan_mask_j)[0]:
                        X_base[i, j] = ve.estimate_feature_value(
                            X_base, i, j, set(), col_means,
                            dirty_df_row_indices=keep_indices,
                        )
                # Residual NaN labels: KNN majority vote
                y_nan = np.isnan(y_base)
                if y_nan.any():
                    n_residual_nan += int(y_nan.sum())
                    for i in np.where(y_nan)[0]:
                        y_base[i] = self._estimate_label_by_knn(X_base, y_base, i, k=5)

            if verbose:
                self.logger.log_info(
                    f"  Base data: {len(X_base)} rows "
                    f"(DeleteFix: {n_deleted} error rows deleted, "
                    f"{n_residual_nan} residual NaN filled)")
        else:
            # Fallback without a detection result: delete every row that contains NaN
            no_nan_mask = ~np.isnan(X_dirty).any(axis=1)
            X_base = X_dirty[no_nan_mask].copy()
            y_base = y[no_nan_mask].copy()

            if verbose:
                self.logger.log_info(f"  Base data: {len(X_base)} rows (NaN-free)")

        return X_base, y_base

    def _build_ve_fill_data(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """VE-Fill strategy: keep every row and repair the detected cells with the value estimator.

        Differences from DeleteFix:
          - no row is deleted, the sample size is kept
          - every NaN is first filled with the column mean, so the matrix is NaN-free
          - the detected error cells are then overwritten with VE estimates (KNN is faster on a NaN-free matrix)
          - label errors are repaired by KNN majority vote

        Cost:
          a VE call on a matrix with NaN repeats the NaN fill inside KNN, O(K x N x M) with that overhead.
          Filling once costs O(N x M), after which the VE pass is O(K x N x M) on a NaN-free matrix.

        Returns:
            (X_filled, y_filled), all rows with every detected error repaired by VE
        """
        X_out = X_dirty.copy()
        y_out = y.copy()

        # Step 0: fill every NaN with the column mean, so the matrix is NaN-free
        col_means = np.nanmean(X_out, axis=0)
        n_total_nan = 0
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                n_total_nan += n_nan
                X_out[nan_mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

        # Fill NaN labels as well
        y_nan = np.isnan(y_out)
        if y_nan.any():
            y_mean = np.nanmean(y_out)
            y_out[y_nan] = y_mean if not np.isnan(y_mean) else 0.0

        # Step 1: error cells
        error_cells = set()  # (row, col) pairs
        label_error_rows = set()

        if detected_errors:
            for err_type in ('missing', 'semantic', 'syntactic'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        error_cells.add((int(item[0]), int(item[1])))
                    elif isinstance(item, dict):
                        row = int(item.get('idx', -1))
                        col = int(item.get('col', -1))
                        if row >= 0 and col >= 0:
                            error_cells.add((row, col))
            for item in detected_errors.get('label_noise', []):
                if isinstance(item, (list, tuple)):
                    label_error_rows.add(int(item[0]))
                elif isinstance(item, dict):
                    row = int(item.get('idx', -1))
                    if row >= 0:
                        label_error_rows.add(row)

        # Step 2: overwrite the detected error cells with VE estimates (NaN-free matrix)
        ve = ValueEstimator(self.config)
        for row, col in error_cells:
            if row < X_out.shape[0] and col < X_out.shape[1]:
                X_out[row, col] = ve.estimate_feature_value(
                    X_out, row, col, set(), col_means)

        # Step 3: repair label errors by KNN majority vote
        for row in label_error_rows:
            if 0 <= row < len(y_out):
                y_out[row] = self._estimate_label_by_knn(X_out, y_out, row, k=5)

        if verbose:
            self.logger.log_info(
                f"  VE-Fill: {len(X_out)} rows "
                f"({n_total_nan} NaN pre-filled, "
                f"{len(error_cells)} error cells estimated by VE, "
                f"{len(label_error_rows)} label errors)")

        return X_out, y_out

    def _normalize_cv_score(self, raw_score: float) -> float:
        """Normalise a raw model score to [0, 1] (same as CleaningEnv._normalize_raw_score)."""
        if self.config.task_type == TaskType.REGRESSION:
            mse = abs(raw_score)
            return 1.0 / (1.0 + mse)
        elif self.config.task_type == TaskType.CLUSTERING:
            return (raw_score + 1.0) / 2.0
        else:
            return raw_score

    def _cv_evaluate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
        """K-fold CV on a candidate table, returns the mean normalised score.

        Args:
            X: candidate feature matrix (NaN-free)
            y: candidate label vector
            n_folds: number of folds

        Returns:
            mean normalised score
        """
        from sklearn.model_selection import KFold

        # Too little data: a single evaluation
        if len(X) < n_folds * 2:
            try:
                model_kwargs = self.config.model_kwargs or {}
                adapter = create_model_adapter(
                    self.config.model_type, self.config.task_type, **model_kwargs)
                adapter.fit(X, y)
                raw = adapter.evaluate(X, y)
                return self._normalize_cv_score(raw)
            except Exception:
                return 0.0

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        model_kwargs = self.config.model_kwargs or {}

        for train_idx, val_idx in kf.split(X):
            try:
                adapter = create_model_adapter(
                    self.config.model_type, self.config.task_type, **model_kwargs)
                adapter.fit(X[train_idx], y[train_idx])
                raw = adapter.evaluate(X[val_idx], y[val_idx])
                scores.append(self._normalize_cv_score(raw))
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _select_best_base(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Choose the clean base strategy automatically.

        X_dirty already holds the ground-truth values of the calibration seed set when
        apply_raha_truth=True, so both candidates share that preprocessing.

        Steps:
          Step 1: DeleteFix candidate from _build_base_data()
          Step 2: VE-Fill candidate from _build_ve_fill_data()
          Step 3: 5-fold CV, the candidate with the higher score is chosen

        Returns:
            (X_base, y_base, strategy_name)
        """
        n_folds = getattr(self.config, 'base_cv_folds', 5)

        # Step 1: DeleteFix candidate
        X_delete, y_delete = self._build_base_data(
            X_dirty, y, detected_errors=detected_errors,
            fill_nan=True, verbose=False)

        # Step 2: VE-Fill candidate
        X_vefill, y_vefill = self._build_ve_fill_data(
            X_dirty, y, detected_errors, verbose=False)

        # Step 3: CV score weighted by the fraction of rows kept
        # Without a clean validation split, self-CV overrates small tables (consistent but poorly generalising),
        # so a weight of sqrt(n/N) penalises the loss of rows
        score_delete = self._cv_evaluate(X_delete, y_delete, n_folds)
        score_vefill = self._cv_evaluate(X_vefill, y_vefill, n_folds)

        n_total = len(X_dirty)
        weight_delete = np.sqrt(len(X_delete) / max(n_total, 1))
        weight_vefill = np.sqrt(len(X_vefill) / max(n_total, 1))

        adj_score_delete = score_delete * weight_delete
        adj_score_vefill = score_vefill * weight_vefill

        # Choose the strategy with the higher weighted score
        if adj_score_vefill > adj_score_delete:
            chosen = 'VE-Fill'
            X_base, y_base = X_vefill, y_vefill
        else:
            chosen = 'DeleteFix'
            X_base, y_base = X_delete, y_delete

        # Always printed (independent of verbose)
        self.logger.log_info(
            f"  Clean base selection ({n_folds}-fold CV weighted by the fraction of rows kept):")
        self.logger.log_info(
            f"    DeleteFix: {len(X_delete)} rows, CV={score_delete:.4f}, "
            f"weight={weight_delete:.3f}, weighted={adj_score_delete:.4f}")
        self.logger.log_info(
            f"    VE-Fill:   {len(X_vefill)} rows, CV={score_vefill:.4f}, "
            f"weight={weight_vefill:.3f}, weighted={adj_score_vefill:.4f}")
        self.logger.log_info(
            f"    -> chosen: {chosen} ({len(X_base)} rows)")

        return X_base, y_base, chosen

    def _build_detected_error_list(self,
                                    detected_errors: Optional[Dict[str, List]],
                                    X_dirty: np.ndarray,
                                    y_dirty: np.ndarray,
                                    verbose: bool,
                                    ) -> Tuple[List[Dict], Set[Tuple[int, int]]]:
        """Fixed error_list of the detected errors (self-supervised mode).

        - missing/semantic/syntactic: repair_value from the ValueEstimator (FD, then KNN, then DOMAIN)
        - label_noise: repair_value from a KNN majority vote

        Returns:
            (fixed_error_list, detected_cells)
        """
        fixed_error_list = []
        detected_cells: Set[Tuple[int, int]] = set()

        if not detected_errors:
            return fixed_error_list, detected_cells

        # ValueEstimator (FD + KNN + DOMAIN)
        value_estimator = ValueEstimator(self.config)
        col_means = np.nanmean(X_dirty, axis=0)

        type_map = {'missing': 0, 'semantic': 1, 'syntactic': 2, 'label_noise': 3}

        for err_type_name, err_type_id in type_map.items():
            for item in detected_errors.get(err_type_name, []):
                if isinstance(item, (list, tuple)):
                    idx = int(item[0])
                    col = int(item[1]) if len(item) > 1 else -1
                elif isinstance(item, dict):
                    idx = int(item.get('idx', -1))
                    col = int(item.get('col', -1))
                else:
                    continue

                if idx < 0:
                    continue

                # repair_value
                if err_type_id == 3:
                    # Label error: KNN majority vote
                    repair_value = self._estimate_label_by_knn(
                        X_dirty, y_dirty, idx, k=5
                    )
                elif col >= 0 and col < X_dirty.shape[1]:
                    # Feature error: FD, then multivariate KNN, then DOMAIN clipping
                    repair_value = value_estimator.estimate_feature_value(
                        X_dirty, idx, col, set(), col_means
                    )
                else:
                    repair_value = 0.0

                fixed_error_list.append({
                    'idx': idx,
                    'col': col,
                    'type': err_type_id,
                    'repair_value': repair_value,
                })

                if col >= 0:
                    detected_cells.add((idx, col))

        if verbose:
            from collections import Counter
            type_counts = Counter(e['type'] for e in fixed_error_list)
            self.logger.log_info(
                f"  Fixed error list: {dict(type_counts)} "
                f"(0=missing, 1=semantic, 2=syntactic, 3=label)")
            self.logger.log_info(f"  Value estimator: {value_estimator.summary()}")

        return fixed_error_list, detected_cells

    def _estimate_label_by_knn(self, X: np.ndarray, y: np.ndarray,
                                idx: int, k: int = 5) -> float:
        """Estimate a label by KNN majority vote or weighted mean.

        Args:
            X: feature matrix
            y: label vector
            idx: target row index
            k: number of neighbours

        Returns:
            the estimated label
        """
        # NaN fill
        X_filled = X.copy()
        for c in range(X_filled.shape[1]):
            nan_mask = np.isnan(X_filled[:, c])
            if nan_mask.any():
                col_mean = np.nanmean(X_filled[:, c])
                X_filled[nan_mask, c] = col_mean if not np.isnan(col_mean) else 0.0

        target = X_filled[idx]
        distances = np.linalg.norm(X_filled - target, axis=1)
        distances[idx] = np.inf

        k = min(k, (distances < np.inf).sum())
        if k == 0:
            return y[idx]

        nearest = np.argsort(distances)[:k]
        nearest_labels = y[nearest]
        valid = nearest_labels[~np.isnan(nearest_labels)]

        if len(valid) == 0:
            return y[idx]

        # Regression: inverse-distance weighted mean
        if self.config.task_type == TaskType.REGRESSION:
            valid_dists = distances[nearest][~np.isnan(nearest_labels)]
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid, weights=weights))

        # Classification: majority vote
        unique, counts = np.unique(valid, return_counts=True)
        return float(unique[np.argmax(counts)])

    # ====================================================================
    # Model management
    # ====================================================================

    def save_agent(self, path: str) -> None:
        """Save the agent."""
        if self.agent is None:
            raise ValueError("Agent is not trained and cannot be saved")
        self.model_io.save_agent(self.agent, path)

    def load_agent(self, path: str) -> BaseAgent:
        """Load the agent."""
        self.agent = self.model_io.load_agent(path, self.config.agent_type)
        return self.agent

    def get_agent(self) -> Optional[BaseAgent]:
        """Return the current agent."""
        return self.agent

    def get_history(self) -> Dict[str, List]:
        """Return the training history."""
        return self.history
