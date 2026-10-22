"""
DemandClean Configuration Management
====================================

Defines the configuration dataclass and enumeration types for the system.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import os


class TaskType(Enum):
    """Task type enumeration."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class ModelType(Enum):
    """Model type enumeration."""
    # Classification models
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    # Regression models
    LINEAR = "linear"
    RIDGE = "ridge"
    XGBOOST_REG = "xgboost_reg"
    # Clustering models
    KMEANS = "kmeans"


class AgentType(Enum):
    """DQN agent type enumeration."""
    SINGLE_STAGE = "single"
    TWO_STAGE = "two_stage"
    DUELING_SINGLE_STAGE = "dueling_single"
    DUELING_TWO_STAGE = "dueling_two_stage"


class DetectorMode(Enum):
    """Detector mode enumeration."""
    AUTO = "auto"       # Automatic detection (FD + RAHA + isnan)
    ORACLE = "oracle"   # Full error labels supplied externally (used for ablations)


class InferenceMode(Enum):
    """Inference mode enumeration."""
    SINGLE_PHASE = "single_phase"
    TWO_PHASE = "two_phase"


@dataclass
class DemandCleanConfig:
    """
    Main configuration class for DemandClean.

    Attributes:
        task_type: Task type (classification / regression)
        model_type: Model type
        agent_type: DQN agent type (single-stage / two-stage)
        detector_mode: Detector mode (auto / oracle)
        inference_mode: Inference mode (single_phase / two_phase)

        n_episodes: Number of training episodes
        repair_lambda: Cost coefficient for truth-based repairs
        min_truth_budget: Minimum number of truth values to use
        max_truth_budget: Maximum number of truth values to use

        state_size: State vector dimensionality
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration rate decay
        learning_rate: Learning rate
        batch_size: Batch size
        memory_size: Size of the experience replay buffer

        missing_rate_range: Missing-value injection rate range
        semantic_rate_range: Semantic-error injection rate range
        syntactic_rate_range: Syntactic-error injection rate range

        rules_path: Path to the rules file (data/{dataset}/rules.txt)
        fd_rules: Parsed FD rule list [(lhs_str, rhs_str), ...]
        column_names: List of data column names

        save_path: Output save path
        model_path: Model save path
        detector_path: Detector save path

        verbose: Whether to output verbose logs
        log_interval: Logging interval (every N episodes)
    """
    # Task configuration
    task_type: TaskType = TaskType.CLASSIFICATION
    model_type: ModelType = ModelType.SVM
    agent_type: AgentType = AgentType.SINGLE_STAGE
    detector_mode: DetectorMode = DetectorMode.AUTO
    inference_mode: InferenceMode = InferenceMode.SINGLE_PHASE

    # Training configuration
    n_episodes: int = 300
    repair_lambda: float = 0.03
    min_truth_budget: Optional[int] = None   # not used by the environment
    max_truth_budget: Optional[int] = None   # cap on GT-repairs, 0 disables them (variant ngt)

    # Repair budget (max_repair_ratio and max_truth_budget bound the number of GT-repairs)
    min_repair_ratio: float = 0.1       # not used by the environment
    max_repair_ratio: float = 0.3       # Repair budget cap: max_repair_count = int(n_errors * ratio)
    repair_sensitivity: float = 10.0    # not used by the environment

    # Oracle mode
    use_clean_validation: bool = True    # Use a clean validation set as the reward signal

    # DQN configuration
    state_size: int = 10  # 8 error-level + 2 global (remaining_budget_ratio, remaining_errors_ratio)
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    learning_rate: float = 0.0005
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 5  # Target network update frequency

    # Error-injection configuration
    missing_rate_range: Tuple[float, float] = (0.02, 0.08)
    semantic_rate_range: Tuple[float, float] = (0.05, 0.15)
    syntactic_rate_range: Tuple[float, float] = (0.1, 0.25)
    label_rate_range: Tuple[float, float] = (0.0, 0.05)  # Defaults to 0; enabled based on detection results

    # FD rule configuration
    rules_path: Optional[str] = None
    fd_rules: Optional[List[Tuple[str, str]]] = None
    column_names: Optional[List[str]] = None
    rich_rules: Optional[Dict[str, Any]] = None  # Rich rule dictionary (DOMAIN/CFD/DC)

    # Training mode
    training_mode: str = "clean_base"  # "clean_base" or "self_supervised"

    # Encoding utilities (used by ErrorInjector to re-encode after injection in CSV space)
    label_encoders: Optional[Dict[str, Any]] = None    # {col_name: LabelEncoder}
    scaler: Optional[Any] = None                        # StandardScaler
    categorical_cols: Optional[set] = None              # Set of categorical columns
    dirty_df: Optional[Any] = None                      # Original dirty DataFrame in CSV space
    clean_df: Optional[Any] = None                      # Original clean DataFrame in CSV space
    label_col: Optional[str] = None                     # Label column name
    protected_cols: Optional[set] = None                # Protected column names (excluded from syntactic injection)

    # Automatic Clean Base selection
    auto_select_base: bool = True        # Whether to automatically pick the optimal clean base strategy (DeleteFix vs VE-Fill)
    base_cv_folds: int = 5              # Number of CV folds

    # Path configuration
    save_path: str = "output"
    model_path: Optional[str] = None
    detector_path: Optional[str] = None

    # Logging configuration
    verbose: bool = True
    log_interval: int = 50

    # RAHA detector configuration
    raha_n_runs: int = 15
    raha_threshold_m: int = 4
    semantic_rate: float = 0.9  # Semantic-error detection rate (used for simulation)

    # Feature-importance refresh interval (None means the environment computes it automatically as max(20, n_errors // 10))
    importance_refresh_interval: Optional[int] = None

    # Reward evaluation configuration
    reward_eval_interval: int = 0          # 0 = adaptive (auto-computed based on data size); >0 = manual override
    eval_sample_ratio: float = 1.0         # Validation sampling ratio (1.0 = full, 0.3 = 30% subsample)
    model_kwargs: Optional[Dict[str, Any]] = None  # Extra arguments forwarded to the model adapter (e.g., n_estimators=10)

    # Shaping decay configuration
    shaping_warmup_ratio: float = 0.5      # Keep full shaping for the first 50% of episodes
    shaping_min_weight: float = 0.1        # Decay floor (not fully disabled; retains weak guidance)

    # Reward shaping constants are task-type specific (regression: delete_shaping_reward -0.05, keep_rate_weight 1.0)
    delete_shaping_reward: float = -0.02       # Shaping reward for delete
    keep_rate_weight: float = 0.2              # Weight of keep_rate in the final reward
    regression_log_normalize: bool = False     # Whether regression tasks use log-compressed normalization 1/(1+log(1+MSE))

    # RAHA truth switches
    apply_raha_truth: bool = True   # Whether to apply truth values from RAHA-annotated rows in the repair
    count_raha_cost: bool = True    # Whether to count the RAHA annotation cost in the total truth cost

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure save_path exists
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # Coerce strings to enums
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        if isinstance(self.agent_type, str):
            self.agent_type = AgentType(self.agent_type)
        if isinstance(self.detector_mode, str):
            self.detector_mode = DetectorMode(self.detector_mode)
        if isinstance(self.inference_mode, str):
            self.inference_mode = InferenceMode(self.inference_mode)

    @property
    def is_classification(self) -> bool:
        """Whether the task is classification."""
        return self.task_type == TaskType.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """Whether the task is regression."""
        return self.task_type == TaskType.REGRESSION

    @property
    def is_clustering(self) -> bool:
        """Whether the task is clustering."""
        return self.task_type == TaskType.CLUSTERING

    @property
    def is_oracle(self) -> bool:
        """Whether the Oracle detector is used."""
        return self.detector_mode == DetectorMode.ORACLE

    @property
    def is_two_phase(self) -> bool:
        """Whether two-phase inference is used."""
        return self.inference_mode == InferenceMode.TWO_PHASE

    @property
    def epsilon(self) -> float:
        """Exploration rate (alias for epsilon_start)."""
        return self.epsilon_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            'task_type': self.task_type.value,
            'model_type': self.model_type.value,
            'agent_type': self.agent_type.value,
            'detector_mode': self.detector_mode.value,
            'inference_mode': self.inference_mode.value,
            'n_episodes': self.n_episodes,
            'repair_lambda': self.repair_lambda,
            'min_truth_budget': self.min_truth_budget,
            'max_truth_budget': self.max_truth_budget,
            'state_size': self.state_size,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'rules_path': self.rules_path,
            'save_path': self.save_path,
            'verbose': self.verbose,
            'apply_raha_truth': self.apply_raha_truth,
            'count_raha_cost': self.count_raha_cost,
            'reward_eval_interval': self.reward_eval_interval,
            'eval_sample_ratio': self.eval_sample_ratio,
            'model_kwargs': self.model_kwargs,
            'auto_select_base': self.auto_select_base,
            'base_cv_folds': self.base_cv_folds,
            'min_repair_ratio': self.min_repair_ratio,
            'max_repair_ratio': self.max_repair_ratio,
            'repair_sensitivity': self.repair_sensitivity,
            'use_clean_validation': self.use_clean_validation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DemandCleanConfig':
        """Create from a dictionary."""
        return cls(**d)

    def copy(self, **updates) -> 'DemandCleanConfig':
        """Create a copy, optionally updating some fields."""
        d = self.to_dict()
        d.update(updates)
        return self.from_dict(d)
