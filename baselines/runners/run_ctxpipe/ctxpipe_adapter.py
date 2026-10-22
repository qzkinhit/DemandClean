"""
CtxPipe Adapter for the DemandClean-Benchmark project.
Adapts the official ctxpipe API to the project's unified interface.

Notes:
- Methods/ctxpipe mirrors the official repository exactly.
- All compatibility shims (device detection, path handling) live in this adapter.
- Uses the pretrained model for inference; no training required.
"""
import os
import sys
import warnings
import pandas as pd
from typing import Tuple, Optional

# ============================================================================
# Step 1: apply compatibility patches before importing ctxpipe.
# ============================================================================

# Environment variables.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add the ctxpipe directory to sys.path (the official layout keeps env.py, config.py, etc. at the root).
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
CTXPIPE_ROOT = os.path.join(PROJECT_ROOT, 'Methods', 'ctxpipe')
sys.path.insert(0, CTXPIPE_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Establish device compatibility before importing torch and ctxpipe.
import torch

# Auto-detect device.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[CtxPipe Adapter] Using device: CUDA")
else:
    DEVICE = torch.device("cpu")
    print(f"[CtxPipe Adapter] Using device: CPU (CUDA not available)")


# ============================================================================
# Step 2: import ctxpipe modules and apply runtime patches.
# ============================================================================

# Import ctxpipe's env module and patch the device.
try:
    import env as ctxpipe_env
    # Patch the device (the upstream hardcodes CUDA).
    ctxpipe_env.DEVICE = DEVICE
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing env module: {e}")
    raise

# Import the config module and patch it.
try:
    import config as ctxpipe_config
    # Patch GlobalConfig's device.
    ctxpipe_config.GlobalConfig.device = DEVICE
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing config module: {e}")
    raise

# Import the remaining modules.
try:
    from ctxpipe.pipegen import PipelineGenerator
    from ctxpipe.dataset import Dataset
    from ctxpipe.info import Info
    from ctxpipe.stats import init_stats_db
    import util as ctxpipe_util
except ImportError as e:
    print(f"[CtxPipe Adapter] Error importing ctxpipe modules: {e}")
    print(f"[CtxPipe Adapter] CTXPIPE_ROOT: {CTXPIPE_ROOT}")
    raise


# ============================================================================
# Step 3: adapter class.
# ============================================================================

class CtxPipeAdapter:
    """
    Adapter class that wraps the official ctxpipe API.
    """

    def __init__(self, model_tag: str = "ctx_50000"):
        """
        Initialize the adapter.

        Args:
            model_tag: pretrained model tag, default "ctx_50000"
        """
        self.model_tag = model_tag
        self._initialized = False

    def _init_ctxpipe(self):
        """Initialize the ctxpipe environment."""
        if self._initialized:
            return

        # Mirror the upstream init function (device has already been patched).
        import numpy as np
        import deterministic

        deterministic.seed_everything()
        np.set_printoptions(suppress=True)
        torch.set_printoptions(sci_mode=False)
        warnings.filterwarnings("ignore")

        # Create the required directories.
        ctxpipe_config.default_config.makedirs()

        self._initialized = True

    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path to forward slashes for Windows/Linux portability.

        Args:
            path: raw path

        Returns:
            normalized path
        """
        return path.replace("\\", "/")

    def has_index_column(self, df: pd.DataFrame) -> bool:
        """Check whether the DataFrame has an `index` column as its first column."""
        first_col = df.columns[0]
        return first_col.strip().lower().replace('\ufeff', '') == 'index'

    def detect_label_column(self, df: pd.DataFrame) -> int:
        """Auto-detect the label column index."""
        label_candidates = ['label', 'target', 'class', 'y', 'output', 'income',
                          'style', 'cnt', 'gt', 'labels', 'sound_pressure_level',
                          'soil_moisture']
        columns_lower = [col.lower() for col in df.columns]

        for candidate in label_candidates:
            if candidate in columns_lower:
                return columns_lower.index(candidate)

        # Fall back to the last column.
        print(f"[CtxPipe Adapter] Warning: No label column found. Using last column.")
        return len(df.columns) - 1

    def run_ctxpipe(
        self,
        dirty_path: str,
        task_name: str,
        label_index: Optional[int] = None,
        output_path: Optional[str] = None,
        schema_preserving: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Run ctxpipe to generate a data preparation pipeline.

        Args:
            dirty_path: path to the dirty CSV
            task_name: task name
            label_index: label column index (auto-detected when None)
            output_path: output directory
            schema_preserving: if True, only apply missing-value imputation

        Returns:
            (processed DataFrame, result dict)
        """
        # Initialize ctxpipe.
        self._init_ctxpipe()

        # Normalize the path.
        dirty_path = self._normalize_path(dirty_path)

        # Load the data.
        df_original = pd.read_csv(dirty_path)
        original_index_series = None

        # Detect an `index` column.
        has_index_col = self.has_index_column(df_original)
        if has_index_col:
            print(f"[CtxPipe Adapter] Detected index column. Removing for processing.")
            index_col_name = df_original.columns[0]
            original_index_series = df_original[index_col_name].copy()
            df = df_original.set_index(index_col_name)
            df.index.name = None
        else:
            df = df_original.copy()

        # Detect or use the supplied label column.
        if label_index is None:
            label_index = self.detect_label_column(df)
            print(f"[CtxPipe Adapter] Auto-detected label column: {label_index} ({df.columns[label_index]})")
        else:
            if has_index_col and label_index > 0:
                label_index = label_index - 1
            print(f"[CtxPipe Adapter] Using label column: {label_index} ({df.columns[label_index]})")

        # Validate the label index.
        if label_index < 0 or label_index >= len(df.columns):
            raise ValueError(f"Invalid label_index {label_index}. Must be in [0, {len(df.columns)-1}]")

        # Schema-preserving mode: only simple imputation.
        if schema_preserving:
            print("[CtxPipe Adapter] Schema-preserving mode: simple imputation only.")
            df_sp = df.copy()
            df_sp = df_sp.replace({"empty": pd.NA, "": pd.NA})

            for col in df_sp.columns:
                if pd.api.types.is_numeric_dtype(df_sp[col]):
                    if df_sp[col].isna().any():
                        df_sp[col] = df_sp[col].fillna(df_sp[col].median())
                else:
                    if df_sp[col].isna().any():
                        mode = df_sp[col].mode()
                        df_sp[col] = df_sp[col].fillna(mode.iloc[0] if not mode.empty else "")

            results = {
                'ai_sequence': ['ImputerNum', 'ImputerCat'],
                'ml_score': None,
                'logical_pipeline': None,
            }

            if has_index_col and original_index_series is not None:
                df_sp.insert(0, 'index', original_index_series.values)

            return df_sp, results

        # Create a temporary CSV (without the index column) and turn "empty" into real NaN.
        temp_csv_path = dirty_path

        # Turn the "empty" string into real NaN so ctxpipe detects missing values correctly.
        df_for_ctxpipe = df.copy()
        df_for_ctxpipe = df_for_ctxpipe.replace({"empty": "", "Empty": "", "EMPTY": ""})
        print(f"[CtxPipe Adapter] Converted 'empty' strings to actual empty values for NaN detection.")

        if has_index_col or True:  # always create a temp file so "empty" values are handled correctly
            import tempfile
            dataset_dir = os.path.dirname(dirty_path)
            temp_fd, temp_csv_path = tempfile.mkstemp(suffix='.csv', prefix='ctxpipe_', dir=dataset_dir)
            os.close(temp_fd)
            df_for_ctxpipe.to_csv(temp_csv_path, index=False)
            temp_csv_path = self._normalize_path(temp_csv_path)

        # Configure Info and config.
        if output_path:
            output_path = self._normalize_path(output_path)
            dataset_dir = os.path.dirname(temp_csv_path)
            dataset_root = os.path.dirname(dataset_dir)

            info = Info(
                aipipe_core_prefix=os.path.join(output_path, "aipipe"),
                result_prefix=os.path.join(output_path, "result"),
                dataset_prefix=dataset_root,
            )

            # Build the task info.
            task_entry = {
                "dataset": os.path.basename(dataset_dir),
                "csv_file": os.path.basename(temp_csv_path),
                "label": str(label_index),
                "model": "LogisticRegression",
                "task_name": task_name,
            }

            import json
            os.makedirs(os.path.dirname(info.task_info_path), exist_ok=True)
            with open(info.task_info_path, 'w', encoding='utf-8') as f:
                json.dump({"0": task_entry}, f)

            with open(info.task_index_path, 'w', encoding='utf-8') as f:
                json.dump(["0"], f)

            # Inject the config.
            ctxpipe_config.set_info(info)
            ctxpipe_config.init()
            init_stats_db(info.stats_db_file_path)

        # Build the dataset object (upstream Dataset class).
        dataset = Dataset(
            name=task_name,
            path=temp_csv_path,
            label_column_id=label_index
        )

        # Generate the pipeline.
        print(f"[CtxPipe Adapter] Generating pipeline with model tag: {self.model_tag}")
        pg = PipelineGenerator(dataset, model_tag=self.model_tag)
        pg.generate()

        # Collect the output.
        try:
            pg_stats = pg.output()
        except Exception as e:
            print(f"[CtxPipe Adapter] Warning: pg.output() failed: {e}")
            pg_stats = None

        results = {
            'ai_sequence': pg.ai_sequence,
            'ml_score': pg.ml_score,
            'logical_pipeline': getattr(pg, 'logical_pipeline', None),
        }

        # Apply the ctxpipe-generated pipeline to the data.
        print(f"[CtxPipe Adapter] Applying generated pipeline to data...")
        cleaned_df = self._apply_pipeline(df_for_ctxpipe, pg.ai_sequence, label_index)
        print(f"[CtxPipe Adapter] Pipeline applied successfully.")

        # Clean up the temporary file.
        if temp_csv_path != dirty_path:
            try:
                os.remove(temp_csv_path)
            except:
                pass

        # Restore the index column.
        if has_index_col and original_index_series is not None:
            cleaned_df.insert(0, 'index', original_index_series.values)

        return cleaned_df, results

    def _apply_pipeline(self, df: pd.DataFrame, ai_sequence: list, label_index: int) -> pd.DataFrame:
        """
        Manually apply the ctxpipe-generated pipeline to the data.

        Args:
            df: original DataFrame
            ai_sequence: pipeline sequence generated by ctxpipe
            label_index: label column index

        Returns:
            processed DataFrame
        """
        from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer
        from sklearn.impute import SimpleImputer
        import numpy as np

        # Separate features and label.
        label_col = df.columns[label_index]
        X = df.drop(columns=[label_col])
        y = df[label_col].copy()

        # Critical: convert "empty" to NaN, then attempt dtype coercion.
        # This lets columns like ibu be recognized as numeric.
        X = X.replace({"empty": np.nan, "Empty": np.nan, "EMPTY": np.nan, "": np.nan})

        # Attempt to coerce each column to numeric.
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='ignore')
            except:
                pass

        # Identify numeric and categorical columns.
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

        print(f"[CtxPipe Adapter] Data shape: {X.shape}, Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")
        print(f"[CtxPipe Adapter] Num columns: {num_cols}")
        print(f"[CtxPipe Adapter] AI Sequence: {ai_sequence}")

        # Apply each step of the pipeline.
        # To preserve the original values and structure (for traditional cleaning evaluation),
        # we only apply imputation and skip steps that would change values or structure.
        #
        # Kept (data cleaning):
        #   - ImputerNum, ImputerCat (missing-value imputation)
        #
        # Skipped (ML prep that would alter values/structure):
        #   - LabelEncoder, OneHotEncoder (encoding)
        #   - StandardScaler, MinMaxScaler, PowerTransformer, etc. (scaling/transform)
        #   - PCA, RandomTreesEmbedding, etc. (feature engineering)

        IMPUTATION_OPS = ['imputernum', 'imputercat', 'simpleimputer']

        for step_name in ai_sequence:
            step_name_lower = step_name.lower() if step_name else ""

            # Skip blank operations.
            if step_name == "blank" or not step_name:
                continue

            # Only execute imputation steps.
            is_imputation = any(op in step_name_lower for op in IMPUTATION_OPS)

            if not is_imputation:
                print(f"[CtxPipe Adapter] Skipped {step_name} (not imputation, would change data values)")
                continue

            try:
                # Numeric imputation.
                if "imputernum" in step_name_lower or "simpleimputer" in step_name_lower:
                    if len(num_cols) > 0:
                        imputer = SimpleImputer(strategy='median')
                        X[num_cols] = imputer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied ImputerNum (median)")

                # Categorical imputation.
                elif "imputercat" in step_name_lower:
                    if len(cat_cols) > 0:
                        imputer = SimpleImputer(strategy='most_frequent')
                        X[cat_cols] = imputer.fit_transform(X[cat_cols])
                        print(f"[CtxPipe Adapter] Applied ImputerCat (most_frequent)")

                elif "minmaxscaler" in step_name_lower:
                    if len(num_cols) > 0:
                        scaler = MinMaxScaler()
                        X[num_cols] = scaler.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied MinMaxScaler")

                elif "maxabsscaler" in step_name_lower:
                    if len(num_cols) > 0:
                        scaler = MaxAbsScaler()
                        X[num_cols] = scaler.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied MaxAbsScaler")

                elif "powertransformer" in step_name_lower:
                    if len(num_cols) > 0:
                        # PowerTransformer requires positive values; shift first.
                        for col in num_cols:
                            if (X[col] <= 0).any():
                                X[col] = X[col] - X[col].min() + 1
                        transformer = PowerTransformer(method='yeo-johnson')
                        X[num_cols] = transformer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied PowerTransformer")

                elif "quantiletransformer" in step_name_lower:
                    if len(num_cols) > 0:
                        transformer = QuantileTransformer(output_distribution='normal')
                        X[num_cols] = transformer.fit_transform(X[num_cols])
                        print(f"[CtxPipe Adapter] Applied QuantileTransformer")

                # Feature engineering.
                elif "numericdata" in step_name_lower:
                    # Keep numeric columns only.
                    if len(cat_cols) > 0:
                        X = X.drop(columns=cat_cols)
                        cat_cols = []
                    print(f"[CtxPipe Adapter] Applied NumericData (dropped cat cols)")

                elif "interactionfeatures" in step_name_lower:
                    # Add interaction features (simplified; numeric columns only).
                    if len(num_cols) >= 2:
                        from itertools import combinations
                        for c1, c2 in list(combinations(num_cols, 2))[:5]:  # cap the number
                            X[f"{c1}_{c2}_mult"] = X[c1] * X[c2]
                        print(f"[CtxPipe Adapter] Applied InteractionFeatures")

                elif "pca" in step_name_lower:
                    from sklearn.decomposition import PCA
                    if len(num_cols) > 0:
                        n_components = min(len(num_cols), X.shape[0] - 1, 10)
                        if n_components > 0:
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(X[num_cols])
                            pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index)
                            X = X.drop(columns=num_cols)
                            X = pd.concat([X, pca_df], axis=1)
                            num_cols = list(pca_df.columns)
                        print(f"[CtxPipe Adapter] Applied PCA")

                elif "randomtreesembedding" in step_name_lower:
                    from sklearn.ensemble import RandomTreesEmbedding
                    if len(num_cols) > 0:
                        rte = RandomTreesEmbedding(n_estimators=10, max_depth=3, random_state=42, sparse_output=False)
                        rte_result = rte.fit_transform(X[num_cols].fillna(0))
                        # Cap the number of features.
                        n_features = min(rte_result.shape[1], 20)
                        rte_df = pd.DataFrame(rte_result[:, :n_features], columns=[f"RTE{i+1}" for i in range(n_features)], index=X.index)
                        X = X.drop(columns=num_cols)
                        X = pd.concat([X, rte_df], axis=1)
                        num_cols = list(rte_df.columns)
                        print(f"[CtxPipe Adapter] Applied RandomTreesEmbedding")

                elif "truncatedsvd" in step_name_lower:
                    from sklearn.decomposition import TruncatedSVD
                    if len(num_cols) > 0:
                        n_components = min(len(num_cols) - 1, X.shape[0] - 1, 10)
                        if n_components > 0:
                            svd = TruncatedSVD(n_components=n_components)
                            svd_result = svd.fit_transform(X[num_cols].fillna(0))
                            svd_df = pd.DataFrame(svd_result, columns=[f"SVD{i+1}" for i in range(n_components)], index=X.index)
                            X = X.drop(columns=num_cols)
                            X = pd.concat([X, svd_df], axis=1)
                            num_cols = list(svd_df.columns)
                        print(f"[CtxPipe Adapter] Applied TruncatedSVD")

                elif "polynomialfeatures" in step_name_lower:
                    from sklearn.preprocessing import PolynomialFeatures
                    if len(num_cols) > 0 and len(num_cols) <= 5:  # cap to avoid feature explosion
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        poly_result = poly.fit_transform(X[num_cols].fillna(0))
                        poly_names = [f"poly_{i}" for i in range(poly_result.shape[1])]
                        poly_df = pd.DataFrame(poly_result, columns=poly_names, index=X.index)
                        X = X.drop(columns=num_cols)
                        X = pd.concat([X, poly_df], axis=1)
                        num_cols = poly_names
                        print(f"[CtxPipe Adapter] Applied PolynomialFeatures")

                elif "variancethreshold" in step_name_lower:
                    from sklearn.feature_selection import VarianceThreshold
                    if len(num_cols) > 0:
                        selector = VarianceThreshold(threshold=0.01)
                        try:
                            X_selected = selector.fit_transform(X[num_cols].fillna(0))
                            selected_cols = [num_cols[i] for i in selector.get_support(indices=True)]
                            X = X.drop(columns=num_cols)
                            selected_df = pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
                            X = pd.concat([X, selected_df], axis=1)
                            num_cols = selected_cols
                            print(f"[CtxPipe Adapter] Applied VarianceThreshold")
                        except:
                            print(f"[CtxPipe Adapter] VarianceThreshold failed, skipping...")

                else:
                    print(f"[CtxPipe Adapter] Unknown step: {step_name}, skipping...")

            except Exception as e:
                print(f"[CtxPipe Adapter] Error applying {step_name}: {e}, skipping...")

        # Merge features and label.
        result_df = X.copy()
        result_df[label_col] = y.values

        # Rearrange columns so that the label stays in its original position.
        cols = list(result_df.columns)
        cols.remove(label_col)
        cols.insert(label_index, label_col)
        result_df = result_df[cols]

        return result_df


if __name__ == "__main__":
    # Quick test.
    adapter = CtxPipeAdapter(model_tag="ctx_50000")
    print("[Test] CtxPipeAdapter initialized successfully.")
