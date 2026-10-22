"""
UniClean wrapper: data cleaning that fuses several cleaning signals and optimizes the cleaning
workflow.

Paper: UniClean (VLDB 2025)

Ground truth used: none (fully automatic).
Dependencies: pyspark>=3.1.1
The wrapper calls the official implementation only.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Put this directory on the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def _check_pyspark():
    """Check that PySpark is installed."""
    try:
        from pyspark.sql import SparkSession
        return True
    except ImportError:
        raise ImportError(
            "UniClean requires PySpark. Install it with: pip install pyspark==3.1.1\n"
            "See Methods/UniClean/requirements.txt"
        )


def _check_clean_module():
    """Check that the Clean module exists."""
    try:
        from Clean import CleanonLocalWithnoSmple
        return True
    except ImportError:
        raise ImportError(
            "UniClean requires the Clean.py module.\n"
            "Ensure Methods/UniClean/Clean.py exists and contains CleanonLocalWithnoSmple function"
        )


class UniCleanWrapper:
    """
    Wrapper of UniClean.

    1. fusion of cleaning signals (constraints, statistics, patterns)
    2. PySpark for large tables
    3. configurable cleaners

    The official implementation needs PySpark, CleanonLocalWithnoSmple of Clean.py, and an
    index column in the data.
    """

    def __init__(self,
                 cleaners: List = None,
                 single_max: int = 10000,
                 batch_size: int = 500,
                 executor_memory: str = "8g",
                 driver_memory: str = "8g",
                 verbose: bool = False):
        """
        Args:
            cleaners: list of cleaners (None takes the default configuration)
            single_max: maximum records per pass (official default 30000)
            batch_size: batch size (official default 500)
            executor_memory: memory of the Spark executor
            driver_memory: memory of the Spark driver
            verbose: print progress information
        """
        self.cleaners = cleaners
        self.single_max = single_max
        self.batch_size = batch_size
        self.executor_memory = executor_memory
        self.driver_memory = driver_memory
        self.verbose = verbose
        self.ground_truth_used = 0
        self.spark = None

    def _init_spark(self, app_name: str = "UniClean"):
        """Create the Spark session."""
        _check_pyspark()

        from pyspark.sql import SparkSession

        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.executor.memory", self.executor_memory) \
                .config("spark.driver.memory", self.driver_memory) \
                .config("spark.executor.memoryOverhead", "8g") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
        return self.spark

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              output_path: str = None,
              cleaners: List = None,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        Run UniClean.

        Args:
            dirty_path: path of the dirty table (CSV with an index column)
            clean_path: path of the clean table (optional)
            output_path: output path
            cleaners: list of cleaners
            index_attribute: name of the index column ('index')

        Returns:
            repaired table and run information

        Raises:
            ImportError: PySpark or the Clean module is missing
        """
        import time

        # Check the dependencies
        _check_pyspark()
        _check_clean_module()

        from pyspark.sql.functions import monotonically_increasing_id
        from Clean import CleanonLocalWithnoSmple

        # initializeSpark
        spark = self._init_spark()

        # Given cleaners or the default ones
        use_cleaners = cleaners or self.cleaners

        if not use_cleaners:
            raise ValueError(
                "UniClean requires configured cleaners (cleaners).\n"
                "Example:\n"
                "  from SampleScrubber.cleaner.single import Number\n"
                "  from SampleScrubber.cleaner.multiple import AttrRelation\n"
                "  cleaners = [\n"
                "      Number('ounces', name='Number_ounces'),\n"
                "      AttrRelation(['brewery_id'], ['brewery_name'], '0')\n"
                "  ]"
            )

        try:
            # Read data
            data = spark.read.csv(dirty_path, header=True, inferSchema=True)

            if index_attribute not in data.columns:
                data = data.withColumn(index_attribute, monotonically_increasing_id())
            data.persist()

            # Run cleaning
            start_time = time.perf_counter()

            # Create a temporary output directory
            if output_path:
                table_path = os.path.dirname(output_path)
            else:
                table_path = os.path.join(_current_dir, 'temp_output')
            os.makedirs(table_path, exist_ok=True)

            if self.verbose:
                print(f"UniClean started, rows: {data.count()}")
                print(f"cleaner count: {len(use_cleaners)}")

            # InvokeofficialUniCleancleaningfunction
            cleaned_data = CleanonLocalWithnoSmple(
                spark,
                use_cleaners,
                data,
                table_path,
                batch_size=self.batch_size,
                single_max=self.single_max
            )

            elapsed_time = time.perf_counter() - start_time

            # ConverttoPandas DataFrame
            result_df = cleaned_data.toPandas()

            # Empty values become "empty"
            result_df = result_df.fillna('empty')
            result_df = result_df.replace('', 'empty')

            if output_path:
                result_df.to_csv(output_path, index=False)

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'UniClean',
                'type': 'data-oriented',
                'auto_level': 1,
                'cleaners_count': len(use_cleaners),
                'elapsed_time': elapsed_time
            }

            return result_df, info

        finally:
            if self.spark:
                self.spark.stop()
                self.spark = None

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used (always 0, UniClean is fully automatic)."""
        return self.ground_truth_used


def uniclean_clean(dirty_path: str,
                   cleaners: List,
                   output_path: str = None,
                   **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function of the UniClean cleaner.

    Args:
        dirty_path: path of the dirty table
        cleaners: list of cleaners (required)
        output_path: output path
        **kwargs: passed to UniCleanWrapper

    Returns:
        repaired table and run information
    """
    wrapper = UniCleanWrapper(cleaners=cleaners, **kwargs)
    return wrapper.clean(dirty_path, output_path=output_path)


# Predefined cleaner configurations (examples)
def get_beers_cleaners():
    """
    Cleaners of the beers dataset.

    Example:
        cleaners = get_beers_cleaners()
        df, info = uniclean_clean('data/beers/dirty_index.csv', cleaners)
    """
    from SampleScrubber.cleaner.single import Number
    from SampleScrubber.cleaner.multiple import AttrRelation

    return [
        Number("ounces", name="Number_ounces"),
        Number("abv", name="Number_abv"),
        AttrRelation(["brewery_id"], ["brewery_name"], '0'),
        AttrRelation(["brewery_id"], ["city"], '1'),
        AttrRelation(["brewery_id"], ["state"], '2')
    ]


def get_hospitals_cleaners():
    """
    Cleaners of the hospitals dataset.

    Example:
        cleaners = get_hospitals_cleaners()
        df, info = uniclean_clean('data/hospitals/dirty_index.csv', cleaners)
    """
    from SampleScrubber.cleaner.single import Number
    from SampleScrubber.cleaner.multiple import AttrRelation

    return [
        Number("Score", name="Number_Score"),
        AttrRelation(["ZIP Code"], ["City"], '0'),
        AttrRelation(["ZIP Code"], ["State"], '1'),
        AttrRelation(["ZIP Code"], ["County Name"], '2')
    ]


def load_cleaners_from_rules(rules_path: str):
    """
    Load the UniClean cleaners of a rule file.

    Args:
        rules_path: rule file (for example data/beers/rules_baselines.txt)

    Returns:
        list of cleaners

    Example:
        cleaners = load_cleaners_from_rules('data/beers/rules_baselines.txt')
        wrapper = UniCleanWrapper(cleaners=cleaners)
        df, info = wrapper.clean(dirty_path)
    """
    from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
    from SampleScrubber.cleaner.multiple import AttrRelation

    cleaners = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"rule file not found: {rules_path}")

    # Parse the rule file
    in_uniclean_section = False
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith('#'):
                continue

            # detectionsection
            if line.startswith('[') and line.endswith(']'):
                in_uniclean_section = (line == '[UNICLEAN]')
                continue

            # Cleaner definitions of the UNICLEAN section
            if in_uniclean_section:
                local_vars = {
                    'Number': Number,
                    'Pattern': Pattern,
                    'Outlier': Outlier,
                    'Date': Date,
                    'AttrRelation': AttrRelation
                }
                try:
                    cleaner = eval(line, {"__builtins__": {}}, local_vars)
                    cleaners.append(cleaner)
                except Exception as e:
                    print(f"Warning: failed to parse cleaner '{line}': {e}")

    return cleaners


def get_cleaners_for_dataset(dataset_name: str, data_dir: str = 'Data'):
    """
    Load the cleaners of a dataset from its rule file.

    Args:
        dataset_name: dataset name (for example 'beers', 'adult')
        data_dir: root directory of the data

    Returns:
        list of cleaners
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return load_cleaners_from_rules(rules_path)
