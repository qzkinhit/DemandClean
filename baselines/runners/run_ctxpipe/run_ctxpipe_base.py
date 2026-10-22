"""
CtxPipe runner (DemandClean-Benchmark integration)

Features:
- RL mode: context-aware, pretrained-weight inference that generates a data preparation pipeline (columns may change).
- Schema-preserving mode (--schema_preserving): imputation only (numeric=median, categorical=mode), columns preserved, suitable for traditional evaluation.

Compatibility:
- The GTE model is loaded from the `GTE_MODEL_PATH` env var first; when no model is available the code falls back to zero vectors (runs, but lower quality).
- Pretrained weights are always mapped onto the CPU at load time.
- Falls back to single-process mode when multiprocessing is unavailable.
"""
import os
import sys
import argparse
import time
import logging
import pandas as pd

# Add the project root to sys.path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from tools.insert_null import inject_missing_values
from ctxpipe_adapter import CtxPipeAdapter


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and a file."""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear any existing handlers
    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    # Set up the CLI.
    parser = argparse.ArgumentParser(
        description='Run CtxPipe - Context-aware Data Preparation Pipeline Construction.'
    )

    # CLI arguments.
    parser.add_argument(
        '--dirty_path', type=str,
        default='../../Data/beers/dirty_index.csv',
        help='Path to the input dirty CSV file.'
    )
    parser.add_argument(
        '--clean_path', type=str,
        default='../../Data/beers/clean_index.csv',
        help='Path to the input clean CSV file (for evaluation).'
    )
    parser.add_argument(
        '--task_name', type=str,
        default='beers_ctxpipe',
        help='Task name for the pipeline construction process.'
    )
    parser.add_argument(
        '--output_path', type=str,
        default='../../results/ctxpipe/',
        help='Path to save the output results.'
    )
    parser.add_argument(
        '--index_attribute', type=str,
        default='index',
        help='Index attribute of data'
    )
    parser.add_argument(
        '--label_index', type=int,
        default=None,
        help='Label column index (0-based). If not specified, will auto-detect.'
    )
    parser.add_argument(
        '--model_tag', type=str,
        default='ctx_50000',
        help='CtxPipe pretrained model tag (default: ctx_50000)'
    )
    parser.add_argument(
        '--schema_preserving', action='store_true',
        help='Output schema-preserving cleaned data (imputation only). Skips RL pipeline output to keep same columns.'
    )

    # Evaluation arguments.
    parser.add_argument(
        '--label_column', type=str, default='style',
        help='Label column name (for downstream task evaluation)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print verbose information'
    )
    parser.add_argument(
        '--use_split', action='store_true',
        help='Use the DemandClean-aligned 60/20/20 split (seed=42)'
    )

    # Parse arguments.
    args = parser.parse_args()
    stra_path = os.path.join(args.output_path, f"{args.task_name}")
    index_attribute = args.index_attribute

    # Create the output directory.
    if not os.path.exists(stra_path):
        os.makedirs(stra_path)

    # Set up logging.
    logger = setup_logging(stra_path, args.task_name)

    # Normalize the null representation to "empty".
    logger.info("Preprocessing missing values...")
    inject_missing_values(
        csv_file=args.clean_path,
        output_file=args.clean_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )
    inject_missing_values(
        csv_file=args.dirty_path,
        output_file=args.dirty_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Running CtxPipe with dirty file: {args.dirty_path}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Model tag: {args.model_tag}")
    logger.info(f"{'='*60}\n")

    # Record the start time.
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Run start time: {start_datetime}")

    # Create the CtxPipe adapter.
    adapter = CtxPipeAdapter(model_tag=args.model_tag)

    # Run CtxPipe to generate the pipeline.
    try:
        cleaned_data, ctxpipe_results = adapter.run_ctxpipe(
            dirty_path=args.dirty_path,
            task_name=args.task_name,
            label_index=args.label_index,
            output_path=stra_path,
            schema_preserving=args.schema_preserving
        )
    except Exception as e:
        logger.error(f"\n{'!'*60}")
        logger.error(f"Error running CtxPipe: {e}")
        logger.error(f"{'!'*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Record the end time.
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save the CtxPipe-processed data.
    # Post-processing: normalize nulls to "empty" (NaN, empty string, etc.).
    cleaned_data = cleaned_data.fillna('empty')
    cleaned_data = cleaned_data.replace('', 'empty')

    res_path = os.path.join(stra_path, f"{args.task_name}_cleaned.csv")
    cleaned_data.to_csv(res_path, index=False)
    logger.info(f"\nCtxPipe output saved to: {res_path}")

    # Save the CtxPipe pipeline info.
    pipeline_info_path = os.path.join(stra_path, f"{args.task_name}_pipeline_info.txt")
    with open(pipeline_info_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CtxPipe Pipeline Information\n")
        f.write("="*60 + "\n\n")
        f.write(f"Task Name: {args.task_name}\n")
        f.write(f"Model Tag: {args.model_tag}\n")
        f.write(f"Execution Time: {elapsed_time:.2f} seconds\n\n")
        f.write(f"Selected AI Sequence:\n")
        for i, primitive in enumerate(ctxpipe_results.get('ai_sequence', []), 1):
            f.write(f"  {i}. {primitive}\n")
        ml_score_val = ctxpipe_results.get('ml_score', None)
        if isinstance(ml_score_val, (int, float)):
            f.write(f"\nML Model Score: {ml_score_val:.4f}\n")
        else:
            f.write(f"\nML Model Score: N/A\n")
        f.write(f"Logical Pipeline: {ctxpipe_results.get('logical_pipeline', 'N/A')}\n")

    logger.info(f"Pipeline info saved to: {pipeline_info_path}")

    # Print CtxPipe results.
    logger.info(f"\n{'='*60}")
    logger.info("CtxPipe Results:")
    logger.info(f"{'='*60}")
    logger.info(f"AI Sequence: {ctxpipe_results.get('ai_sequence', [])}")
    ml_score_val = ctxpipe_results.get('ml_score', None)
    if isinstance(ml_score_val, (int, float)):
        logger.info(f"ML Score: {ml_score_val:.4f}")
    else:
        logger.info("ML Score: N/A")
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")
    logger.info(f"{'='*60}\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"CtxPipe finished successfully!")
    logger.info(f"Results saved to: {stra_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
