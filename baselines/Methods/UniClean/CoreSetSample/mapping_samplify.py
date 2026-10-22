import os
import shutil
import time

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pyspark.sql.window import Window


def Generate_Sample(TotalData, sset, tset, models=None, save_path=None, single_max=30000000):
    """
    Draw a sample of a Spark dataset and store it as a pandas DataFrame.

    Parameters:
    - file_load: path of the CSV file
    - sset: source attributes, for example ['zip']
    - tset: target attributes, for example ['city']
    - p: share of the dataset in the core set
    - save_path: output path, None for no output
    """

    # # sampling
    start_time = time.time()
    # Read the required columns only
    required_columns = list(set(sset + tset + ['index']))  # Drop duplicates
    data = TotalData.select(required_columns)
    sampled_data, error_threshold = block_sample(data, models, single_max=single_max)
    # Save the data
    if save_path:
        block_path = os.path.join(save_path, "sample_data.csv")
        sampled_data.write.csv(block_path, header=True, mode='overwrite')
        print(f"Sample data saved to: {block_path}")

    # # time the sampling
    print(f"core sample drawn in {time.time() - start_time} s")

    return sampled_data,error_threshold


def block_sample(df, models, single_max=30000):
    error_threshold = 1
    sample_probability_columns=[]
    for i, model in enumerate(models):
        sourceSet = model.source
        targetSet = model.target
        group_columns = list(sourceSet.union(targetSet))
        # null becomes '__NULL__' so that groupBy counts it
        for col in group_columns:
            # df = df.withColumn(col, F.when(F.col(col).isNull(), F.lit("__NULL__")).otherwise(F.col(col)))
            df = df.withColumn(col, F.when((F.col(col).isNull()) | (F.col(col) == "empty"), F.lit("__NULL__")).otherwise(
            F.col(col)))
        # Aggregate every source-target combination (count and target_count)
        rule_grouped = df.groupBy(*group_columns).count()
        window_spec_partition = Window.partitionBy(*sourceSet)
        rule_grouped = rule_grouped.withColumn(f"target_total_{i}", F.sum("count").over(window_spec_partition)) \
            .withColumn(f"target_count_{i}", F.col("count"))
        # Maximum over the values other than __NULL__
        rule_grouped = rule_grouped.withColumn(f"max_target_count_{i}",
                                               F.max(F.when(F.col(list(targetSet)[0]) != '__NULL__', F.col("count"))).over(
                                                   window_spec_partition))

        # rule_grouped = rule_grouped.withColumn(f"target_total_{i}", F.sum("count").over(window_spec_partition)) \
        #     .withColumn(f"target_count_{i}", F.col("count")).withColumn(f"max_target_count_{i}",
        #                                                                 F.max(F.col("count")).over(
        #                                                                     window_spec_partition))
        # Share of the most frequent target of every source
        rule_grouped = rule_grouped.withColumn(f"max_target_rate_{i}", F.col(f"max_target_count_{i}") / F.col(f"target_total_{i}"))
        # Minimum share
        error_threshold = min(rule_grouped.agg(F.min(f"max_target_rate_{i}")).collect()[0][0],error_threshold)

        # rank_total_i and rank_count_i
        rule_grouped = rule_grouped.withColumn(f"rank_count_{i}", F.dense_rank().over(
            Window.partitionBy(*sourceSet).orderBy(F.desc("count")))) \
            .withColumn(f"rank_total_{i}", F.sum(F.col(f"rank_count_{i}")).over(window_spec_partition))

        # Join with the original data frame
        df = df.join(
            rule_grouped.select(*group_columns, f"target_total_{i}", f"target_count_{i}", f"rank_total_{i}",
                                f"rank_count_{i}"), group_columns, "left")
        # Sampling probability from the condition function
        # UDF
        if 'condition_func' in model.fixValueRules:
            condition_func_udf = F.udf(lambda *source_attrs: model.fixValueRules['condition_func'](tuple(source_attrs)), BooleanType())

            # Condition column
            condition_col = condition_func_udf(*[F.col(col) for col in sourceSet])
            sample_probability_column = F.when(condition_col, 1.0).otherwise(F.col(f"target_count_{i}") / F.col(f"target_total_{i}"))
        else:
            sample_probability_column = F.col(f"target_count_{i}") / F.col(f"target_total_{i}")

        # Final sampling probability of every row from the information of every rule
        sample_probability_columns.append(sample_probability_column)

    final_sample_probability = sample_probability_columns[0]
    for col in sample_probability_columns[1:]:
        final_sample_probability *= col
    df = df.withColumn("p", final_sample_probability)
    # Deduplicate
    if len(models) == 1:
        model = models[0]
        sourceSet = model.source
        targetSet = model.target
        # Group by sourceSet and targetSet
        group_columns = list(sourceSet.union(targetSet))

        # Target distribution of every source combination
        grouped = df.groupBy(*group_columns).count()

        # Number of distinct targets of every source combination
        window_spec_partition = Window.partitionBy(*sourceSet)
        targetSetlist = list(targetSet)
        grouped = grouped.withColumn("target_count", F.count(targetSetlist[0]).over(window_spec_partition))

        # target_count > 1: sort and set the sample count, otherwise sample_count is 0
        window_spec_rank = Window.partitionBy(*sourceSet).orderBy("count")
        grouped = grouped.withColumn("rank",
                                     F.when(F.col("target_count") > 1, F.dense_rank().over(window_spec_rank)).otherwise(
                                         0)) \
            .withColumn("sample_count", F.when(F.col("rank") > 0, F.col("rank")).otherwise(0))

        # Deduplicate by the sample_count computed above
        window_spec_target = Window.partitionBy(*group_columns).orderBy(F.rand())
        df = df.join(grouped, group_columns) \
            .withColumn("row_num", F.row_number().over(window_spec_target)) \
            .filter(F.col("row_num") <= F.col("sample_count")) \
            .drop("count", "target_count", "rank", "sample_count", "row_num")
    else:
        attr_set = set()
        for model in models:
            attr_set = attr_set.union(model.source)
            attr_set = attr_set.union(model.target)
        group_columns = list(attr_set)
        # Distribution of every combination
        grouped = df.groupBy(*group_columns).count()

        # Sort the blocks by size, dense_rank gives equal counts the same rank
        window_spec_partition = Window.orderBy(F.asc("count"))
        grouped = grouped.withColumn("rank", F.dense_rank().over(window_spec_partition))
        grouped = grouped.withColumn("sample_count", F.col("rank"))

        # Draw the sample
        window_spec_target = Window.partitionBy(*group_columns).orderBy(F.rand())
        df = df.join(grouped, group_columns) \
            .withColumn("row_num", F.row_number().over(window_spec_target)) \
            .filter(F.col("row_num") <= F.col("sample_count")) \
            .drop("count", "rank", "sample_count", "row_num")

    # Sampling
    df = df.filter((F.col("p") < 1.0))
    # Draw the sample
    df = df.withColumn("rand_val", F.rand())
    # sample_rate, at least 1
    # Non-null values per column
    print("size of the inconsistent data: ")
    count=df.count()
    print(count)
    if df.count() != 0:
        if len(models) == 1:
            if count > single_max:
                model = models[0]
                sourceSet = model.source
                # Sort by several columns and take the first single_max rows
                df = df.orderBy([F.col(col) for col in sourceSet]).limit(single_max)
        else:
            if count > single_max:
                # Step 1: source attribute sets of the models
                source_columns = []
                for model in models:
                    source_columns.append(model.source)  # model.source is an attribute set
                # Step 2: attribute set with the fewest distinct value combinations
                min_count = float("inf")  # Initial maximum
                min_source_set = None  # Attribute set with the fewest combinations
                for source_set in source_columns:
                    # Distinct value combinations of every attribute set
                    distinct_count = df.select([F.col(col) for col in source_set]).distinct().count()
                    print(f"distinct value combinations of {source_set}: {distinct_count}")
                    # Fewest combinations
                    if distinct_count < min_count:
                        min_count = distinct_count
                        min_source_set = source_set

                # Step 3: process the attribute set with the least data
                if min_source_set:
                    print(f"attribute set with the least data: {min_source_set}, count: {min_count}")
                    # Sort by this attribute set and take the first single_max rows
                    df = df.orderBy([F.col(col) for col in min_source_set]).limit(single_max)
                # Filter by p and rand_val
            df = df.filter(
                    (F.col("p") < error_threshold) | (F.col("rand_val") <= F.col("p"))
                )
    # Drop the temporary columns
    sampled_data = df.drop("rand_val", *[f"target_total_{i}" for i in range(len(models))],
                           *[f"target_count_{i}" for i in range(len(models))], "final_sample_probability",
                           *[f"rank_total_{i}" for i in range(len(models))],
                           *[f"rank_count_{i}" for i in range(len(models))])
    return sampled_data, error_threshold

def save_sample(df: DataFrame, output_path: str, csv_filename: str, temp_folder: str):
    """
    Save a DataFrame as CSV and copy it to the target folder.

    :param df: DataFrame to save
    :param output_path: target folder
    :param csv_filename: name of the CSV file
    :param temp_folder: temporary folder
    """
    # Save the CSV with one partition to get a single file
    df.repartition(1).write.mode('overwrite').option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
        .option("header", "true").csv(path=os.path.join(temp_folder, "sampleddata"), encoding="utf-8")

    # Copy the produced CSV file to the target folder under its name
    for filename in os.listdir(os.path.join(temp_folder, "sampleddata")):
        if filename.startswith("part-") and filename.endswith(".csv"):
            full_file_name = os.path.join(temp_folder, "sampleddata", filename)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(output_path, csv_filename))
                break
