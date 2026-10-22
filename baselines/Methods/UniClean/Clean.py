import os
import time
from pyspark.sql import functions as F

from AnalyticsCache.cleaner_associations_cycle import associations_classier, discover_cleaner_associations, \
    PreParamClassier
from AnalyticsCache.get_cleaner_excute_info import sort_opinfo_by_weights, generate_plantuml_corrected, \
    getSingle_opinfo, cleaner_grouping
# from AnalyticsCache.get_planuml_graph import PlantUML
from AnalyticsCache.handle_rules import getOpWeights, transformRulesToSpark, transformKnowledgeToSpark
from CoreSetSample.mapping_samplify import Generate_Sample
from SampleScrubber.cleaner_model import NOOP
from SampleScrubber.param_selector import customclean


def CleanonLocal(spark, cleanners, data, table_path, batch_size=500, single_max=30000):
    current = 'Tmp1'
    print("Initializing the cleaners and analysing their dependencies...")
    Edges, singles, multis = PreParamClassier(cleanners)

    print("Single-attribute cleaning operators:")
    for single in singles:
        print(f"single-attribute cleaning operator: {single}")
        # Domain of the single attribute (the "domain" attribute is the target of the operator)
        target_column = single.domain

        # Select the target column of the Spark DataFrame as a pandas DataFrame
        sample_df = data.select(target_column).distinct().toPandas()

        if sample_df.empty:
            print(f"attribute {target_column} has no data, skipped.")
            continue

        print(f"sampled block of attribute {target_column}: {sample_df.shape[0]} rows")

        # Clean with customclean
        preCleaners = [single]
        _, output, editRuleList, _ = customclean(sample_df, precleaners=preCleaners)

        if editRuleList:
            # Apply the cleaning rules to the Spark DataFrame
            print(f"applying the rules of {target_column}, edit rules: {len(editRuleList)}")
            data, _ = transformRulesToSpark(editRuleList, data, batch_size=batch_size)
        else:
            print(f"attribute {target_column} has no applicable rule.")

    source_sets, target_sets, explain, processing_order, plt = discover_cleaner_associations(Edges)
    print("Execution levels and target models...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("Execution levels (parallel groups):", levels)
    print("Target models:", models)

    editRuleDict = {}  # Provenance weights
    editRules = NOOP()  # No-op rule
    nextClean = True

    level_clean_status = {}
    clean_attempts = {}  # Number of cleaning passes per node

    while nextClean:
        nextClean = False

        for level_index, level in enumerate(nodes):
            # Initialize the state of a level not seen before
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in clean_attempts:
                clean_attempts[level_index] = {node: 0 for node in level}

            print(f"\nlevel {level_index + 1}, nodes: {level}")
            editRuleLists = []  # Cleaning operations of the level

            for node in level:
                # Only the nodes of the level for which no operation was found yet
                if level_clean_status[level_index][node]:
                    print(f"  skipping finished node: {node}")
                    continue

                clean_attempts[level_index][node] += 1  # Count the pass

                if clean_attempts[level_index][node] > 3:  # Finished after more than 3 passes
                    print(f"  node {node} exceeded the pass limit, marked finished")
                    level_clean_status[level_index][node] = True
                    continue

                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  node {node} has cleaning signals")
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)
                    tset = list(tset)
                    print(f"  sampling: source attributes {sset} -> target attribute {node}")
                    sample_Block_df, error_threshold = Generate_Sample(data, sset, tset, models=models[node])
                    sample_df = sample_Block_df.toPandas()
                    print(f"block size: {sample_df.shape[0]}")
                    if sample_df.shape[0] < single_max:
                        preCleaners = [single for single in singles if single.domain in (sset + [node])]
                        _, output, _, _ = customclean(sample_df, precleaners=preCleaners)
                        # Weight the operators of the block
                        blockModels = None
                        weights = [1] * len(models[node])
                        for model, weight in zip(models[node], weights):
                            if blockModels is None:
                                blockModels = model * weight
                            else:
                                blockModels += model * weight
                        partition = [list(model.source) for model in models[node]]
                        print('blocking attributes')
                        print(partition)
                        editRule1, output, editRuleList, _ = customclean(output, cleaners=[blockModels],
                                                                         partition=partition,
                                                                         error_threshold=error_threshold)
                        editRule *= editRule1
                    else:
                        print(f"block size above the threshold, models run one by one")
                        for m in models[node]:
                            print(f"model: {m}")
                            # Source and target attributes of the model
                            sset = list(m.source)
                            tset = list(m.target)
                            sample_Block_df, error_threshold = Generate_Sample(data, sset, tset, models=[m],
                                                                               single_max=single_max)
                            sample_df = sample_Block_df.toPandas()
                            if sample_df.shape[0] == 0:
                                print(f"no data after model {m}")
                                continue
                            preCleaners = [single for single in singles if single.domain in (sset + [node])]
                            _, output, _, _ = customclean(sample_df, precleaners=preCleaners)
                            blockModels = m  # A single model
                            partition = [list(m.source)]
                            editRule1, output, editRuleList1, _ = customclean(output, cleaners=[blockModels],
                                                                              partition=partition,
                                                                              error_threshold=error_threshold)
                            editRule *= editRule1
                            editRuleList.extend(editRuleList1)
                else:
                    print(f"  node {node} has no cleaning signal")

                editRuleLists.extend(editRuleList)
                editRules *= editRule

                if len(editRuleList) == 0:
                    print(f"  node {node} has no cleaning operation")
                    level_clean_status[level_index][node] = True
                else:
                    print(f"  node {node} has {len(editRuleList)} cleaning operations")
                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # Reset to a new list if it was not one
                else:
                    editRuleDict[str(node)] = editRuleList  # New key with a new list

            print(f"cleaning of {level} finished")
            print("\nEdit rules accumulated in this level:", str(len(editRuleLists)))

            if editRuleLists:  # Edit rules present
                nextClean = True
                # data, _ = transformRulesToSpark(editRuleLists, data, batch_size=batch_size)
                # print("writing the data")
                # path = table_path + 'Tmp'
                # data.coalesce(1).write.mode('overwrite').csv(path, header=True)
                # print("rows written: " + str(data.count()))
                # os.system('sync')  # flush caches
                # data.unpersist()
                # spark.catalog.clearCache()
                # print("reading the latest cleaned data", str(path))
                # data = spark.read.csv(path, header=True, inferSchema=True)
                # print("rows read: " + str(data.count()))
                # data.persist()
                # print(f"data changed, cleaning node {level} again")
                while True:
                    # Handle the rules in batches
                    data, remaining_rules = transformRulesToSpark(editRuleLists, data, batch_size=batch_size, maxhandle=3000)
                    print("Writing the data...")
                    # Temporary path
                    if current == 'Tmp1':
                        path = table_path + 'Tmp2'
                        current = 'Tmp2'
                    else:
                        path = table_path + 'Tmp1'
                        current = 'Tmp1'
                    # Write the data to the file system
                    data.coalesce(1).write.mode('overwrite').csv(path, header=True)
                    print("rows written: " + str(data.count()))
                    os.system('sync')  # Flush caches
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("reading the latest cleaned data", str(path))
                    data = spark.read.csv(path, header=True, inferSchema=True)
                    print("rows read: " + str(data.count()))
                    data.persist()
                    # print(f"data changed, cleaning node {level} again")

                    # Stop when no rule is left
                    if not remaining_rules:
                        print("All rules applied")
                        break

                    print(f"{len(remaining_rules)} cleaning operations left, continuing")
                    editRuleLists = remaining_rules  # Continue with the remaining rules

                print(f"data changed, cleaning node {level} again")
                break
            else:
                print("No cleaning operation in this level, next level")
    print("\nProvenance of the cleaning rules:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    single_opinfo = getSingle_opinfo(singles)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print("\nCleaning workflow:")
    sorted_dict = sort_opinfo_by_weights(grouped_opinfo, group_weights)

    # OpPlantuml = generate_plantuml_corrected(single_opinfo, grouped_opinfo, sorted_dict)
    # PlantUML().process_str(OpPlantuml, directory=table_path)
    # plt.savefig(rf"{table_path}/acplt.png", format="png")
    # print("association graph saved to:", rf"{table_path}/acplt.png")
    # print("\ncleaning workflow diagram:", "CleanerPlantuml.svg")

    return data


def CleanonCloud(spark, cleaners, data, table_name, database_name, batch_size=300, single_max=20000, maxhandle=1000):
    # Initialize and analyse the cleaners
    print("Initializing the cleaners and analysing their dependencies...")
    Edges, singles, multis = PreParamClassier(cleaners)
    source_sets, target_sets, explain, processing_order = discover_cleaner_associations(Edges)
    print("Execution levels and target models...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("Execution levels (parallel groups):", levels)
    print("Target models:", models)

    editRuleDict = {}  # Provenance weights
    editRules = NOOP()  # No-op rule
    nextClean = True
    tmp_table1 = f"{database_name}.{table_name}_tmp1"
    tmp_table2 = f"{database_name}.{table_name}_tmp2"
    current_tmp_table = tmp_table1
    level_clean_status = {}  # Cleaning state of the nodes of each level
    level_knowledge_status = {}  # Has the knowledge base been applied

    while nextClean:
        nextClean = False
        for level_index, level in enumerate(nodes):
            # Initialize the state of a level not seen before
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in level_knowledge_status:
                level_knowledge_status[level_index] = {node: False for node in level}

            print(f"\nlevel {level_index + 1}, nodes: {level}")
            editRuleLists = []  # Cleaning operations of the level

            for node in level:
                # Only the unfinished nodes of the level
                if level_clean_status[level_index][node]:
                    print(f"  skipping finished node: {node}")
                    continue

                # Cleaning operations of the current node
                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  node {node} has cleaning signals")

                    # Knowledge base applied already
                    if not level_knowledge_status[level_index][node]:
                        # **transformKnowledgeToSpark is called here**
                        for cleaner in models[node]:
                            data = transformKnowledgeToSpark(spark, database_name, table_name, data, cleaner)

                        # **write and read the data**
                        print("Writing the data...")
                        next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                        data.write.mode('overwrite').saveAsTable(next_tmp_table)
                        print(f"rows written: {data.count()}")
                        os.system('sync')  # Flush caches
                        data.unpersist()
                        spark.catalog.clearCache()
                        print("Reading the latest cleaned data...")
                        data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                        print(f"rows read: {data.count()}")
                        data.persist()
                        current_tmp_table = next_tmp_table
                        level_knowledge_status[level_index][node] = True

                    # Empty set
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)  # As a list
                    tset = list(tset)
                    print(f"  sampling: source attributes {sset} -> target attribute {node}")
                    sample_Block_df, error_threshold = Generate_Sample(
                        data, sset, tset, models=models[node], single_max=single_max
                    )
                    sample_df = sample_Block_df.toPandas()
                    print(f"block size: {sample_df.shape[0]}")

                    preCleaners = [single for single in singles if
                                   any(attr_set in sset + [node] for attr_set in single.domain)]
                    _, output, _, _ = customclean(sample_df, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight
                    partition = [list(model.source) for model in models[node]]
                    editRule1, output, editRuleList, _ = customclean(
                        output, cleaners=[blockModels], partition=partition, error_threshold=error_threshold
                    )
                    editRule *= editRule1
                else:
                    print(f"  node {node} has no cleaning signal")

                editRuleLists.extend(editRuleList)
                editRules *= editRule  # Accumulate the applied rules

                # A node without applicable operations needs no further pass
                if not editRuleList:
                    level_clean_status[level_index][node] = True  # Mark the node finished

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # Reset to a new list if it was not one
                else:
                    editRuleDict[str(node)] = editRuleList  # New key with a new list

            print(f"cleaning of {level} finished")
            print("\nEdit rules accumulated in this level:", str(len(editRuleLists)))

            if editRuleLists:  # Edit rules present
                nextClean = True
                while True:
                    # Process the data, returns the data and the remaining rules
                    data, remaining_rules = transformRulesToSpark(
                        editRuleLists, data, batch_size=batch_size, maxhandle=maxhandle
                    )
                    print("Writing the data...")
                    next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                    data.write.mode('overwrite').saveAsTable(next_tmp_table)
                    print(f"rows written: {data.count()}")
                    os.system('sync')  # Flush caches
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("Reading the latest cleaned data...")
                    data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                    print(f"rows read: {data.count()}")
                    data.persist()
                    current_tmp_table = next_tmp_table

                    if not remaining_rules:
                        print("All rules applied")
                        break  # Every rule applied, stop
                    print(f"{len(remaining_rules)} cleaning operations left, continuing")
                    editRuleLists = remaining_rules  # Continue with the remaining rules

                print(f"data changed, cleaning node {level} again")
                # Clean this level again instead of moving to the next one
                break
            else:
                print("No cleaning operation in this level, next level")

    print("\nProvenance of the cleaning rules:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print(group_weights)
    return data


def CleanonCloudRandom(spark, cleaners, data, table_name, database_name, batch_size=300, single_max=20000,
                       maxhandle=1000):
    # Initialize and analyse the cleaners
    print("Initializing the cleaners and analysing their dependencies...")
    Edges, singles, multis = PreParamClassier(cleaners)
    source_sets, target_sets, explain, processing_order = discover_cleaner_associations(Edges)
    print("Execution levels and target models...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("Original execution levels (parallel groups):", levels)
    print("Target models:", models)

    # Reverse the levels for backward cleaning
    levels = levels[::-1]
    nodes = nodes[::-1]
    print("Reversed execution levels (parallel groups):", levels)

    editRuleDict = {}  # Provenance weights
    editRules = NOOP()  # No-op rule
    nextClean = True
    tmp_table1 = f"{database_name}.{table_name}_tmp1_random"
    tmp_table2 = f"{database_name}.{table_name}_tmp2_random"
    current_tmp_table = tmp_table1
    level_clean_status = {}  # Cleaning state of the nodes of each level
    level_knowledge_status = {}  # Has the knowledge base been applied

    while nextClean:
        nextClean = False
        for level_index, level in enumerate(nodes):
            # Initialize the state of a level not seen before
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in level_knowledge_status:
                level_knowledge_status[level_index] = {node: False for node in level}

            print(f"\nreversed level {level_index + 1}, nodes: {level}")
            editRuleLists = []  # Cleaning operations of the level

            for node in level:
                # Only the unfinished nodes of the level
                if level_clean_status[level_index][node]:
                    print(f"  skipping finished node: {node}")
                    continue

                # Cleaning operations of the current node
                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  node {node} has cleaning signals")

                    # Knowledge base applied already
                    if not level_knowledge_status[level_index][node]:
                        # **transformKnowledgeToSpark is called here**
                        for cleaner in models[node]:
                            data = transformKnowledgeToSpark(spark, database_name, table_name, data, cleaner)

                        # **write and read the data**
                        print("Writing the data...")
                        next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                        data.write.mode('overwrite').saveAsTable(next_tmp_table)
                        print(f"rows written: {data.count()}")
                        os.system('sync')  # Flush caches
                        data.unpersist()
                        spark.catalog.clearCache()
                        print("Reading the latest cleaned data...")
                        data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                        print(f"rows read: {data.count()}")
                        data.persist()
                        current_tmp_table = next_tmp_table
                        level_knowledge_status[level_index][node] = True

                    # Empty set
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)  # As a list
                    tset = list(tset)
                    print(f"  sampling: source attributes {sset} -> target attribute {node}")
                    sample_Block_df, error_threshold = Generate_Sample(
                        data, sset, tset, models=models[node], single_max=single_max
                    )
                    sample_df = sample_Block_df.toPandas()
                    print(f"block size: {sample_df.shape[0]}")

                    preCleaners = [single for single in singles if
                                   any(attr_set in sset + [node] for attr_set in single.domain)]
                    _, output, _, _ = customclean(sample_df, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight
                    partition = [list(model.source) for model in models[node]]
                    editRule1, output, editRuleList, _ = customclean(
                        output, cleaners=[blockModels], partition=partition, error_threshold=error_threshold
                    )
                    editRule *= editRule1
                else:
                    print(f"  node {node} has no cleaning signal")

                editRuleLists.extend(editRuleList)
                editRules *= editRule  # Accumulate the applied rules

                # A node without applicable operations needs no further pass
                if not editRuleList:
                    level_clean_status[level_index][node] = True  # Mark the node finished

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # Reset to a new list if it was not one
                else:
                    editRuleDict[str(node)] = editRuleList  # New key with a new list

            print(f"cleaning of {level} finished")
            print("\nEdit rules accumulated in this level:", str(len(editRuleLists)))

            if editRuleLists:  # Edit rules present
                nextClean = True
                while True:
                    # Process the data, returns the data and the remaining rules
                    data, remaining_rules = transformRulesToSpark(
                        editRuleLists, data, batch_size=batch_size, maxhandle=maxhandle
                    )
                    print("Writing the data...")
                    next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                    data.write.mode('overwrite').saveAsTable(next_tmp_table)
                    print(f"rows written: {data.count()}")
                    os.system('sync')  # Flush caches
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("Reading the latest cleaned data...")
                    data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                    print(f"rows read: {data.count()}")
                    data.persist()
                    current_tmp_table = next_tmp_table

                    if not remaining_rules:
                        print("All rules applied")
                        break  # Every rule applied, stop
                    print(f"{len(remaining_rules)} cleaning operations left, continuing")
                    editRuleLists = remaining_rules  # Continue with the remaining rules

                print(f"data changed, cleaning node {level} again")
                # Clean this level again instead of moving to the next one
                break
            else:
                print("No cleaning operation in this level, next level")

    print("\nProvenance of the cleaning rules:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print(group_weights)
    return data

def CleanonLocalWithnoSmple(spark, cleanners, data, table_path, batch_size=500, single_max=30000):
    current = 'Tmp1'
    print("Initializing the cleaners and analysing their dependencies...")
    Edges, singles, multis = PreParamClassier(cleanners)

    print("Single-attribute cleaning operators:")
    for single in singles:
        print(f"single-attribute cleaning operator: {single}")
        # Domain of the single attribute (the "domain" attribute is the target of the operator)
        target_column = single.domain

        print(f"cleaning attribute {target_column}")
        # Spark DataFrame to pandas DataFrame
        df_pandas = data.select(target_column).distinct().toPandas()

        if df_pandas.empty:
            print(f"attribute {target_column} has no data, skipped.")
            continue

        print(f"attribute {target_column}: {df_pandas.shape[0]} rows")

        # Clean with customclean
        preCleaners = [single]
        _, output, editRuleList, _ = customclean(df_pandas, precleaners=preCleaners)

        if editRuleList:
            # Apply the cleaning rules to the Spark DataFrame
            print(f"applying the rules of {target_column}, edit rules: {len(editRuleList)}")
            data, _ = transformRulesToSpark(editRuleList, data, batch_size=batch_size)
        else:
            print(f"attribute {target_column} has no applicable rule.")

        # Number cleaners: Spark cast of the string column to double
        # Numeric strings ("-107.0") become doubles, other strings ("empty") become null
        # Avoids a when-otherwise chain per value, which overflows the stack
        from SampleScrubber.cleaner.single import Number as NumberCleaner
        if isinstance(single, NumberCleaner):
            from pyspark.sql.types import DoubleType
            print(f"Spark cast(double) of attribute {target_column}")
            data = data.withColumn(target_column, F.col(target_column).cast(DoubleType()))

    source_sets, target_sets, explain, processing_order, plt = discover_cleaner_associations(Edges)
    print("Execution levels and target models...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("Execution levels (parallel groups):", levels)
    print("Target models:", models)

    editRuleDict = {}  # Provenance weights
    editRules = NOOP()  # No-op rule
    nextClean = True

    level_clean_status = {}
    clean_attempts = {}  # Number of cleaning passes per node

    while nextClean:
        nextClean = False

        for level_index, level in enumerate(nodes):
            # Initialize the state of a level not seen before
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in clean_attempts:
                clean_attempts[level_index] = {node: 0 for node in level}

            print(f"\nlevel {level_index + 1}, nodes: {level}")
            editRuleLists = []  # Cleaning operations of the level

            for node in level:
                # Only the nodes of the level for which no operation was found yet
                if level_clean_status[level_index][node]:
                    print(f"  skipping finished node: {node}")
                    continue

                clean_attempts[level_index][node] += 1  # Count the pass

                if clean_attempts[level_index][node] > 3:  # Finished after more than 3 passes
                    print(f"  node {node} exceeded the pass limit, marked finished")
                    level_clean_status[level_index][node] = True
                    continue

                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  node {node} has cleaning signals")
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)
                    tset = list(tset)

                    # Spark DataFrame to pandas DataFrame
                    df_pandas = data.select(['index']+sset + [node]).toPandas()
                    print(f"block size: {df_pandas.shape[0]}")

                    preCleaners = [single for single in singles if single.domain in (sset + [node])]
                    _, output, _, _ = customclean(df_pandas, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight

                    partition = [list(model.source) for model in models[node]]
                    print("blocking attributes:")
                    print(partition)
                    editRule1, output, editRuleList, _ = customclean(output, cleaners=[blockModels],
                                                                     partition=partition)
                    editRule *= editRule1
                else:
                    print(f"  node {node} has no cleaning signal")

                editRuleLists.extend(editRuleList)
                editRules *= editRule

                if len(editRuleList) == 0:
                    print(f"  node {node} has no cleaning operation")
                    level_clean_status[level_index][node] = True
                else:
                    print(f"  node {node} has {len(editRuleList)} cleaning operations")

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList
                else:
                    editRuleDict[str(node)] = editRuleList

            print(f"cleaning of {level} finished")
            print("\nEdit rules accumulated in this level:", str(len(editRuleLists)))

            if editRuleLists:
                nextClean = True
                data, _ = transformRulesToSpark(editRuleLists, data, batch_size=batch_size)
                break
            else:
                print("No cleaning operation in this level, next level")
    print("\nProvenance of the cleaning rules:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    single_opinfo = getSingle_opinfo(singles)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print("\nCleaning workflow:")
    sorted_dict = sort_opinfo_by_weights(grouped_opinfo, group_weights)

    # OpPlantuml = generate_plantuml_corrected(single_opinfo, grouped_opinfo, sorted_dict)
    # PlantUML().process_str(OpPlantuml, directory=table_path)
    # plt.savefig(rf"{table_path}/acplt.png", format="png")
    # print("association graph saved to:", rf"{table_path}/acplt.png")

    return data