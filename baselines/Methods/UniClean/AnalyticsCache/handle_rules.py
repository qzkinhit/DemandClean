import numpy
import numpy as np
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
from functools import reduce


def getOpWeights(editRuleDict, groupOpInfo):
    """
    Weight of every operation.

    Parameters:
    editRuleDict (dict): edit rules
    groupOpInfo (dict): grouped operation information

    Returns:
    dict: weight of every operation
    """
    op_weights_dict = {}
    TotalRule = 0
    for level_index in groupOpInfo:  # Analyse every group
        level = groupOpInfo[level_index]
        for targetNode in level:
            group_weight = []
            weight = 0
            for cleaner in level[targetNode]:
                for source in cleaner.source:
                    weight += analyticsRuleDict(editRuleDict[targetNode], source)
                TotalRule += weight
                group_weight.append(weight)
            op_weights_dict[targetNode] = group_weight
    # Divide every element of each sublist by TotalRule
    if TotalRule == 0:
        return op_weights_dict
    else:
        # Update the dictionary
        op_weights = {key: [float(x) / TotalRule for x in values] for key, values in op_weights_dict.items()}
    return op_weights


def analyticsRuleDict(editRuleList, source):
    """
    Weight of a rule list.

    Parameters:
    editRuleList (list): edit rules
    source (str): source string

    Returns:
    int: weight
    """
    weight = 0
    for rule in editRuleList:
        if source in rule.predicate[0]:
            weight += 1
    return weight


def transformRulesToSQL(EditRuleList, target_table, source_table):
    """
    Turn a list of edit rules into SQL statements.

    Parameters:
    EditRuleList (list): edit rules

    Returns:
    list: SQL statements
    """
    SQLList = []
    # Copy the table and add a clean column
    for EditRule in EditRuleList:
        column = EditRule.domain
        predicates = EditRule.predicate[0]
        values = EditRule.predicate[1]
        value = EditRule.repairvalue

        for val_tuple in values:
            condition = " AND ".join([f"{pred} = '{val}'" for pred, val in zip(predicates, val_tuple)])
            sql = f"UPDATE data SET {column} = '{value}' WHERE {condition};"
            SQLList.append(sql)
    return SQLList


from pyspark.sql import functions as F
from functools import reduce
import numpy as np

def transformRulesToSpark(EditRuleList, df, batch_size=300, maxhandle=20000):
    """
    Apply a list of edit rules to a DataFrame with the Spark API.

    Parameters:
    EditRuleList (list): edit rules
    df (DataFrame): DataFrame to modify
    batch_size (int): rules per batch
    maxhandle (int): maximum number of rules handled

    Returns:
    DataFrame: modified DataFrame
    list: rules left unhandled
    """
    handled_rules = 0  # Number of handled rules
    all_conditions = {}  # Conditions and repair values per column

    for i in range(0, len(EditRuleList), batch_size):
        batch = EditRuleList[i:i + batch_size]

        # Merge the conditions into the dictionary
        for EditRule in batch:
            try:
                column = EditRule.domain
                predicate_attrs = EditRule.predicate[0]
                value_sets = EditRule.predicate[1]
                repair_value = EditRule.repairvalue

                # Condition list of the column
                if column not in all_conditions:
                    all_conditions[column] = []

                # Integer values
                value_sets = [
                    tuple(map(lambda x: int(x) if isinstance(x, np.integer) else x, v))
                    for v in value_sets
                ]

                # A single repair condition
                conditions = []
                for value_set in value_sets:
                    disguised_missing_values = {"__NULL__", "null", "N/A", "empty", ""}

                    # Sub-conditions
                    sub_conditions = [
                        (F.col(attr).isNull() | F.col(attr).isin(disguised_missing_values)) if val == "__NULL__"
                        else F.col(attr) == val
                        for attr, val in zip(predicate_attrs, value_set)
                    ]

                    # Combine the sub-conditions
                    if sub_conditions:
                        condition = reduce(lambda x, y: x & y, sub_conditions)
                        conditions.append(condition)

                # Add the condition and repair value to the column
                if conditions:
                    final_condition = reduce(lambda x, y: x | y, conditions)
                    all_conditions[column].append((final_condition, repair_value))
                    handled_rules += 1

                    # maxhandle reached
                    if handled_rules >= maxhandle:
                        # Apply the collected rules before returning early
                        df = apply_conditions_to_df(df, all_conditions)
                        remaining_rules = EditRuleList[i + batch_size:]
                        print(f"maximum number of rules {maxhandle} handled, returning early")
                        return df, remaining_rules
            except Exception as e:
                print(f"error while handling rule {EditRule}: {e}")
                continue

    # Apply the conditions after all rules
    df = apply_conditions_to_df(df, all_conditions)

    # Trigger an action to materialize the DataFrame
    print(f"rows read: {df.count()}")
    print(f"{handled_rules} rules handled, updating the data...")
    df = df.persist()
    print("all edit rules applied")

    # Every rule handled: the DataFrame and an empty list of remaining rules
    return df, []

def apply_conditions_to_df(df, all_conditions):
    """
    Apply conditions and repair values to a DataFrame.

    Parameters:
    df (DataFrame): DataFrame to modify
    all_conditions (dict): conditions and repair values per column

    Returns:
    DataFrame: modified DataFrame
    """
    for column, cond_list in all_conditions.items():
        # A `when` condition per repair condition
        column_update = F.when(cond_list[0][0], cond_list[0][1])
        for cond, repair_value in cond_list[1:]:
            column_update = column_update.when(cond, repair_value)
        # Keep the original value otherwise
        df = df.withColumn(column, column_update.otherwise(F.col(column)))
    return df
def transformKnowledgeToSpark(spark, database_name,table_name,df, cleaner):
    """
    Apply an external knowledge-base table to a DataFrame with Spark.

    Parameters:
    spark (SparkSession): Spark session
    database_name (str): database name
    df (DataFrame): DataFrame to modify
    cleaner (AttrRelation): one AttrRelation
    batch_size (int): rules per batch

    Returns:
    DataFrame: modified DataFrame
    """

    source_attributes = list(cleaner.source)
    target_attribute = list(cleaner.target)[0]  # target is a single attribute

    # Name of the knowledge-base table
    source_str = '_'.join(source_attributes)
    target_str = target_attribute
    knowledge_table_name = f"{table_name}_external_knowledge_{source_str}_to_{target_str}"

    # Does the table exist
    table_exists = spark._jsparkSession.catalog().tableExists(database_name, knowledge_table_name)
    if not table_exists:
        print(f"knowledge-base table {database_name}.{knowledge_table_name} not found, cleaner skipped.")
        return df

    # Read the knowledge-base table
    knowledge_df = spark.table(f"{database_name}.{knowledge_table_name}")

    # Join it with the DataFrame
    join_conditions = [df[attr] == knowledge_df[attr] for attr in source_attributes]
    df = df.join(knowledge_df, on=join_conditions, how='left')

    # Update the target attribute
    df = df.withColumn(target_attribute,
                       when(col('target_value').isNotNull(), col('target_value')).otherwise(col(target_attribute)))

    # Drop the temporary columns
    for attr in source_attributes:
        df = df.drop(knowledge_df[attr])
    df = df.drop('target_value')

    # Persist (optional)
    df.count()
    # df.unpersist()
    # df = df.persist()

    print(f"cleaner handled: {knowledge_table_name}")
    return df

# Example
# df, remaining_rules = transformRulesToSpark(EditRuleList, df, batch_size=300, maxhandle=2000)
