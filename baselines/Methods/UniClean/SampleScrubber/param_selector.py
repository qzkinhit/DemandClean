"""
Module: ParamSelector
Best-first search that expands the most promising nodes by the quality function and the edit penalty.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from SampleScrubber.param_builder import ParamBuilder
from SampleScrubber.cleaner_model import NOOP
from SampleScrubber.cleaner.clean_penalty import EditPenalty

DEFAULT_CONFIG = {
    'maxSearch': 10,
    'editCost': 1,
    'simfunc': {},  # Similarity measure
    'w2v': '../../resources/GoogleNews-vectors-negative300.bin'  # Word-vector model
}


def customclean(df, precleaners=[], cleaners=[], partition=None, error_threshold=None, config=None):
    """
    Takes a list of patterns and dependencies and returns a cleaned instance and a cleaning program.

    Positional arguments:
    df -- pandas DataFrame
    precleaners -- preprocessing operators
    cleanners -- operators applied after preprocessing
    partition -- blocking rule that partitions the dataset
    config -- configuration object
    """

    if config is None:
        config = DEFAULT_CONFIG
    op = NOOP()

    logging.debug('Starting the select with the following config: ' + str(df.shape) + " " + str(config))

    if needWord2Vec(config):  # Is word2vec needed
        w2v = loadWord2Vec(config['w2v'])

        logging.debug('use word2vec for semantic similarity')

    else:
        w2v = None

    config['model'] = w2v
    positiveOp = []
    negativeOp = []  # Operators found to be bad at some computational cost
    neutralOp = []  # Operators that are not that bad but were expensive to discard
    bestopList = list()
    pruningModel = None
    score = [0, 0]
    if partition is not None:
        block_indices = find_blocks(df, partition, error_threshold)
        # blocks = set(df[partition].dropna().values)
        for i, indices in enumerate(block_indices):
            block_df = df.loc[indices].copy()
            if block_df.shape[0] < 2:
                continue
            logging.info(f"Processing Block size={block_df.shape[0]}")
            logging.info("Computing Block " + str(i + 1) + " out of " + str(len(block_indices)))
            print("Computing Block " + str(i + 1) + " out of " + str(len(block_indices)))
            print(f"Processing Block size={block_df.shape[0]}")
            # if block_df.shape[0] >= 50:
            #     print(111)
            now = datetime.now()
            op1, cleaned_block, score_1, PositiveOp_1, neutralOp_1, negativeOp_1, bestopList_1 = handlecleaner(block_df,
                                                                                                               precleaners,
                                                                                                               cleaners,
                                                                                                               config,
                                                                                                               error_threshold=error_threshold)
            if not isinstance(op1, NOOP):
                op = op * op1
                score[0] += score_1[0]
                score[1] += score_1[1]
                positiveOp.extend(PositiveOp_1)
                neutralOp.extend(neutralOp_1)
                negativeOp.extend(negativeOp_1)
                bestopList += bestopList_1
                df.loc[indices] = cleaned_block
            logging.info(f"Block processing time: {(datetime.now() - now).total_seconds()} seconds")
            print(f"Block processing time: {(datetime.now() - now).total_seconds()} seconds")
            # print((datetime.now() - now).total_seconds())
    else:
        logging.warning("No partitioning rule specified, processing might be slow")
        op, df, score_1, positiveOp, neutralOp, negativeOp, bestopList = handlecleaner(df, precleaners, cleaners,
                                                                                       config,
                                                                                       error_threshold=error_threshold)
        score[0] += score_1[0]
        score[1] += score_1[1]

    return op, df, bestopList, [positiveOp, neutralOp, negativeOp, score]


def find_blocks(df, partition, error_threshold):
    def filter_partition_attributes(df, partition):
        attribute_group_sizes = {}
        removed_attributes = []  # Deleted attribute combinations
        for attrs in partition:
            group_size = df.groupby(attrs).ngroups
            attribute_group_sizes[tuple(attrs)] = group_size
            print(f"Grouping by {attrs} yields {group_size} groups.")

        total_rows = len(df)
        selected_attributes = list(partition)

        while True:
            group_product = 1
            for attrs in selected_attributes:
                group_product *= attribute_group_sizes[tuple(attrs)]

            print(f"Current group product: {group_product}, Total rows: {total_rows}")

            if group_product <= total_rows:
                break

            # Delete the attribute combination with the most groups
            max_group_attr = max(selected_attributes, key=lambda attrs: attribute_group_sizes[tuple(attrs)])
            print(f"Removing attribute group {max_group_attr} with {attribute_group_sizes[tuple(max_group_attr)]} groups.")
            removed_attributes.append(max_group_attr)  # Deleted attributes
            selected_attributes.remove(max_group_attr)

        print(f"Final selected attributes for partitioning: {selected_attributes}")
        print(f"Removed attributes for potential later use: {removed_attributes}")
        return selected_attributes, removed_attributes

    def initial_block_mapping(df, partition):
        initial_blocks = df.groupby(partition[0]).groups
        block_mapping = {}
        for key, indexes in initial_blocks.items():
            for index in indexes:
                block_mapping[index] = set(indexes)
        return block_mapping

    def update_block_mapping(df, partition, error_threshold, block_mapping):
        updated = False
        for attrs in partition:
            value_to_indexes = df.groupby(attrs).groups
            for indexes in value_to_indexes.values():
                if len(indexes) > 1:
                    connected_indexes = set()
                    for index in indexes:
                        if df.loc[index, 'p'] < error_threshold:
                            connected_indexes.update(block_mapping[index])
                    if connected_indexes:
                        for index in connected_indexes:
                            if block_mapping[index] != connected_indexes:
                                block_mapping[index] = connected_indexes
                                updated = True
        return block_mapping, updated

    def further_split_empty_blocks(df, final_blocks, removed_attributes):
        empty_values = ['__NULL__', 'empty', 'null']  # Missing-value tokens
        new_final_blocks = []

        for block in final_blocks:
            all_empty = True
            for index in block:
                # Is the block empty under the current attribute combinations
                if any(
                    pd.notna(df.loc[index, attr]) and str(df.loc[index, attr]).strip() not in empty_values
                    for attr_group in partition for attr in attr_group
                ):
                    all_empty = False
                    break

            if all_empty:
                if removed_attributes:
                    print(f"Block {block} has all empty values in current attributes. Further splitting with removed attributes.")
                    # Group the block by the deleted attribute combinations
                    temp_df = df.loc[block]  # Only the rows of the current block
                    for attrs in removed_attributes:
                        temp_blocks = temp_df.groupby(attrs).groups.values()
                        new_final_blocks.extend(temp_blocks)
                        print(f"Further split block {block} using {attrs}, creating {len(temp_blocks)} new blocks.")
                        break
            else:
                new_final_blocks.append(block)

        return new_final_blocks

    removed_attributes=[]
    # Delete the largest combination whose group product exceeds the row count
    if len(partition) > 1:
        partition, removed_attributes = filter_partition_attributes(df, partition)

    # Step 1: initial blocks
    print("Starting initial block mapping...")
    block_mapping = initial_block_mapping(df, partition)
    if len(partition) > 1:
        # Step 2: update the block map until it is stable
        print("Updating block mapping until stable...")
        while True:
            block_mapping, updated = update_block_mapping(df, partition, error_threshold, block_mapping)
            if not updated:
                break

    # Final list of blocks
    final_blocks = set(frozenset(indexes) for indexes in block_mapping.values())
    print(f"Initial number of blocks: {len(final_blocks)}")

    # Step 3: split the blocks that are empty under the current combinations
    final_blocks = further_split_empty_blocks(df, final_blocks, removed_attributes)
    print(f"Final number of blocks after further splitting: {len(final_blocks)}")

    # Index sets of the blocks
    return [list(block) for block in final_blocks]
def handlecleaner(df, precleaners, cleaners, config, error_threshold=0.2):
    op = NOOP()
    score = [0, 0]
    positiveOp = []
    negativeOp = []  # Operators found to be bad at some computational cost
    neutralOp = []  # Negative examples that are not that bad but were expensive to discard
    bestopList = []
    for p in precleaners:
        preLanguage = p.preProcess()
        df,opLists= preLanguage.run(df)
        bestopList += opLists
        op = op * preLanguage
        logging.debug('Enforcing pattern cleaner=' + str(p))
    # print(df)
    for c in cleaners:
        print("begin cleaner" + str(c))
        cleanop, df, score_1, PositiveOp_1, neutralOp_1, negativeOp_1, bestopList_1 = selector(df, c, depth=config[
            'maxSearch'], editCost=config['editCost'], simfunc=config['simfunc'], word2vec=config['model'],
                                                                                               error_threshold=error_threshold)
        if not isinstance(cleanop, NOOP):
            op = op * cleanop
            score[0] = score_1[0] + score[0]
            score[1] = score_1[1] + score[1]
            positiveOp.extend(PositiveOp_1)
            neutralOp.extend(neutralOp_1)
            negativeOp.extend(negativeOp_1)
            bestopList += bestopList_1
    return op, df, score, positiveOp, neutralOp, negativeOp, bestopList


def selector(df, cleaner, depth, editCost, simfunc, word2vec, error_threshold=0.2):
    """selector: select among the parameters generated for a cleaner.
        Positional arguments:
        df -- pandas DataFrame
        cleaner -- cleaning operator
        depth -- maximum search depth
        editCost -- scale of the edit penalty
        simfunc -- similarity measure
        word2vec -- word2vec model (loaded once)
        """
    domain = cleaner.domain
    bestopList = []
    positiveOp = []  # Best operator of each round
    EditPenaltytObj = EditPenalty(df.copy(), domain, simfunc, word2vec)
    epq = EditPenaltytObj.quality
    bad_op_cache = set()
    negativeOp = []  # Operators found to be bad at some computational cost
    neutralOp = []  # Negative examples that are not that bad but were expensive to discard
    search_start_time = datetime.now()

    all_operations = set()  # Set of explored operations
    p = ParamBuilder(df, cleaner, EditPenaltytObj, error_threshold=error_threshold)
    clean_bool, score = p.getOperations()
    best = (2.0, NOOP(), df, NOOP(), -1)
    # Best result: a tuple with the initial score (2.0), the operator sequence (NOOP()), the DataFrame df, the last applied operator (NOOP()) and the current inconsistencies
    select_start_time = datetime.now()
    have_search = False
    # qfnEval = cleaner.quality(df)
    # print(qfnEval)
    truth_depth = 0
    for i in range(depth):
        value, op, frame, lastop, error = best
        if not isinstance(lastop, NOOP):
            # Update the cache
            bestopList.append(lastop)
        if len(clean_bool) <= 1 or error == 0:
            truth_depth = i + 1
            break
        try:
            logging.debug('select depth=' + str(i))
        except:
            print("files cannot be written in parallel")
        if not isinstance(lastop, NOOP):
            # Update the cache
            clean_bool, prunnum = UpdateCleanBool(clean_bool, lastop, bad_op_cache)
            print("UpdateCleanBool,purned:" + str(prunnum))
            score[1] += prunnum
            # Empty bad_op_cache
            bad_op_cache = set()
            have_search = False

        print("clean_bool length:" + str(len(clean_bool)))
        # df = frame
        for j, opbranch in enumerate(clean_bool):
            logging.debug('At op=' + str(i) + ': ' + opbranch.name)
            if not isinstance(opbranch, NOOP):
                all_operations.add(opbranch)
            else:
                continue
            if opbranch.name in bad_op_cache:
                continue
            output,_ = opbranch.run(frame)
            # Pruning
            qfnEval = cleaner.quality(output)
            now_error = np.sum(qfnEval)
            if error <= now_error and error != -1:  # The inconsistency should decrease
                negativeOp.append(opbranch)  # One opbranch run spent to find the operator useless, a useful negative example
                bad_op_cache.add(opbranch.name)
                score[1] += 1
                print('pruned op=' + str(j) + ': ' + opbranch.name)
                continue
            else:
                positiveOp.append(opbranch)
            # if pruningRules(CoreSetOutput):
            #     logging.debug('Pruned Search Branch=' + str(l) + ' ' + opbranch.name)
            #     continue

            editfn = epq(output)
            cost_average = (np.sum(qfnEval) + editCost * np.sum(editfn)) / output.shape[0]

            # n = (costEval.agg(F.sum("qfn")).first()[0] + editCost * sum(editfn)) / (CoreSetOutput.count())
            if cost_average < best[0]:
                # For good rules, spend time on updating the provenance
                if isinstance(op, NOOP):
                    op.name = "Init quaility: " + opbranch.opinformation  # Initial quality
                opbranch.opinformation = cleaner.msg
                opbranch.name = opbranch.name + opbranch.opinformation + ')'
                opbranch.quality_history = cleaner.quality_history
                print('newest best op: ' + opbranch.name)
                # print('cost_average =' + str(cost_average))
                # print('qfnEval = ' + str(np.mean(qfnEval)))
                # logging.debug('selected op=' + str(j) + ': ' + opbranch.name)
                # score = cost_average
                # if  have_search:
                #     best = (cost_average, nextop, output, opbranch, np.sum(qfnEval))
                #     break
                # else:
                nextop = op * opbranch
                best = (cost_average, nextop, output, opbranch, now_error)
                print("now error:", now_error)
                have_search = True
            else:
                neutralOp.append(opbranch)  # Negative example that cost two computations (run op and compute qfn)
        if not have_search:
            truth_depth = i + 1
            break
        logging.debug(
            'Search Depth=' + str(i) + " took " + str((datetime.now() - search_start_time).total_seconds()))
    logging.debug('Search  took ' + str((datetime.now() - search_start_time).total_seconds()))
    score[0] *= depth
    score[1] += (score[1] * (depth - truth_depth))
    return best[1], best[2], score, positiveOp, neutralOp, negativeOp, bestopList


def UpdateCleanBool(clean_bool, lastop, bad_op_cache):
    if not isinstance(lastop, NOOP):
        # Attributes of the last operation
        predicate_colList = lastop.predicate[0]  # Locating attribute
        predicate_valueTuple = lastop.predicate[1]  # Value of the locating attribute
        repaired_value = lastop.repairvalue  # Repaired value
        repaired_column = lastop.domain  # Attribute to repair

        # Set of operations that satisfy the condition
        new_clean_bool = set()
        old_clean_bool_num = len(clean_bool)
        for op in clean_bool:
            if isinstance(op, NOOP):
                new_clean_bool.add(op)
            else:
                if op.name in bad_op_cache:  # Skip rules rejected in the previous round
                    continue
                elif predicate_colList == op.predicate[0] and predicate_valueTuple == op.predicate[
                    1] and op.domain in predicate_colList:  # Predicted attribute of the best locating rule, not to be modified
                    continue
                elif predicate_colList == op.predicate[0] and predicate_valueTuple == op.predicate[
                    1] and repaired_column == op.domain:  # Repair content of the best rule, the other candidates are worse
                    continue
                else:
                    new_clean_bool.add(op)
        new_clean_bool_num = len(new_clean_bool)
        prunnum = old_clean_bool_num - new_clean_bool_num
        print("pruned: " + str(prunnum))
        # Update clean_bool
        return new_clean_bool, prunnum
    else:
        return clean_bool, 0


def pruningRules(output):
    if output.shape[1] == 0:
        return True

    elif output.shape[0] == 0:
        return True

    return False


def needWord2Vec(config):
    """Is word2vec needed."""
    semantic = 'semantic' in [config['simfunc'][k] for k in config['simfunc']]

    return semantic


def loadWord2Vec(filename):
    """Load the word2vec model."""
    return KeyedVectors.load_word2vec_format(filename, binary=True)
