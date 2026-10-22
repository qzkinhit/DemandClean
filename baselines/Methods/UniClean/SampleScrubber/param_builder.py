import logging
from itertools import product

import numpy as np

from SampleScrubber.cleaner_model import Uniop, ParametrizedLanguage, NOOP


class ParamBuilder(object):  # Parameter construction
    def __init__(self, df, Cleaner, editPenalty, error_threshold=0.2):
        self.df = df
        self.CleanerObject = Cleaner
        self.editPenalty = editPenalty
        self.error_threshold = error_threshold

    def getOperations(self):
        score = [0, 0]
        operations = []
        op = Uniop
        orig = list(op.paramDescriptor.values())
        orig.remove(ParametrizedLanguage.DOMAIN)
        # Parameter combinations of each column
        for target_col in list(self.CleanerObject.target):
            for params in product(*[self.paramSampler(index, target_col) for index in orig]):
                args = {k: v for k, v in zip(op.paramDescriptor.keys(), (target_col,) + params)}  # Dictionary of keys holding the parameters
                score[0] += 1
                if self.optimizeParams(args):
                    score[1] += 1
                    continue
                operations.append(op(**args))

        if 'ExtEdit' in self.CleanerObject.fixValueRules:
            ExtEditList = self.CleanerObject.fixValueRules['ExtEdit']
            operations.extend(ExtEditList)
        operations.append(NOOP())
        logging.debug("Library generator created " + str(len(operations)) + " operations")
        return operations, score

    def paramSampler(self, index, col):
        if index == ParametrizedLanguage.REPAIRVALUE:
            return self.RepairValueSampler(col)
        elif index == ParametrizedLanguage.PREDICATE:
            return self.predicateSampler()
        elif index == ParametrizedLanguage.OPINFORMATION:
            return self.OpInformationSampler()
        else:
            raise ValueError("Error in: " + index)

    def OpInformationSampler(self):
        result = list()
        result.append(str(self.CleanerObject))
        return result

    def predicateSampler(self):
        predicates = []
        # if self.dataset.types[c] == 'cat': #only take categorical values
        for colList in self.LocateAttrSampler():  # Guard against swapped values
            predicates.extend(
                self.getPredicatesDeterministic(self.CleanerObject.quality, colList))
        logging.debug("Predicate Sampler get " + str(len(predicates)))
        return predicates

    def RepairValueSampler(self, col):
        # Default invalid values
        invalid_values = [None, "", '__NULL__', '__NAN__', 'empty', np.nan]

        # Invalid values from invalidDict, if present
        if 'invalidDict' in self.CleanerObject.fixValueRules:
            invalid_dict_values = self.CleanerObject.fixValueRules['invalidDict'].get(col, [])
            invalid_values.extend(invalid_dict_values)

        # Aggregate the existing values of the column. Values below self.error_threshold are very likely wrong and are no candidates
        try:
            # Values whose p is at least the threshold and that are not invalid
            filtered_values = [
                val for val in set(self.df[self.df['p'] >= self.error_threshold][col].values)
                if val not in invalid_values
            ]
        except KeyError:
            # Without a p column, every value of the column that is not invalid
            print(f"column 'p' is missing, column {col} is not filtered, every valid value is a candidate.")
            filtered_values = [
                val for val in set(self.df[col].values)
                if val not in invalid_values
            ]
        if 'ExtDict' in self.CleanerObject.fixValueRules:
            ext_dict_values = self.CleanerObject.fixValueRules['ExtDict'].get(col, [])
            # Keep the values present in ext_dict_values
            filtered_values = [val for val in filtered_values if val in ext_dict_values]

        return filtered_values

    def LocateAttrSampler(self):  # Attribute threshold used for locating errors
        LocateList = []
        for cleaner in self.CleanerObject.cleanerList:
            LocateList.append(list(cleaner.source))
        return LocateList

    def optimizeParams(self, args):
        # Prune with the external knowledge base, or prune clearly wrong parameters
        for tup in args['predicate'][1]:
            if len(tup) == 0 or any(val is None or val == '' for val in tup):
                return True
        if args['predicate'][1] == '':
            return True
        # if '__NULL__' in list(args['predicate'][1])[0]:
        #     return True
        if not bool(list(args['predicate'][1])[0]):
            return True

            # External edit rules present
        if 'ExtEdit' in self.CleanerObject.fixValueRules:
            ExtEditList = self.CleanerObject.fixValueRules['ExtEdit']
            # Prune parameters whose predicate equals an external edit rule
            for edit in ExtEditList:
                if edit['predicate'][0] == args['predicate'][0] and edit['predicate'][1] == args['predicate'][1]:
                    return True

        if args['domain'] in args['predicate'][0]:  # Operator without modification
            domain_index = args['predicate'][0].index(args['domain'])
            for val in args['predicate'][1]:
                if args['repairvalue'] == val[domain_index]:
                    return True

        tuples = args['predicate'][1]  # List of tuples

        for tup in tuples:
            # Does a tuple contain None or NaN
            if any(val is None or val != val for val in tup):
                return True
        return (args['repairvalue'] != args['repairvalue']) or (args['repairvalue'] is None) or (
                len(args['predicate'][1]) == 0)

    def getPredicatesDeterministic(self, quality, colList):
        # a = self.error_threshold  # empirical threshold of the predicted error probability
        # Quality array
        # quality_array = quality(self.df)
        # Apply the condition
        # q_array = np.where(quality_array > a, 1, 0)  # above a: possibly wrong
        q_array = np.sign(quality(self.df))
        vals_inside = set()  # Wrong values
        tuples_inside = set()  # Indices of possibly wrong tuples
        for i in range(len(self.df)):
            val_tuple = tuple(self.df[colList].iloc[i])
            # val = self.df[colList].iloc[i]
            try:
                index = self.df['index'].iloc[i]
            except:
                raise NotImplementedError("the data has no index column")  # Raise: no index
            if q_array[i] == 1.0:
                if '__NULL__' not in val_tuple:
                    vals_inside.add(val_tuple)
                    tuples_inside.add(index)
        # print(tuples_inside)
        return [(colList, {p}, tuples_inside) for p in vals_inside]
