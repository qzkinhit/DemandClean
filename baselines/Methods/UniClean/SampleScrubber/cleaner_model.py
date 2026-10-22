"""Module: UniOpModel
    A common structure of cleaning operators, the interface for new operators. It defines the
    operator structure, the evaluation modes of operators and the structure of cleaning parameters.
"""
import datetime
import logging

import numpy as np

from SampleScrubber.util.format import formatString


class CleanerModel(object):
    """Base wrapper class of a cleaning operator.
    """

    def __init__(self, domain=None, source=None, target=None, msg=None):
        """
        Define the source, target and domain attribute sets and further attributes.

        Input:
        - domain: attribute domain involved, empty by default
        - source: source attributes, empty by default
        - target: target attributes, empty by default
        - msg: message of the operator

        Output:
        - source, target, domain, quality_history, msg, cleanerList and fixValueRules of the object
        """
        self.source = source if source else set()
        self.target = target if target else set()
        self.domain = domain if domain else set()
        self.quality_history = {}  # Latest evaluation
        self.msg = msg
        self.cleanerList = []
        self.fixValueRules = {}

    def quality(self, df):
        """
        Public quality interface, wraps the quality function of a database instance.

        Input:
        - df: a Spark DataFrame

        Output:
        - quality value of every row
        """
        return self._quality_null(df, self._quality(df))

    def _quality(self, df):
        """
        Abstract method, subclasses implement the quality evaluation.

        Input:
        - df: a Spark DataFrame

        Output:
        - quality value of every row
        """
        raise NotImplementedError("quality function of the operator not implemented")

    def _quality_null(self, df, qfn):
        """
        Quality evaluation with missing values taken into account.

        Input:
        - df: a Spark DataFrame
        - qfn: quality values of the rows

        Output:
        - updated quality values
        """
        for col in self.domain:
            for i in range(len(df)):
                if not bool(df.iloc[i][col]):
                    qfn[i] = 1
        return qfn

    def __str__(self):
        """
        Message of the cleaning operator.

        Output:
        - message string
        """
        try:
            return self.msg
        except:
            return 'no message for this cleaning operator'

    def __add__(self, other):
        """
        Addition: combine two quality functions.

        Input:
        - other: another CleanerModel

        Output:
        - a new CleanerModel holding the mean of both quality functions
        """
        uom = CleanerModel()

        def new_quality(df):
            self_value = self.quality(df)
            other_value = other.quality(df)
            result = (self_value + other_value) / 2
            uom.quality_history = {
                "left": (self.msg, np.mean(self_value)),
                "operator": "+",
                "right": (other.msg, np.mean(other_value)),
                "result": np.mean(result)
            }
            uom.msg = str(uom.quality_history).replace("\'", "")
            return result

        uom.quality = new_quality
        uom.cleanerList = self.cleanerList
        uom.cleanerList.append(other)
        uom.target = self.target.union(other.target)
        uom.source = self.source.union(other.source)
        uom.domain = self.domain.union(other.domain)
        uom.fixValueRules = self.fixValueRules.copy()
        uom.fixValueRules.update(other.fixValueRules)
        uom.msg = str(self) + '+' + str(other)
        return uom

    def __mul__(self, other):
        """
        Multiplication: element-wise comparison or product of two quality functions.

        Input:
        - other: a number (weight) or another CleanerModel

        Output:
        - a new CleanerModel with the element-wise result
        """
        uom = CleanerModel()
        try:
            fother = float(other)

            def new_quality(df):
                self_value = self.quality(df)
                result = fother * self_value
                uom.quality_history = {
                    "left": (self.msg, np.mean(self_value)),
                    "operator": "*",
                    "right": (str(fother), fother),
                    "result": np.mean(result)
                }
                uom.msg = str(uom.quality_history).replace("\'", "")
                return result

            uom.quality = new_quality
            uom.cleanerList = self.cleanerList
            uom.domain = self.domain
            uom.fixValueRules = self.fixValueRules.copy()
            uom.msg = self.msg + '*' + str(fother)
            uom.target = self.target
            uom.source = self.source

        except:
            def new_quality(df):
                self_value = self.quality(df)
                other_value = other.quality(df)
                result = np.maximum(self_value, other_value)
                uom.quality_history = {
                    "left": (self.msg, np.mean(self_value)),
                    "operator": "^",
                    "right": (other.msg, np.mean(other_value)),
                    "result": np.mean(result)
                }
                uom.msg = str(uom.quality_history).replace("\'", "")
                return result

            uom.cleanerList = self.cleanerList
            # Update the cleaning signals and parameters
            uom.cleanerList.append(other)
            uom.quality = new_quality
            uom.domain = self.domain.union(other.domain)
            uom.fixValueRules = self.fixValueRules.copy()
            uom.fixValueRules.update(other.fixValueRules)
            uom.target = self.target.union(other.target)
            uom.source = self.source.union(other.source)
            uom.msg = self.msg + '^' + str(other.msg)

        return uom


class Language(object):
    """
    Base class of repair-language parameters, operations that run on a data frame.
    """

    def __init__(self, runfunc, reward=1, provenance=[]):
        """
        Parameters:
            runfn: function of the operator, applied to the data frame
            reward: integer contribution of the operator to the repair
            provenance: list, origin of the operator
        """
        self.runfunc = lambda df: runfunc(df)  ## df is the data frame
        self.reward = reward# Cost of the cleaning operation
        if provenance != []:
            self.provenance = provenance  # Origin of the operator
        self.quality_history = {}  # Latest evaluation

    def set_reward(self, r):
        self.reward = r

    def run(self, df):
        """
        Run the operator, record its run time and return the resulting data frame.
        """
        op_start_time = datetime.datetime.now()  ## start time
        df_copy = df.copy(deep=True)
        result,opLists = self.runfunc(df_copy)
        logging.debug(
            'Running ' + self.name + ' took ' + str((datetime.datetime.now() - op_start_time).total_seconds()))
        return result,opLists

    def __mul__(self, other):
        """
            Composition of data-frame operations: the operator combines two operations into one
        """
        new_runfunc = lambda df, a=self, b=other: b.runfunc(a.runfunc(df))
        new_op = Language(new_runfunc, provenance=self.provenance + other.provenance)
        new_op.name = (self.name + "\n" + other.name).strip()
        new_op.reward = self.reward + other.reward
        return new_op

    def __pow__(self, b):
        """
            Repeated application
        """
        op = self
        for i in range(b):
            op *= self
        return op

    def __str__(self):
        return self.name

    __repr__ = __str__


class ParametrizedLanguage(Language):
    """
    Parametrized operation that runs on a data frame.
    """

    # Parameter attribute constants
    DOMAIN = 0
    REPAIRVALUE = 1
    PREDICATE = 2
    OPINFORMATION = 3
    FORMAT = 4

    def __init__(self, runfunc, params, reward=0):
        """
        Parameters:
        runfunc: logic applied to every row of the data frame
        params: basic repair parameters
        reward: number of repairs
        """
        self.CheckParams(params)
        super(ParametrizedLanguage, self).__init__(runfunc, reward)

    def CheckParams(self, params):
        """
        Validate the parameters.
            Raises ValueError for invalid parameters.
        """
        try:
            self.paramDescriptor  # Parameter description present
        except:
            raise NotImplemented("Language does not have paramDescriptor")

        for p in params:
            if p not in self.paramDescriptor:
                raise ValueError("Parameter " + str(p) + " not defined")

            if self.paramDescriptor[p] not in range(5):
                raise ValueError("Parameter " + str(p) + " has an invalid descriptor")


class Uniop(ParametrizedLanguage):
    """
    Repair meta-language that covers every cleaning operation
    """

    paramDescriptor = {'domain': ParametrizedLanguage.DOMAIN,
                       'repairvalue': ParametrizedLanguage.REPAIRVALUE,
                       'predicate': ParametrizedLanguage.PREDICATE,
                       'opinformation': ParametrizedLanguage.OPINFORMATION,
                       }

    def __init__(self, domain, repairvalue, predicate, opinformation):
        """
        Replacement operator.

        Parameters:
        domain: str, column whose values are replaced
        predicate: tuple with the lookup condition
        repairvalue: str, replacement value
        opinformation: str, origin of the cleaning
        """
        # Predicate of the repair condition
        # predicate has the form (column_list, tuple_list)
        basic_predicate = lambda row: tuple(row[col] for col in predicate[0]) in predicate[1]and (
                row['index'] in predicate[2])

        # basic_predicate = lambda row: (row[predicate[0]] in predicate[1]) and (
        #         row['index'] in predicate[2])

        self.domain = domain
        self.predicate = predicate
        self.repairvalue = repairvalue
        self.opinformation = opinformation
        self.name = 'Uniop(df,' + formatString(domain) + ',' + formatString(repairvalue) + ',' + str(
            predicate[0:2]) + ','

        def runfunc(df, domain=domain, predicate=basic_predicate, v=repairvalue):
            def __internal(row):
                # Replace the value when the condition holds, keep it otherwise
                if predicate(row):
                    return v
                else:
                    return row[domain]

            df[domain] = df.apply(lambda row: __internal(row), axis=1)  # Apply row by row
            return df,[]

        self.provenance = [self]
        super(Uniop, self).__init__(runfunc, ['domain', 'repairvalue', 'predicate', 'opinformation'], reward=1)


class NOOP(Language):
    """
        No-op
    """

    def __init__(self):
        def runfunc(df):  # No-op on the data
            return df,[]

        self.name = "NOOP"
        self.provenance = [self]
        reward = 0
        super(NOOP, self).__init__(runfunc, reward=reward,provenance=self.provenance)