from collections import Counter
import numpy as np
from SampleScrubber.cleaner_model import CleanerModel


class AttrRelation(CleanerModel):
    """AttrRelation: dependency between two attribute sets, repaired with minimal repairs. A functional
    dependency (A -> B) means that every B attribute can be repaired from the A attributes.
    It covers six rule operators (FD, CFD, DC, PFD, MD and RFD, the last one relaxed at sampling
    time) and can use external knowledge.
    """

    def __init__(self, source, target, name, Cpredicate_list=None,ExtDict=None,invalidDict=None,
                 edit_rule=None,DCpredicate_list=None):
        """
        Define the source and target attribute sets, the cleaning signal fixValueRules and further
        attributes.

        source -- list of attribute names
        target -- list of attribute names
        name -- name of the relation
        condition_func -- condition on the tuple of source values
        ExtDict -- external dictionary of possible target values
        edit_rule -- edit rule
        invalidDict -- known invalid values
        """
        super().__init__(source=set(source), target=set(target), domain=set(source + target))
        self.name = name
        self.fixValueRules = {}
        self.quality_history = {}
        self.msg = '[FunctionalDependency:(s: %s, t: %s)]' % (self.source, self.target)  # Minimal repair under the functional dependency by default
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if Cpredicate_list is not None:
            self.fixValueRules['Cpredicate_list'] = Cpredicate_list
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule
        if DCpredicate_list is not None:
            self.fixValueRules['DCpredicate_list'] = DCpredicate_list
        self.cleanerList = [self]
        # Complex predicates expressed as denial constraints

    def __str__(self):
        return self.msg

    # Error probability of each row under the current rule
    def _quality(self, df):
        """
        Error probability of each row under the current rule.

        df -- the DataFrame to evaluate
        """
        N = df.shape[0]  # Number of rows
        kv = {}

        # Group the target values by source tuple
        for i in range(N):
            s = tuple(df[list(self.source)].iloc[i, :])  # Source tuple
            t = tuple(df[list(self.target)].iloc[i, :])  # Target tuple

            if s in kv:
                kv[s].append(t)  # Keep every occurring target, with repetitions
            else:
                kv[s] = [t]

        qfn_a = np.zeros(N)  # Quality array

        for i in range(N):
            s = tuple(df[list(self.source)].iloc[i, :])  # Source tuple
            t = tuple(df[list(self.target)].iloc[i, :])  # Target tuple
            count = Counter(kv[s])  # Occurrences of each target
            normalization = len(kv[s])  # Number of targets of the source tuple s

            if len(count.keys()) == 1:
                qfn_a[i] = 0  # A single target value: no error
            else:
                qfn_a[i] = 1 - float(count[t]) / normalization  # Share of the target, a small share means a high error probability
            if 'ExtDict' in self.fixValueRules:
                dic =self.fixValueRules['ExtDict']
                if t not in dic:
                    qfn_a[i] = 1.0
            if 'invalidDict' in self.fixValueRules:
                dic =self.fixValueRules['invalidDict']
                if t in dic:
                    qfn_a[i] = 1.0
            if 'Cpredicate_list' in self.fixValueRules:
                for cp in self.fixValueRules['Cpredicate_list']:
                    val = cp.get(df).iloc[i]
                    if not cp.eval(val, df):  # No match: the row is not affected, error probability 0
                        qfn_a[i] = 0.0
                        break
            if 'DCpredicate_list' in self.fixValueRules:
                for dcp in self.fixValueRules['DCpredicate_list']:
                    val = dcp.get(df).iloc[i]
                    if dcp.eval(val, df): # Match: the denial constraint is violated, error probability 1
                        qfn_a[i] = 1.0
                        break
        return qfn_a  # Error probability of each row


class Predicate(CleanerModel):
    """
    Predicate: the basic predicate structure.
    """

    def __init__(self, local_attr, expression):
        """ Constructor of DCPredicate.

        local_attr -- an attribute name
        expression -- a function fn: domain(local_attr) x df -> {true, false}
        """
        self.local_attr = local_attr
        self.expression = expression

    def get(self, df):
        """Project the data frame onto the attribute.

        df -- a data frame
        """
        return df[self.local_attr]

    def eval(self, value, df):
        """Evaluate the predicate.

        value -- a value of domain(local_attr)
        df -- a data frame
        """
        return self.expression(value, df)
