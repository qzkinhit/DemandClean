import numpy as np
import pandas as pd

from SampleScrubber.cleaner_model import CleanerModel
from SampleScrubber.util import distance


class EditPenalty(CleanerModel):
    """EditPenalty: quality penalty of modifications to the dataset.
     A change is compared with the current dataset and scored as an edit penalty.
    """

    def __init__(self, dataset, domain, metric={}, w2vModel=None):
        """The constructor takes a dataset and a dictionary that maps attributes to similarity measures.
        source -- a dataframe
        metric -- dictionary from attributes to a similarity measure, one of ['jaccard', 'semantic', 'edit']
        w2vModel -- model of the semantic similarity ('semantic')
        """
        self.source_dataset = dataset
        # 'edit' similarity by default
        self.metric = {s: 'edit' for s in dataset.columns.values}
        # No semantic similarity by default
        semantic = False
        for m in metric:  # m is the attribute name
            self.metric[m] = metric[m]
            if metric[m] == 'semantic':
                semantic = True
        self.word_vectors = w2vModel
        if semantic and w2vModel is None:
            raise ValueError("the semantic similarity needs a word2vec model")
        self.cache = {}
        super().__init__(domain=domain)  # Call the constructor of the parent class
    def _quality(self, df):
        N = df.shape[0]
        p = df.shape[1]
        qfn_a = np.zeros((N,))  # qfn_a is a quality score
        if self.source_dataset.shape != df.shape:  # Datasets of different shape: an array of ones
            return np.ones((N,))
        for i in range(N):
            for j in self.domain:
                target = str(df.iloc[i][j])
                source = str(self.source_dataset.iloc[i][j])  # str(None) gives 'None', not the empty string
                cname = j  # cname is the attribute name
                # Fast paths for special cases
                if target == source:
                    continue
                elif df.iloc[i][j] is None:
                    continue
                elif self.source_dataset.iloc[i][j] is None:
                    qfn_a[i] = 1.0 / p + qfn_a[i]
                    continue
                elif target == '' and source != target:
                    qfn_a[i] = 1.0 / p + qfn_a[i]
                    continue
                elif source == '' and source != target:
                    qfn_a[i] = 1.0 / p + qfn_a[i]
                    continue

                if self.metric[cname] == 'edit':
                    qfn_a[i] = self.edit(target, source) / p + qfn_a[i]
                elif self.metric[cname] == 'jaccard':
                    qfn_a[i] = self.jaccard(target, source) / p + qfn_a[i]
                elif self.metric[cname] == 'semantic':
                    qfn_a[i] = self.semantic(target, source) / p + qfn_a[i]
                else:
                    raise ValueError('unknown similarity measure: ' + self.metric[cname])
        return qfn_a

    def _quality_null(self, df, qfn):
        return qfn

    def edit(self, target, source):
        # Is target or source '__NULL__' or NaN
        if target == '__NULL__' or source == '__NULL__' or pd.isna(target) or pd.isna(source):
            return 0  # 0 if one of them is '__NULL__' or NaN
        else:
            return distance.levenshtein(target, source)  # Edit distance otherwise

    def jaccard(self, target, source):
        ttokens = set(target.lower().split())
        stokens = set(source.lower().split())
        intersection_len = len(ttokens.intersection(stokens))
        union_len = len(ttokens.union(stokens))
        jaccard_similarity = 1.0 - (intersection_len + 0.0) / (union_len + 0.0)  # jaccard = intersection over union

        return jaccard_similarity

    def semantic(self, target, source):
        ttokens = set(target.lower().split())
        stokens = set(source.lower().split())

        sim = []

        for t in ttokens:
            t_similarities = [(self.word_vectors.similarity(t, r) + 1.0) / 2 for r in
                              stokens]  # word_vectors.similarity() lies in [-1, 1]
            sim.extend(t_similarities)

        return 1 - np.mean(sim)  # Mean semantic similarity
