from array import array


def levenshtein(seq1, seq2, method=1):
    """Normalized Levenshtein distance between `seq1` and `seq2`.
    The value lies in [0, 1], 0 for equal sequences and 1 for completely different ones.
    0.0     if seq1 == seq2
    1.0     if len(seq1) == 0 or len(seq2) == 0
    edit distance over the length of the shortest alignment otherwise
    """

    if seq1 == seq2:
        return 0.0
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 1.0
    if len1 < len2:  # minimize the arrays size
        len1, len2 = len2, len1
        seq1, seq2 = seq2, seq1

    return editdistance(seq1, seq2) / float(len1)


"""
test:
    editdistance('1','abc')
    3
    editdistance('1','abcd')
    4
    editdistance('1234234','abcd')
    7
"""


def editdistance(seq1, seq2, max_dist=-1):
    """Absolute edit distance between the sequences `seq1` and `seq2`.
        The edit distance is the minimum number of edit operations that turn one sequence into
        the other. The allowed operations are

            * deletion: ABC -> BC, AC, AB
            * insertion: ABC -> ABCD, EABC, AEBC..
            * substitution: ABC -> ABE, ADC, FBC..
    """

    if seq1 == seq2:
        return 0

    len1, len2 = len(seq1), len(seq2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if len1 < len2:
        len1, len2 = len2, len1
        seq1, seq2 = seq2, seq1

    column = array('L', range(len2 + 1))

    for x in range(1, len1 + 1):
        column[0] = x
        last = x - 1
        for y in range(1, len2 + 1):
            old = column[y]
            cost = int(seq1[x - 1] != seq2[y - 1])
            column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
            last = old
    return column[len2]
