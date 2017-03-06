from numpy import min, sum, array
from nltk.corpus import stopwords
from Levenshtein import distance


def leve_list(seq1, seq2):
    seq1 = [word for word in seq1 if word not in stopwords.words('english')]
    seq2 = [word for word in seq2 if word not in stopwords.words('english')]
    print(seq1, seq2)
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 1.0
    if len1 > len2:
        len1, len2 = len2, len1
        seq1, seq2 = seq2, seq1
    cross_dist = array([[distance(x, y)/max(len(x), len(y)) for y in seq2] for x in seq1])
    mins = min(cross_dist, axis=0)
    mins.sort()
    score = sum(mins[:len1])/len1
#     return score, cross_dist
    return score
