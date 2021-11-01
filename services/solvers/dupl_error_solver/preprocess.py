from itertools import chain


def morph_parse(parsed):
    words = [[s[1] for s in x] for x in parsed]
    lemmas = [[s[2] for s in x] for x in parsed]
    pos = [[s[3] for s in x] for x in parsed]
    return words, lemmas, pos


def find_dupl_inds(words, lemmas, pos):
    func_pos = ["ADP", "CCONJ", "PUNCT", "SCONJ", "PART", "X"]
    dupls = []
    for i in range(len(lemmas) - 2 + 1):
        inner_dupls = list(
            set([x for j, x in enumerate(lemmas[i]) if (lemmas[i].count(x) > 1) and (pos[i][j] not in func_pos)])
        )
        inner_dupls_inds = []
        if inner_dupls:
            for item in inner_dupls:
                inner_dupls_inds.append([i for i, x in enumerate(lemmas[i]) if x == item])
        outer_dupls = list(
            set([x for j, x in enumerate(lemmas[i + 1]) if (x in lemmas[i]) and (pos[i + 1][j] not in func_pos)])
        )
        outer_dupls_inds = []
        if outer_dupls:
            for item in outer_dupls:
                outer_dupls_first = [j for j, x in enumerate(lemmas[i]) if x == item]
                outer_dupls_second = [j for j, x in enumerate(lemmas[i + 1]) if x == item]
                outer_dupls_inds.append((outer_dupls_first, outer_dupls_second))
        dupls.append((list(chain.from_iterable(inner_dupls_inds)), outer_dupls_inds))
    return dupls
