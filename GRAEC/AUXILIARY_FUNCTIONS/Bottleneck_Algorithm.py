"""
Labelling algorithm as described in the paper
"""

from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Process_Units import Case


def algorithm(set_of_cases):
    assert (all(isinstance(c, Case) for c in set_of_cases))
    labels = dict()

    # Create all variants
    variants = dict()
    for case in set_of_cases:
        variants.setdefault(case.str_trace(), set()).update({case})

    if Parameters.Demo:
        assert (len(variants) == 1)

    # Line 1
    for variant, variant_cases in variants.items():

        # (Pre algorithm) Split between H and L
        h_i = {c for c in variant_cases if c.get_duration() <= Parameters.d}
        l_i = {c for c in variant_cases if c.get_duration() > Parameters.d}

        # Line 2 (kind of)
        sigma = variant.split(';')

        # Line 3
        for case in h_i:
            # Line 4
            labels[case.case_id] = Parameters.short_label
            # Line 5

        avg = dict()
        std = dict()

        # Line 6
        for k in range(1, len(sigma)):
            # Line 7 and 8
            durations = [case.get_event(k).time - case.get_event(k - 1).time for case in h_i]
            avg[k] = 1 / len(h_i) * sum(durations)
            std[k] = (sum([(i - avg[k]) ** 2 for i in durations]) / len(h_i)) ** 0.5
            # Line 9

        # Line 10
        for case in l_i:
            # Line 11
            k_0 = set(range(1, len(sigma)))
            # Line 12
            k_1 = {k for k in k_0 if case.get_event(k).time - case.get_event(k - 1).time > avg[k]}
            # Line 13
            k_2 = {k for k in k_1 if
                   (case.get_event(k).time - case.get_event(k - 1).time) / (case.get_duration()) > Parameters.alpha}

            # Line 14
            if len(k_2) != 0:
                # Line 15
                f = lambda x: (case.get_event(x).time - case.get_event(x - 1).time - avg[x]) / (std[x])
                labels[case.case_id] = case.get_event(max([k for k in k_2], key=f)).act
            # Line 16
            else:
                # Line 17
                f = lambda x: (case.get_event(x).time - case.get_event(x - 1).time)
                labels[case.case_id] = case.get_event(max([k for k in k_1], key=f)).act
            # Line 18
        # Line 19
    # Line 20

    return labels
