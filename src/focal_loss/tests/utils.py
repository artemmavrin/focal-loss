"""Test case utilities"""

import itertools

from absl.testing import parameterized


def named_parameters_with_testcase_names(**kwargs):
    """Generate named parameter dicts with test names for parameterized test
    cases.
    """

    def _named_parameters(**kws):
        """Generate named parameter dicts for parameterized test cases.

        For example,
        _named_parameters(a=[0, 1], b=2) -> {'a': 0, 'b': 2}, {'a': 1, 'b': 2}
        """
        combinations = ([(k, v) for v in (vs if isinstance(vs, list) else [vs])]
                        for k, vs in kws.items())
        return list(map(dict, itertools.product(*combinations)))

    named_parameters = _named_parameters(**kwargs)
    index_padding = len(str(len(named_parameters)))
    for index, keywords in enumerate(named_parameters):
        name = f'_{index:0{index_padding}}'
        for key, value in sorted(keywords.items(), key=lambda pair: pair[0]):
            key = ''.join(filter(str.isalnum, str(key)))
            value = ''.join(filter(str.isalnum, str(value)))
            name += f'_{key}_{value}'
        keywords['testcase_name'] = name
    return parameterized.named_parameters(named_parameters)
