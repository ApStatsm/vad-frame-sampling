"""Dataset adapters contain labels, data identity and evaluation-specific rules."""

from . import ucf, xd


ADAPTERS = {'ucf': ucf, 'xd': xd}


def get_dataset(key):
    try:
        return ADAPTERS[key]
    except KeyError as error:
        raise ValueError(f'Unknown dataset: {key}') from error
