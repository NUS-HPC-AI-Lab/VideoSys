import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import inspect

import pytest

import examples.cogvideox.sample as cogvideox
import examples.latte.sample as latte
import examples.open_sora.sample as open_sora
import examples.open_sora_plan.sample as open_sora_plan


@pytest.mark.parametrize("file", [cogvideox, latte, open_sora, open_sora_plan])
def test_examples(file):
    funcs = inspect.getmembers(file, inspect.isfunction)
    for name, func in funcs:
        try:
            func()
        except Exception as e:
            raise Exception(f"Failed to run {name} in {file.__file__}") from e
