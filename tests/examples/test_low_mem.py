import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import inspect

import pytest

import examples.cogvideox.sample as cogvideox
import examples.latte.sample as latte
import examples.open_sora.sample as open_sora
import examples.open_sora_plan.sample as open_sora_plan


@pytest.mark.parametrize("file", [open_sora_plan, open_sora, latte, cogvideox])
def test_examples(file):
    funcs = inspect.getmembers(file, inspect.isfunction)
    print(f"Running {len(funcs)} functions in {file.__file__}")
    for name, func in funcs:
        print(f"Running {name} in {file.__file__}")
        if name.startswith("run_low_mem"):
            try:
                func()
            except Exception as e:
                raise Exception(f"Failed to run {name} in {file.__file__}") from e
