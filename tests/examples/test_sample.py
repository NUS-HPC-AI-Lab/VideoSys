import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import inspect

import pytest

import examples.cogvideox.sample as cogvideox
import examples.latte.sample as latte
import examples.open_sora.sample as open_sora
import examples.open_sora_plan.sample as open_sora_plan
import examples.vchitect.sample as vchitect

files = [cogvideox, latte, open_sora, open_sora_plan, vchitect]
members = []

for file in files:
    for m in inspect.getmembers(file, inspect.isfunction):
        members.append(m)
print(members)


@pytest.mark.parametrize("members", members)
def test_examples(members):
    name, func = members
    try:
        func()
    except Exception as e:
        raise Exception(f"Failed to run {name} in {file.__file__}") from e
