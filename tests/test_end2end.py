import subprocess

import pytest

scripts = [
    "scripts/opensora/sample_opensora.sh",
]


@pytest.mark.parametrize("script", scripts)
def test_bash_script(script):
    try:
        result = subprocess.run(["bash", script], check=True, capture_output=True, text=True)
        print(f"Output of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script {script} failed with return code {e.returncode}\nError Output: {e.stderr}")
