## Coding Standards

### Unit Tests
We use [PyTest](https://docs.pytest.org/en/latest/) to execute tests. You can install pytest by `pip install pytest`. As some of the tests require initialization of the distributed backend, GPUs are needed to execute these tests.

To set up the environment for unit testing, first change your current directory to the root directory of your local ColossalAI repository, then run
```bash
pip install -r requirements/requirements-test.txt
```
If you encounter an error telling "Could not find a version that satisfies the requirement fbgemm-gpu==0.2.0", please downgrade your python version to 3.8 or 3.9 and try again.

If you only want to run CPU tests, you can run

```bash
pytest -m cpu tests/
```

If you have 8 GPUs on your machine, you can run the full test

```bash
pytest tests/
```

If you do not have 8 GPUs on your machine, do not worry. Unit testing will be automatically conducted when you put up a pull request to the main branch.


### Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style meets our requirements. We use pre-commit hook to make sure the code is aligned with the writing standard. To set up the code style checking, you need to follow the steps below.

```shell
# these commands are executed under the Colossal-AI directory
pip install pre-commit
pre-commit install
```

Code format checking will be automatically executed when you commit your changes.
