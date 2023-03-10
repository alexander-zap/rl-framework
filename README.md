# Reinforcement Learning Framework

An easy-to-read Reinforcement Learning (RL) framework. Provides standardized interfaces and implementations to various Reinforcement Learning methods and environments. Also this is the main place to start your journey with Reinforcement Learning and learn from tutorials and examples.

## Getting Started

### Activate your development environment

If you are on a UNIX-based OS:
You are fine. Continue with the next step.

If you are on Windows:
Make sure to use a WSL Python interpreter as your development environment, since we require a UNIX-based system underneath Python to run a lot of the environments and algorithms.
For users using PyCharm, see https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html for more information.
For users using Visual Studio Code, see https://code.visualstudio.com/docs/remote/wsl-tutorial and https://code.visualstudio.com/docs/remote/wsl for more information.

### Install all dependencies in your development environment

To set up your local development environment, please run:

    poetry install

Behind the scenes, this creates a virtual environment and installs `rl_framework` along with its dependencies into a new virtualenv. Whenever you run `poetry run <command>`, that `<command>` is actually run inside the virtualenv managed by poetry.

You can now import functions and classes from the module with `import rl_framework`.

### Preparation for pushing your models to the HuggingFace Hub
1. Create an account to HuggingFace and sign in. ➡ https://huggingface.co/join
2. Create a new token with write role. ➡ https://huggingface.co/settings/tokens
3. Store your authentication token from the Hugging Face website. ➡ `huggingface-cli login`


### Testing

We use `pytest` as test framework. To execute the tests, please run

    pytest tests

To run the tests with coverage information, please use

    pytest tests --cov=src --cov-report=html --cov-report=term

and have a look at the `htmlcov` folder, after the tests are done.

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks (`notebooks/`) without running into import errors by running:

    poetry run jupyter notebook

or

    poetry run jupyter-lab

This starts the jupyter server inside the project's virtualenv.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

    poetry add ipykernel
    poetry run python -m ipykernel install --user --name="rl-framework"

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Distribution Package

To build a distribution package (wheel), please use

    python setup.py bdist_wheel

this will clean up the build folder and then run the `bdist_wheel` command.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

    pip install -U pre-commit
    pre-commit install

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## Contact

Alexander Zap (alexander.zap@alexanderthamm.com)

## License

© Alexander Zap
