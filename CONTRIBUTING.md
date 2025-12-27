# Contributing to Arbitrium Framework

First off, thank you for considering contributing to Arbitrium Framework! It's people like you that make the open-source community such a great place.

We welcome any type of contribution, not just code. You can help with:

- **Reporting a bug**
- **Discussing the current state of the code**
- **Submitting a fix**
- **Proposing new features**
- **Becoming a maintainer**

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your forked repository to your local machine:

    ```sh
    git clone https://github.com/YOUR_FORK_USERNAME/arbitrium-core.git
    cd arbitrium-core
    ```

3. **Set up the environment**. We recommend using a virtual environment.

    ```sh
    python -m venv venv
    source venv/bin/activate
    pip install -e .[dev]
    ```

4. **Install the pre-commit hooks**, which will help ensure your code adheres to our style guidelines:

    ```sh
    pre-commit install
    ```

## Making Changes

1. Create a new branch for your feature or bug fix:

    ```sh
    git checkout -b my-feature-branch
    ```

2. Make your changes to the code.
3. As you make changes, ensure you run the quality checks:

    ```sh
    pre-commit run --all-files
    ```

4. If you are adding new functionality, please add tests to cover it. Run the test suite to ensure everything is working:

    ```sh
    pytest
    ```

## Submitting a Pull Request

1. Push your branch to your fork on GitHub:

    ```sh
    git push origin my-feature-branch
    ```

2. Open a **Pull Request** from your branch to the `main` branch of the original Arbitrium Framework repository.
3. In the description of your Pull Request, please explain the changes you have made and reference any related issues.

Once you've submitted your pull request, a maintainer will review it as soon as possible. We may ask for some changes before merging.

Thank you for your contribution!
