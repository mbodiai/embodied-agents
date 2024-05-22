# Contributing

## Development Practices

### Git Workflow

To ensure a clean and linear project history, we enforce specific Git practices:

### Commit Messages

- Always use gitlens (preferred), easycode, or write good commit messages.


### Pulling Changes

Always try to pull with `--ff-only` or `--rebase` when pulling changes from the `main` branch:

```bash
git pull --rebase
```

We recommend setting up your Git configuration as:

```bash
git config --global pull.ff only

git config --global pull.rebase true

```

### Style Guide

1. Lint with Ruff
2. Place tests in a dedicated tests/ directory, with each test file named with a test_ prefix.
3. Prefix all test function names with test_ to enable Pytest to automatically identify and execute them.
4. Use Google-style docstrings to document all public classes, methods, functions, and modules.
5. Include an example usage in the docstring for each important class, method, and function.
6. Pydantic models should be tested for json serialization, deserialization, and saving to h5 files.
