# Contributing Guidelines

## Git Workflow 💻
Ensure that you use tools like gitlens (preferred), easycode, or write good commit messages for clarity.
When pulling changes from the main branch, always use `--rebase` or `--ff-only`:

`git pull --rebase`

To set up your Git configuration for a smoother workflow:
```
git config --global pull.ff only
git config --global pull.rebase true
```


### 🌐 Environment Setup

We use [hatch](https://hatch.pypa.io/1.12/) for packaging and managing dependencies.

```console
git clone https://github.com/mbodiai/embodied-agents.git
source install.bash
hatch run pip install '.[audio]'
```


## 🛠️ Style Guide
[Optional VS Code Profile With All the Extensions and Shortcuts You Need](https://vscode.dev/profile/github/dadb33644d0ab9fcdeb1ec686561d070)

### 1. Run linting with Ruff 🧹
Ensure your code is free from linting errors by running Ruff.

### 2. Organize tests in a dedicated directory 📁
Create a parallel test file in the tests/ directory. Name each test file with a test_ prefix.

### 3. Naming test functions 📝
Prefix all test function names with test_ so Pytest can automatically detect and execute them.

### 4. Google-style 📚 docstrings with examples 💡
Use Google-style docstrings to document all public classes, methods, functions, and modules. Example:
```
def example_function(param1, param2):
    """This is a one-line summary of the function.

    More detail here.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.

    Returns:
        bool: Description of the return value.

    Example:
      >>> add(2, 3)
    """
    return True
```    
### 5. Test Pydantic models 🧪
Ensure Pydantic models are thoroughly tested for:
```
JSON serialization
Deserialization
Saving to h5 files Example test cases:
def test_pydantic_model_to_json():
    # Your test code here
    pass

def test_pydantic_model_from_json():
    # Your test code here
    pass

def test_pydantic_model_save_to_h5():
    # Your test code here
    pass
```
Following these guidelines will help maintain clean, well-documented, and tested code. Happy coding! 🚀

