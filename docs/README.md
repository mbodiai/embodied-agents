# Mbodied Agents Documentation

### Professional Guide to Setting Up Project and Sphinx Documentation

This guide provides step-by-step instructions to reproduce the structure of the project, set up Sphinx for documentation, and organize the `docs` directory. 

#### Step 1: Set Up the Project Directory Structure

Start by creating the main project directory and its subdirectories. This can be achieved using terminal commands.

```bash
# Navigate into main project directory
cd mbodied
```
#### Step 2: Set Up Python Virtual Environment

Set up a Python virtual environment to manage dependencies.

```bash
# Run install.bash
source install.bash

# Run hatch shell
hatch shell

# Install Sphinx and the SphinxAwesome theme
pip install sphinx sphinxawesome-theme
```

#### Step 3: Initialize Sphinx in the `docs` Directory

Navigate to the `docs` directory and set up Sphinx.

```bash
# Navigate to the docs directory
cd docs

# Run Sphinx quickstart and follow the prompts
sphinx-quickstart
```

Follow the prompts in `sphinx-quickstart` to configure basic settings for your Sphinx documentation.

#### Step 4: Set Up Auto-Generation for Sphinx Documentation

Ensure all directories/files you want to include in the documentation have an `__init__.py` file.

```bash

# Generate .rst files for the modules
sphinx-apidoc -o . ../mbodied/
```

This command generates `.rst` files in the `docs` directory, creating the documentation structure for your modules.

#### Step 5: Organize the `docs` Directory

Verify the `docs` directory structure and move generated `.rst` files into appropriate subdirectories. 

Your `docs` directory should look like this after initial setup:

```plaintext
docs/
├── _build/
├── conf.py
├── index.rst
├── make.bat
├── Makefile
├── modules.rst
├── main.rst
├── README.md
├── mbodied.agents.backends.rst
├── mbodied.agents.language.rst
├── mbodied.agents.rst
├── mbodied.agents.sense.rst
├── mbodied.base.rst
├── mbodied.data.rst
├── mbodied.hardware.rst
└── mbodied.types.rst
```

#### Step 6: Configure `conf.py` for Sphinx

Replace the contents of the `conf.py` file in the `docs` directory with the following:

```python
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # Set path to the directory containing docstrings

project = 'Mbodied Agents'
copyright = '2024, mbodi ai team'
author = 'mbodi ai team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]

html_sidebars = {
  "**": ["sidebar_main_nav_links.html", "sidebar_toc.html"]
}

# Select theme for both light and dark mode
pygments_style = "default"
pygments_style_dark = "monokai"

html_permalinks_icon = "<span>∞</span>"
```

#### Step 7: Reorganize the `docs` Directory Structure

Reorganize the `docs` directory to nest `.rst` files properly and create additional directories.

1. **Move `.rst` files:**

   Move all `mbodied.*.rst` files into the `building_blocks` directory.

   ```bash
   mkdir -p docs/building_blocks/{backend,cognitive_agent,controls,hardware_interface,message,recorder,sample_class}
   mv docs/mbodied.*.rst docs/building_blocks/
   ```

2. **Create new directories and move `index.rst`:**

   Create directories for different sections of your documentation.

   ```bash
   mkdir docs/{concepts,contributing,dev_setup,getting_started,glossary,installation,overview}
   ```

3. **Update the `modules.rst` file:**

   Edit `modules.rst` to include a table of contents for the main project.

   ```plaintext
   mbodied
   ==============

   .. toctree::
      :maxdepth: 4

      mbodied
   ```

4. **Rename `main.rst` to `mbodied.rst`:**

   ```bash
   mv docs/main.rst docs/mbodied.rst
   ```

5. **Ensure each new directory has an `index.rst` file:**

   Create an `index.rst` in each of the new directories to provide a starting point for documentation within those sections.

   ```bash
   touch docs/building_blocks/{backend,cognitive_agent,controls,hardware_interface,message,recorder,sample_class}/index.rst
   touch docs/{concepts,contributing,dev_setup,getting_started,glossary,installation,overview}/index.rst
   ```

#### Step 8: Build the Documentation

Navigate to the `docs` directory and build the HTML documentation.

```bash
cd docs
make html
```

Check the `_build/html` directory for the generated documentation.

Ensure the full directory structure is correct by comparing it here:

.
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── assets
│   ├── architecture.jpg
│   ├── demo_gif.gif
│   └── logo.jpeg
├── docs
│   ├── Makefile
│   ├── README.md
│   ├── _build
│   │   ├── doctrees
│   │   └── html
│   ├── _static
│   │   └── custom.css
│   ├── building_blocks
│   │   ├── backend
│   │   ├── cognitive_agent
│   │   ├── controls
│   │   ├── hardware_interface
│   │   ├── index.rst
│   │   ├── message
│   │   ├── recorder
│   │   └── sample_class
│   ├── concepts
│   │   ├── agents
│   │   ├── controls
│   │   ├── index.rst
│   │   └── types
│   ├── conf.py
│   ├── contributing
│   │   └── index.rst
│   ├── dev_setup
│   │   └── index.rst
│   ├── getting_started
│   │   └── index.rst
│   ├── glossary
│   │   └── index.rst
│   ├── index.rst
│   ├── mbodied.agents.backends.rst
│   ├── mbodied.agents.language.rst
│   ├── mbodied.agents.rst
│   ├── mbodied.agents.sense.rst
│   ├── mbodied.base.rst
│   ├── mbodied.data.rst
│   ├── mbodied.hardware.rst
│   ├── mbodied.rst
│   ├── mbodied.types.rst
│   ├── installation
│   │   └── index.rst
│   ├── make.bat
│   ├── modules.rst
│   └── overview
│       └── index.rst
├── examples
│   ├── simple_robot_agent.ipynb
│   ├── simple_robot_agent.py
│   └── simple_robot_agent_with_SimplerEnv.ipynb
├── install.bash
├── pyproject.toml
├── resources
│   ├── locobot.jpeg
│   └── xarm.jpeg
├── src
│   └── mbodied
│       ├── __about__.py
│       ├── __init__.py
│       ├── __pycache__
│       ├── agents
│       ├── base
│       ├── data
│       ├── hardware
│       └── types
├── tests
│   ├── __pycache__
│   │   ├── test_anthropic.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_audio.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_backend.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_messages.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_motions.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_openai.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_recording.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_replaying.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_sample.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_senses.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_serializer.cpython-311-pytest-8.2.2.pyc
│   │   ├── test_sim_interface.cpython-311-pytest-8.2.2.pyc
│   │   └── test_xarm_interface.cpython-311-pytest-8.2.2.pyc
│   ├── test_anthropic.py
│   ├── test_audio.py
│   ├── test_backend.py
│   ├── test_messages.py
│   ├── test_motions.py
│   ├── test_openai.py
│   ├── test_recording.py
│   ├── test_replaying.py
│   ├── test_sample.py
│   ├── test_senses.py
│   ├── test_serializer.py
│   ├── test_sim_interface.py
│   └── test_xarm_interface.py
└── venv
    ├── bin
    │   ├── __pycache__
    │   ├── activate
    │   ├── activate.csh
    │   ├── activate.fish
    │   ├── activate.nu
    │   ├── activate.ps1
    │   ├── activate_this.py
    │   ├── docutils
    │   ├── normalizer
    │   ├── pip
    │   ├── pip-3.8
    │   ├── pip3
    │   ├── pip3.8
    │   ├── pybabel
    │   ├── pygmentize
    │   ├── python -> /Users/beau/.pyenv/versions/3.8.13/bin/python3.8
    │   ├── python3 -> python
    │   ├── python3.8 -> python
    │   ├── rst2html.py
    │   ├── rst2html4.py
    │   ├── rst2html5.py
    │   ├── rst2latex.py
    │   ├── rst2man.py
    │   ├── rst2odt.py
    │   ├── rst2odt_prepstyles.py
    │   ├── rst2pseudoxml.py
    │   ├── rst2s5.py
    │   ├── rst2xetex.py
    │   ├── rst2xml.py
    │   ├── rstpep2html.py
    │   ├── sphinx-apidoc
    │   ├── sphinx-autogen
    │   ├── sphinx-build
    │   ├── sphinx-quickstart
    │   ├── wheel
    │   ├── wheel-3.8
    │   ├── wheel3
    │   └── wheel3.8
    ├── lib
    │   └── python3.8
    └── pyvenv.cfg

### Final Notes

- **Customization**: Customize `index.rst` files and other documentation as needed for your project.
- **Dependencies**: Add any additional dependencies to your virtual environment as required by your project.
- **Continuous Integration**: Consider setting up a CI/CD pipeline to automate documentation builds.

This guide provides a comprehensive setup process for the current project and Sphinx documentation