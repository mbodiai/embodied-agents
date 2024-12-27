import os
import yaml

# Define the base directory of your Python package
base_dir = "mbodied"
# Define the output directory for API reference markdown files
output_dir = "docs/api"
# Path to the mkdocs.yml configuration file
config_path = "mkdocs.yml"


# Function to create markdown file content
def create_markdown_content(module_path):
    return f"::: {module_path}\n"


# Function to walk through the base directory and generate API reference
def generate_api_reference(base_dir, output_dir):
    api_structure = {}

    # Collect existing .md files
    existing_md_files = set()
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".md"):
                existing_md_files.add(os.path.abspath(os.path.join(root, file)))

    generated_md_files = set()

    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        # Skip directories without an __init__.py file
        if "__init__.py" not in files:
            continue

        for file in files:
            # Skip files that are not Python modules or are named __about__.py
            if file.endswith(".py") and file != "__init__.py" and file != "__about__.py":
                # Get the module path
                module_path = os.path.join(root, file).replace("/", ".").replace("\\", ".").replace(".py", "")

                # Prepare the markdown file path
                relative_path = os.path.relpath(root, base_dir)
                file_name = file.replace(".py", ".md")
                md_path = os.path.join(output_dir, relative_path, file_name)

                # Ensure output directories exist
                os.makedirs(os.path.dirname(md_path), exist_ok=True)

                # Write the content to markdown file
                content = create_markdown_content(module_path)
                with open(md_path, "w") as md_file:
                    md_file.write(content)

                # Add to generated_md_files
                generated_md_files.add(os.path.abspath(md_path))

                # Organize structure for mkdocs.yml nav
                # Split the relative path into sections for nesting
                path_parts = relative_path.split(os.sep) if relative_path != "." else []
                current_level = api_structure
                for part in path_parts:
                    part_title = part.replace("_", " ").title()
                    current_level = current_level.setdefault(part_title, {})

                # Add the file to the correct nested section
                item_title = file.replace(".py", "").replace("_", " ").title()
                current_level[item_title] = md_path.replace("docs/", "")

    # Delete .md files that are no longer generated
    md_files_to_delete = existing_md_files - generated_md_files
    for md_file in md_files_to_delete:
        os.remove(md_file)
        print(f"Removed outdated documentation file: {md_file}")

    # Remove empty directories
    for root, dirs, files in os.walk(output_dir, topdown=False):
        if not dirs and not files:
            os.rmdir(root)
            print(f"Removed empty directory: {root}")

    return api_structure


# Generate API reference markdown files and get the structure
api_structure = generate_api_reference(base_dir, output_dir)


# Function to create the nav list for mkdocs.yml
def create_nav_structure(api_structure):
    def build_nav(structure):
        nav_list = []
        for section, content in sorted(structure.items()):
            if isinstance(content, dict):
                nav_list.append({section: build_nav(content)})
            else:
                nav_list.append({section: content})
        return nav_list

    return build_nav(api_structure)


# Generate the nav structure
nav = create_nav_structure(api_structure)

# Load the existing mkdocs.yml configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Replace the existing API Reference section in nav with the new structure
for index, item in enumerate(config["nav"]):
    if "API Reference" in item:
        config["nav"][index]["API Reference"] = nav
        break

# Write the updated configuration back to mkdocs.yml
with open(config_path, "w") as file:
    yaml.dump(config, file, sort_keys=False)

print("Updated mkdocs.yml with the new API Reference structure.")
