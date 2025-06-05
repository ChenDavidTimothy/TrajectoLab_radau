#!/usr/bin/env python3
"""
Generate examples documentation automatically from examples folder structure.

This script scans the examples/ folder and automatically generates RST documentation
for each example, including both README.md content and Python code with syntax highlighting.
"""

from pathlib import Path


def generate_examples_docs():
    """
    Generate examples documentation from the examples folder structure.

    Scans examples/ folder for directories containing:
    - {example_name}.py - Main Python file
    - README.md - Explanation content

    Generates RST files with both README content and highlighted code.
    """
    # Determine project root (docs/source/ -> project_root)
    project_root = Path(__file__).resolve().parent.parent.parent
    examples_dir = project_root / "examples"

    print(f"Project root: {project_root}")
    print(f"Examples directory: {examples_dir}")
    print("Generating examples documentation...")

    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return

    # Create output directory
    docs_examples_dir = Path(__file__).parent / "examples"
    docs_examples_dir.mkdir(exist_ok=True)
    print(f"Output directory: {docs_examples_dir}")

    # Scan for example directories
    example_dirs = [d for d in examples_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    example_dirs.sort()  # Consistent ordering

    if not example_dirs:
        print("❌ No example directories found")
        return

    print(f"Found {len(example_dirs)} example directories")

    generated_examples = []

    # Process each example directory
    for example_dir in example_dirs:
        example_name = example_dir.name
        python_file = example_dir / f"{example_name}.py"
        readme_file = example_dir / "README.md"

        print(f"  Processing: {example_name}")

        # Validate required files exist
        if not python_file.exists():
            print(f"    ⚠ Skipping - missing Python file: {python_file.name}")
            continue

        if not readme_file.exists():
            print("    ⚠ Skipping - missing README.md")
            continue

        # Read README content
        try:
            readme_content = readme_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"    ❌ Error reading README.md: {e}")
            continue

        # Generate RST content for any README content
        rst_content = generate_example_rst(
            example_name=example_name,
            readme_content=readme_content,
            python_file_path=python_file,
            relative_python_path=f"../../../examples/{example_name}/{example_name}.py",
        )

        # Write RST file
        safe_name = example_name.replace("-", "_").replace(" ", "_")
        rst_file = docs_examples_dir / f"{safe_name}.rst"
        rst_file.write_text(rst_content, encoding="utf-8")

        generated_examples.append((safe_name, example_name, readme_content))
        print(f"    ✓ Generated: {rst_file.name}")

    # Generate examples index
    if generated_examples:
        index_content = generate_examples_index(generated_examples)
        index_file = docs_examples_dir / "index.rst"
        index_file.write_text(index_content, encoding="utf-8")
        print(f"  ✓ Generated index: {index_file.name}")

        print("\n✓ Examples documentation complete!")
        print(f"  Generated {len(generated_examples)} example docs in {docs_examples_dir}")
    else:
        print("❌ No valid examples found to document")


def generate_example_rst(
    example_name: str, readme_content: str, python_file_path: Path, relative_python_path: str
) -> str:
    """Generate RST content for a single example."""
    # Create title with proper RST formatting
    title = example_name.replace("_", " ").replace("-", " ").title()
    title_underline = "=" * len(title)

    # Extract first line of README as description (if available)
    readme_lines = readme_content.split("\n")
    description = readme_lines[0] if readme_lines else "TrajectoLab example problem."

    rst_content = f"""{title}
{title_underline}


Code Implementation
-------------------

.. literalinclude:: {relative_python_path}
   :language: python
   :caption: examples/{example_name}/{example_name}.py
   :linenos:

Running This Example
--------------------

.. code-block:: bash

    cd examples/{example_name}
    python {example_name}.py

"""

    return rst_content


def generate_examples_index(generated_examples: list) -> str:
    """Generate the main examples index RST file."""

    index_content = """Examples Gallery
================

TrajectoLab comes with comprehensive example problems demonstrating different optimal control scenarios.
Each example includes detailed explanations and complete, runnable code.

All examples follow the same structure:

* **Problem Description**: Detailed explanation of the physical system and objectives
* **Complete Implementation**: Full Python code with comments
* **Ready to Run**: Just navigate to the example folder and run the Python file

Available Examples
------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

"""

    # Add each generated example to the toctree
    for safe_name, _, _ in generated_examples:
        index_content += f"   {safe_name}\n"

    index_content += """

Running Examples
----------------

Each example can be run directly from its folder:

.. code-block:: bash

    # Navigate to any example
    cd examples/robot_arm
    python robot_arm.py

    # Or run from project root
    python examples/hypersensitive/hypersensitive.py

The examples demonstrate:

"""

    # Add brief descriptions for each example
    for _, example_name, readme_content in generated_examples:
        # Extract first line as brief description
        first_line = readme_content.split("\n")[0].strip()
        if first_line and not first_line.startswith("#"):
            example_title = example_name.replace("_", " ").replace("-", " ").title()
            index_content += f"* **{example_title}**: {first_line}\n"

    return index_content


if __name__ == "__main__":
    generate_examples_docs()
