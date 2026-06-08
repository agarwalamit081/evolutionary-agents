"""Generate test file boilerplate for TypeScript, JavaScript, or Python.
Usage: python generate_test_boilerplate.py <source_file>
"""
import os
import sys


def generate_boilerplate(source_file):
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    ext = os.path.splitext(source_file)[1]

    if ext in (".ts", ".js"):
        test_file = f"{base_name}.test{ext}"
        content = f"""describe('{base_name}', () => {{
  it('should [expected behavior] when [condition]', () => {{
    // Arrange

    // Act

    // Assert
  }});
}});
"""
    elif ext == ".py":
        test_file = f"test_{base_name}.py"
        content = f"""import pytest


def test_{base_name}_happy_path():
    # Arrange
    pass

    # Act
    pass

    # Assert
    pass


def test_{base_name}_edge_case():
    # Arrange
    pass

    # Act & Assert
    with pytest.raises(ExpectedException):
        pass
"""
    else:
        print(f"Unsupported file type: {ext}")
        return

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated: {test_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_boilerplate(sys.argv[1])
    else:
        print("Usage: python generate_test_boilerplate.py <source_file>")
