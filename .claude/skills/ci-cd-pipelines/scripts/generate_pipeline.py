import sys
import os

def generate_pipeline(project_type):
    project_type = project_type.lower()
    filename = ".github/workflows/ci.yml"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if project_type == "node":
        content = """name: Node.js CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    steps:
    - uses: actions/checkout@v4
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    - run: npm ci
    - run: npm run build --if-present
    - run: npm test
"""
    elif project_type == "python":
        content = """name: Python CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8
    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Test with pytest
      run: pytest
"""
    elif project_type == "docker":
        content = """name: Docker Image CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build the Docker image
      run: docker build . --file Dockerfile --tag my-image-name:$(date +%s)
"""
    else:
        print(f"Unknown project type: {project_type}. Supported: node, python, docker")
        return

    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Successfully generated: {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_pipeline(sys.argv[1])
    else:
        print("Usage: python generate_pipeline.py <node|python|docker>")