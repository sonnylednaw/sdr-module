# Project Documentation Setup

📖 Generate API documentation using Doxygen for Linux, macOS, and Windows

## Prerequisites

- Python 3.6+
- [Doxygen](https://www.doxygen.nl/) 1.9.1+
- (Optional) [Graphviz](https://graphviz.org/) for diagram generation

## Installation Guide

### 🐧 Linux (Debian/Ubuntu)
Install Doxygen
```bash
sudo apt-get update && sudo apt-get install doxygen
```
Optional: Install Graphviz for diagrams
```bash
sudo apt-get install graphviz
```
Verify installation
```bash
doxygen --version
```


### 🍎 macOS (Homebrew)
Install Doxygen
```bash
brew install doxygen
```

Optional: Install Graphviz for diagrams
```bash
brew install graphviz
```

Verify installation
```bash
doxygen --version
```


### 🪟 Windows
1. **Installer Method**:
   - Download [Windows binary](https://www.doxygen.nl/download.html)
   - Run installer (add Doxygen to PATH during installation)

2. **Chocolatey (Admin PowerShell)**:
    ```powershell
    choco install doxygen
    ```
    Optional: Install Graphviz for diagrams
    ```powershell
    choco install graphviz
    ```
   

3. **WSL (Recommended for advanced users)**:
    - Install Doxygen in WSL using the Linux instructions above

## Usage
1. Generate documentation
    ```bash
    cd doxygen && doxygen Doxyfile
    ```
2. Open `html/index.html` in your browser
   ```bash
    open doxygen/html/index.html
   ```
