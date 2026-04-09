# Running ISOFIT Examples

ISOFIT includes a collection of examples that demonstrate how to use the software for various atmospheric correction tasks. These range from simple analytical examples to complex real-world scenarios. This guide explains how to obtain, set up, and execute these examples on your system.

## Prerequisites

Before running any examples, ensure you have:

1. **ISOFIT installed** - See [Installation](getting_started/installation.md) for setup instructions
2. **Required data files** - Download with: `isofit download all`
3. **Examples available** - Build after downloading with: `isofit build`

???+ info
    Examples will be available under `~/.isofit/examples/` after building. You can override this location using `isofit -p examples /your/custom/path` before downloading. Alternatively, use `isofit -b /path/to/store/downloads all` to download `all` extras to the specified `-b` path. See [Additional Data](extra_downloads/data.md) for more.

## Available Examples

| Example | Location | Description |
|---------|----------|-------------|
| **Santa Monica** | `20151026_SantaMonica/` | AVIRIS-NG data processing with 6S radiative transfer |
| **Pasadena** | `20171108_Pasadena/` | Urban hyperspectral scene analysis |
| **Thermal IR** | `20190806_ThermalIR/` | Thermal infrared data processing |
| **Image Cube (Small)** | `image_cube/small/` | Fast analytical retrieval example |
| **Image Cube (Medium)** | `image_cube/medium/` | More complex retrieval scenario |
| **Prism Multisurface** | `20231110_Prism_Multisurface/` | Multi-component surface model demonstration |
| **AV3 Calibration** | `20250308_AV3Cal_wltest/` | Wavelength testing and calibration |
| **NEON** | `NEON/` | National Ecological Observatory Network data |
| **Lake Mary** | `LakeMary/` | Single-pixel test case |
| **SeaBASS PRISM** | `SeaBASS_prism_001/` | Single-pixel PRISM-water test case |

## Quick Start

### 1. Find Your Examples

Locate where examples are installed:

```bash
cd $(isofit path examples)
ls
```

### 2. Run an Example

Navigate to an example directory and execute its script:

```bash
cd $(isofit path examples)/image_cube/small
bash default.sh
```

Or, specify the full path:

```bash
bash $(isofit path examples)/image_cube/small/default.sh
```

## Execution Methods

### Method 1: Direct Execution (Linux/macOS)

Navigate to an example directory and execute the bash script:

```bash
cd ~/.isofit/examples/image_cube/small
bash default.sh
```

??? tip "Using uv"

    If you are using uv to manage your virtual environment, you can activate the environment via: `.venv/bin/activate`

    This will make the `isofit` command always available which is needed for the scripts.

### Method 1b: Direct Execution (Windows PowerShell)

Navigate to an example directory and execute the PowerShell script:

```powershell
cd C:\Users\YourName\.isofit\examples\image_cube\small
powershell -File default.ps1
```

Or directly execute from PowerShell:

```powershell
powershell -File C:\Users\YourName\.isofit\examples\image_cube\small\default.ps1
```

??? tip "Using uv"
    If you are using uv to manage your virtual environment, you can activate the environment via: `.venv\Scripts\Activate.ps1`

    This will make the `isofit` command always available which is needed for the scripts.

### Method 2: Using Python (All Platforms)

Most examples also provide Python execution scripts:

```bash
cd ~/.isofit/examples/20151026_SantaMonica
python run.py
```

### Method 3: Docker (All Platforms, Recommended)

Docker provides a pre-configured environment with all dependencies. See [Docker](docker.md) for detailed setup instructions.

Quick example:

```bash
docker run --rm -it --shm-size=16gb jammont/isofit bash examples/image_cube/small/analytical.sh
```

### Method 4: Jupyter Notebooks

Launch Jupyter with:

```bash
docker run --rm --shm-size=16gb -p 8888:8888 jammont/isofit
```

Then navigate to `http://localhost:8888` in your browser. See [Docker](docker.md) for more details.

## Windows-Specific Guidance

Windows users have several options for running ISOFIT examples. This section addresses common considerations and solutions.

### Important Note for Windows Users

Bash scripts (`.sh` files) cannot be executed directly on Windows Command Prompt. You have several solutions:

#### Option A: Use Git Bash (Recommended for Development)

If you have Git installed, you already have Git Bash available:

1. **Via ISOFIT CLI**:

   ```bash
   isofit dev bash
   ```

   This launches a Git Bash terminal.

2. **Manually**:

   Find `git-bash.exe` in your Git installation (typically `C:\Users\YourName\.isofit\windows\PortableGit\` if you downloaded it via `isofit download windows` or `isofit download all`)

This terminal can be used to follow the Linux/MacOS instructions above.

???+ warning "Path Separators"
    In Git Bash, use forward slashes (`/`) instead of backslashes (`\`). For example: `/c/Users/YourName/isofit/examples` instead of `C:\Users\YourName\isofit\examples`

??? note "Using uv"
    If your venv is Windows-formatted (eg. ".venv/Scripts" instead of ".venv/bin"), then to activate the environment within Git Bash is done differently: `source .venv/Scripts/activate`

#### Option B: Use Python Scripts

Execute Python versions of the examples instead of bash scripts:

```bash
python C:\Users\YourName\.isofit\examples\20151026_SantaMonica\run.py
```

Or from the example directory:

```bash
cd C:\Users\YourName\.isofit\examples\20151026_SantaMonica
python run.py
```

#### Option C: Use Windows Subsystem for Linux (WSL2)

For users wanting a native-like Linux experience:

1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Setup a Python environment
3. Install ISOFIT and run examples as you would on Linux

???+ info
    WSL2 provides superior performance and access to Linux-native tools. It's recommended for serious ISOFIT development on Windows, though no support is offered by the ISOFIT team.

#### Option D: Use PowerShell Directly

Windows PowerShell users can now execute `.ps1` scripts directly:

```powershell
# Navigate to example directory
cd C:\Users\YourName\.isofit\examples\image_cube\small

# Execute the PowerShell script
powershell -File default.ps1
```

Or run from any location:

```powershell
powershell -File C:\Users\YourName\.isofit\examples\image_cube\small\default.ps1
```

This is the most native Windows approach and doesn't require Git Bash or WSL2.

#### Option F: Docker (Containerized Execution)

Docker Desktop for Windows provides the most reliable, isolated environment:

```bash
docker run --rm -it --shm-size=16gb jammont/isofit bash examples/image_cube/small/default.sh
```

This is the most robust solution for Windows users, as it avoids shell compatibility issues entirely.

### Windows Path Configuration

When running ISOFIT on Windows, path handling requires attention:

**In batch/CMD files**, use backslashes:
```batch
isofit run C:\Users\YourName\.isofit\examples\config.json
```

**In Git Bash**, use forward slashes:
```bash
isofit run /c/Users/YourName/.isofit/examples/config.json
```

**In Python**, use raw strings or forward slashes:
```python
import os
path = r"C:\Users\YourName\.isofit\examples\config.json"
# or
path = "C:/Users/YourName/.isofit/examples/config.json"
```

### Windows Example: Complete Walkthrough

Here's a complete example for Windows users using PowerShell (the simplest approach):

```powershell
# Open PowerShell

# Navigate to examples
cd C:\Users\YourName\.isofit\examples

# Enter a specific example
cd 20151026_SantaMonica

# Run the PowerShell script
powershell -File run.ps1

# Or run the Python version (alternative)
python run.py
```

**Using Git Bash** (if you have Git installed):

```bash
# Open Git Bash

# Navigate to examples
cd $(isofit path examples)

# Enter a specific example
cd 20151026_SantaMonica

# Run the example
bash default.sh

# Or run the Python version
python run.py
```

## Environment Variables

Certain ISOFIT operations are sensitive to thread configuration. When running examples, you may want to set:

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

These are typically included in example scripts but can be set manually before execution.

**In Git Bash:**
```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
bash analytical.sh
```

**In Windows Command Prompt:**
```batch
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
python run.py
```

**In PowerShell:**
```powershell
$env:MKL_NUM_THREADS=1
$env:OMP_NUM_THREADS=1
powershell -File run.ps1
# or
python run.py
```

???+ info
    These environment variables are automatically set in the generated `.ps1`, `.sh`, and `.py` script files, so manual configuration is only needed if running commands directly in the shell.

## Running Multiple Examples

To test that your installation is working correctly, run several examples:

```bash
# Santa Monica example
bash $(isofit path examples)/20151026_SantaMonica/run.sh

# Image cube analytical
bash $(isofit path examples)/image_cube/small/default.sh
```

## Next Steps

After successfully running examples:

1. **Modify parameters** - Edit configuration JSON files to experiment with different settings
2. **Read documentation** - Check [Best Practices](getting_started/best_practices.md) for optimization guidance
3. **Use your data** - Apply the ISOFIT workflow to your own hyperspectral data
4. **Contribute** - Share your own examples with the ISOFIT community
