# Python Community Notebooks

## Overview

The **Python Community Notebooks** repository contains a curated collection of Jupyter notebooks demonstrating practical applications of the [onc](https://github.com/OceanNetworksCanada/api-python-client) Python client library. These notebooks showcase practices for data manipulation, analysis, and visualization using Ocean Networks Canada (ONC) datasets.

This repository serves as a learning resource for users at all levels seeking to deepen their understanding of oceanographic data processing and analysis techniques.

### Disclaimer

Community notebooks are provided as-is and are not regulated, endorsed, developed, or maintained by Ocean Networks Canada. All feature requests and bug reports must be directed to the original notebook author.

## Quick Start

### Prerequisites

- Python 3.10+
- An ONC API token (see [Token Configuration](#token-configuration))
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/OceanNetworksCanada/python-community-notebooks.git
   cd python-community-notebooks
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r XXX/requirements.txt # Adjust the path as needed
   ```

4. **Configure your ONC token** (see [Token Configuration](#token-configuration))

5. **Launch Jupyter in the terminal or using your favorite IDE** 
   ```bash
   jupyter notebook
   ```

## Notebook Catalog

### IanTBlack

#### barkley_canyon_vertical_profiler/

- **bacvp_ctd_up_profiles.ipynb**  
  Demonstrates the creation of vertical profile plots for sea water potential density using CTD data.  
  *Keywords:* gsw

- **bacvp_videocam_opencv.ipynb**  
  Illustrates video processing and frame extraction from videocam sources.  
  *Keywords:* opencv-python, video


#### british_columbia_ferries/

- **distance_from_terminals.ipynb**  
  Computes and visualizes time series data for ferry-to-terminal distances using geospatial calculations.  
  *Keywords:* geopy, xarray

- **grid_transit.ipynb**  
  Analyzes transit grid patterns and routing through ferry data.  
  *Keywords:* xarray

- **system_sampling_state.ipynb**  
  Examines sampling state and temporal coverage of ferry system measurements.  
  *Keywords:* 

- **transit_identifier.ipynb**  
  Identifies and classifies distinct transit events within ferry operational data.  
  *Keywords:* 

- **twsb_tsg_clean_no_gaps.ipynb**  
  Performs data quality assurance and gap-filling for TSG (Thermosalinograph) measurements.  
  *Keywords:* xarray


#### multiple_datasets/

- **fraser_river_plume.ipynb**  
  Analyzes and visualizes the transport and evolution of the Fraser River plume using multi-source oceanographic data.  
  *Keywords:* hypercoast, xarray, cartopy, gsw, plume

## Guidelines and Best Practices

The following guidelines are recommended for consistency across contributed notebooks. Individual contributors may maintain their own conventions; consult their notebook documentation for specific implementation details.

### Token Configuration

Authentication to the ONC API requires a personal token. For instructions on obtaining a token, refer to the [api-python-client documentation](https://github.com/OceanNetworksCanada/api-python-client#obtaining-a-token).

#### Recommended: Environment Variable Storage

Store your token as an environment variable rather than hardcoding it in notebooks. The `onc` library (version 2.6.0+) automatically reads the `ONC_TOKEN` environment variable when instantiating the `ONC` class.

**Configuration Steps:**

1. Create a `.env` file in the repository root:
   ```
   ONC_TOKEN=your_token_here
   ```

2. Add the initialization code to your notebook:
   ```python
   from dotenv import load_dotenv
   from onc import ONC
   
   load_dotenv()
   onc = ONC()  # Automatically uses ONC_TOKEN from environment
   ```

The `load_dotenv()` function searches for the `.env` file in the current directory and parent directories, allowing all notebooks to share a single configuration file.

### Dependency Management

#### Virtual Environments

Since different notebooks may require conflicting library versions, **always use virtual environments** to isolate dependencies. You can also use [uv](https://docs.astral.sh/uv/pip/environments/) or IDEs like [VS Code](https://code.visualstudio.com/docs/python/environments) and [PyCharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html) to manage virtual environments.

You may create multiple virtual environments for different notebook sets if needed.

#### Requirements Files

Each contributor should maintain a `requirements.txt` file documenting all external library dependencies. You may maintain:
- A single `requirements.txt` covering all your notebooks
- Multiple `requirements.txt` files for notebook-specific dependencies

#### Installation in Notebooks

Optionally include an initialization cell to install dependencies within a notebook:
```python
!pip install -r ./requirements.txt  # Adjust path as needed
```

## Contributing

Contributions are welcome! When adding new notebooks, please follow these standards:

### Notebook Structure

- Include a descriptive title and markdown cells explaining the analysis objective
- Optionally add a brief overview of the data sources and processing steps
- Document all external dependencies in a `requirements.txt` file
- Use clear variable names and include inline comments for complex calculations
- Add your notebook description to this README under the appropriate category. Common dependency libraries like `numpy`, `pandas`, `matplotlib` and `onc` can be omitted in the keywords.

### Code Organization

Helper functions and shared utilities could be organized in separate Python modules. To import modules from parent directories within notebooks, use the standard `sys` library, or the notebooks magic command `%cd`:

```python
import sys
sys.path.append('..')
import module_name
```

```python
%cd ..
import module_name
```