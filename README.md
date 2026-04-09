# Python Community Notebooks

## Overview

This repository contains community-maintained notebooks and scripts demonstrating practical use of the [onc](https://github.com/OceanNetworksCanada/api-python-client) Python client and ONC datasets.

The repository is organized by analysis purpose (not by contributor).

### Disclaimer

Community content is provided as-is and is not regulated, endorsed, developed, or maintained by Ocean Networks Canada. Feature requests and bug reports should be directed to the original notebook or script author.

## Repository Structure

Folders are organized by purpose:

- `tutorials/`: Entry-level guided notebooks for learning core ONC and analysis concepts.
- `workflows/`: Reusable method notebooks. They teach repeatable techniques that can be applied to many datasets.
- `case-studies/`: End-to-end, question-driven analyses focused on a specific oceanographic story.
- `data-pipelines/`: Script-first data acquisition and processing workflows, with optional companion notebooks.
- `shared/`: Common helper code, schemas, and templates used across notebooks and scripts.

Notebook listings are maintained in [CATALOG.md](CATALOG.md).

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- ONC API token
- Google account (optional, for running notebooks in Google Colab)

### Setup

For direct Google Colab usage (optional):

- Some notebooks support running in Colab before any local setup.
- Use the notebook links in [CATALOG.md](CATALOG.md) to open the notebook on GitHub, then click the **Open in Colab** badge in the first cell (if present).
- In Colab, set `ONC_TOKEN` using **Secrets** (key icon in the left sidebar): add a secret named `ONC_TOKEN`, then run the notebook token initialization cell. Adding a secret in Google Colab is a one-time effort, but you might need to toggle the notebook access for each new notebook to grant the access.

1. **Clone the repository**
   ```bash
   git clone https://github.com/OceanNetworksCanada/python-community-notebooks.git
   cd python-community-notebooks
   ```

2. **Install common dependencies**
    ```bash
    uv pip install -r requirements.txt
    ```

3. **Create a `.env` file for your token**
   ```bash
   echo "ONC_TOKEN=your_token_here" > .env
   ```

4. **Open any notebook** in Jupyter or VS Code, and run the cells in the notebook.

## Contributing

Contribution guidelines are in [CONTRIBUTING.md](CONTRIBUTING.md).