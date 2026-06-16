# Contributing

Thank you for contributing to Python Community Notebooks.

## Where To Place Content

Place your work under the best matching top-level category:

- `tutorials`: Entry-level learning notebooks.
- `workflows`: Reusable, method-focused notebooks.
- `case-studies`: End-to-end, question-driven analyses.
- `data-pipelines`: Script-first acquisition/processing workflows with optional notebooks.
- `shared`: Common helper code, schemas, and templates used across notebooks and scripts.

## Colab-compatible notebooks

We encourage contributors to create notebooks that are compatible with Google Colab whenever possible.

- Include an **"Open in Colab"** badge in the first cell of the notebook. The markdown link is `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OceanNetworksCanada/python-community-notebooks/blob/main/AAA/BBB/CCC.ipynb)`. 
- Handle dependencies in the notebook.
- Add the bootstrap cell for token initialization and sys.path update before importing libraries.
- You can open/test notebooks in Colab using any of the following:
    - Open the notebook in GitHub, then replace `github.com` with [githubtocolab.com](https://githubtocolab.com/) in the URL.
    - Open [Google Colab](https://colab.research.google.com/), choose **GitHub**, and paste the notebook GitHub URL.
    - Build a direct Colab URL in this format: `https://colab.research.google.com/github/OceanNetworksCanada/python-community-notebooks/blob/<branch-or-main>/AAA/BBB/CCC.ipynb`.
    - Click the **Open in Colab** badge in the first cell and update the branch name in the url for non-main branches.

## Dependency Management

- Prefer local dependency files inside related notebooks/scripts. Common libraries like `matplotlib`, `numpy`, and `pandas` do not need to be listed per notebook, as they are included in the root requirements.txt and Google Colab. 
- Non-versioned dependencies are easier to maintain and usually better for simple notebooks that rely on stable, common packages.
- Versioned dependencies are better when reproducibility matters or when a notebook depends on fragile scientific/geospatial stacks.
- Recommended default: avoid strict pinning for simple community notebooks, but pin versions when a notebook is known to break across package releases or when results must be reproducible.

Recommended dependency initialization cell (**always** install onc library because Google Colab does not have it by default):

```python
!uv pip install -q onc xxx yyy
```

## Bootstrap
### Token Initialization

- Use `ONC_TOKEN` from Colab secrets (the "key" icon in the left sidebar) or local `.env`.
- Do not hardcode API tokens in notebooks or scripts.

### Helper Files In Colab

If a notebook depends on helper scripts or shared modules in this repository, put them in the `shared` folder, and use a bootstrap cell in the notebooks. This keeps local runs clean while enabling imports in Colab.

### Recommended bootstrap cell

```python
# Bootstrap cell
import sys
from pathlib import Path
import os

def in_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_repo_for_shared_imports():
    """
    Locate (or clone, in Colab) the notebook repository and add it to ``sys.path``.
    Assume that only the root directory has the ``shared`` directory.

    Notes:
        - In Colab, this ensures the repository exists at
          ``/content/python-community-notebooks`` by cloning it if needed.
        - In local environments, this searches upward from the current working
          directory for ``shared`` to identify the repo root.
        - If a repo root is found, it is prepended to ``sys.path`` to enable
          imports from the repository (e.g., ``shared`` modules).
    """
    repo_dir: Path | None = None

    if in_colab():
        repo_dir = Path('/content/python-community-notebooks')
        if not repo_dir.exists():
            !git clone --depth 1 https://github.com/OceanNetworksCanada/python-community-notebooks.git /content/python-community-notebooks
    else:
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / 'shared').exists():
                repo_dir = candidate
                break

    if repo_dir is not None and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))



def init_token():   
    """
    Initialize the ONC_TOKEN environment variable.

    Notes:
        - If in Colab, add your ONC_TOKEN secrets by clicking the key icon on the left sidebar
        - If in local, create .env file in the root directory, and add ONC_TOKEN=XXX in your .env file
    """
    if in_colab():
        from google.colab import userdata
        os.environ['ONC_TOKEN'] = userdata.get('ONC_TOKEN')
    else:
        from dotenv import load_dotenv
        load_dotenv()

init_token()
setup_repo_for_shared_imports()
```

If helper files are not needed, remove `setup_repo_for_shared_imports()` definition and usage from the cell.

## Catalog Updates

When adding or moving notebooks, update [CATALOG.md](CATALOG.md):

- Add a short description.
- Add keywords (optional but recommended).

