# Notebook Reproduction Analysis

This repo contains the merged notebook reproduction workflow from Sang's local
Polars/type-analysis work and Kobi's `kobi/jupyter-repro` branch.

## Main Files

- `final_nb_repro_start.py` runs a notebook, captures typed outputs, and compares
  the saved outputs with the rerun outputs.
- `final_type_analysis.py` installs the notebook formatter that records
  `application/x-python-type` and `application/x-python-dataframe` metadata.

## Comparison Rules

- Basic Python output comparison uses Kobi's logic: literal parsing, numeric
  tolerance, NumPy array comparison, direct equality, and string fallback.
- DataFrame kind is detected from `final_type_analysis.py` metadata.
- Polars DataFrames use Sang's local confidence comparison logic.
- Pandas DataFrames use Kobi's BeautifulSoup plus `pandas.read_html` comparison
  path.
- Pandas and Polars results use the same presentation fields:
  `dataframe_kind`, `comparison_path`, `coverage`, `confidence`, and `match`.

## Usage

Run the analysis with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe final_nb_repro_start.py <notebook-path.ipynb>
```

Or from Python:

```python
from final_nb_repro_start import NBRepro

repro = NBRepro("<notebook-path.ipynb>")
results = repro.compare()
print(results)
```

By default, rerun notebooks are written next to the input notebook with
`-out.ipynb` appended to the filename. Use `write_output=False` to run without
writing a new notebook.
