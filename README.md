# Research Notebook Reproduction

This repo compares saved Jupyter notebook outputs with outputs produced by rerunning the same notebook. It adds type metadata to notebook display results, detects DataFrame libraries, and chooses the right comparison path for each output.

## Flow

1. `final_nb_repro_start.py` opens the input notebook.
2. Before running notebook cells, it installs the formatter from `final_type_analysis.py`.
3. The formatter adds type metadata to each display output.
4. If the output is a DataFrame, the formatter records whether it is Pandas or Polars.
5. The notebook is rerun cell by cell.
6. Saved outputs are compared with rerun outputs.
7. Results report `match`, `confidence`, `comparison_path`, and DataFrame-specific metadata when available.

## Comparison Paths

- Basic Python values use Kobi's comparison logic.
- Pandas DataFrames use Kobi's HTML table parsing and DataFrame comparison path.
- Polars DataFrames use Sang's confidence-based comparison logic.
- NumPy arrays and floats use tolerant comparison logic.

More implementation details are in [information.md](information.md).

## Requirements

Use a Python environment with:

- `nbformat`
- `nbclient`
- `ipykernel`
- `pandas`
- `polars`
- `numpy`
- `beautifulsoup4`

Install them with:

```powershell
pip install nbformat nbclient ipykernel pandas polars numpy beautifulsoup4
```

## Run

From the repo root:

```powershell
python final_nb_repro_start.py <notebook-path.ipynb>
```

On this machine, the local virtual environment command is:

```powershell
.\.venv\Scripts\python.exe final_nb_repro_start.py <notebook-path.ipynb>
```

The script writes a rerun notebook next to the input notebook using the `-out.ipynb` suffix, then prints the comparison results.

## Python Usage

```python
from final_nb_repro_start import NBRepro

repro = NBRepro("<notebook-path.ipynb>")
results = repro.compare()
print(results)
```

To avoid writing a rerun notebook:

```python
repro = NBRepro("<notebook-path.ipynb>", write_output=False)
results = repro.compare()
```
