import nbformat
from nbclient import NotebookClient
import inspect
import json


class NBRepro:
    def __init__(
        self, notebook_path, output_path=None, kernel_name="python3", write_output=True
    ):
        self.notebook_path = notebook_path
        if write_output:
            if output_path is None:
                output_path = notebook_path.replace(".ipynb", "-out.ipynb")
            self.output_path = output_path
        else:
            self.output_path = None
        self.kernel_name = kernel_name
        self._nb = None
        self._rerun_nb = None

        self.registration_code = ""
        self.type_analyzers = {}
        self.output_comparators = {}

    def read_nb(self):
        with open(self.notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        return nb

    @property
    def nb(self):
        if self._nb is None:
            self._nb = self.read_nb()
        return self._nb

    @property
    def rerun_nb(self):
        if self._rerun_nb is None:
            self._rerun_nb = self.rerun()
        return self._rerun_nb

    def register_type_analyzer(self, type_or_predicate, analyzer_func):
        self.type_analyzers[type_or_predicate] = analyzer_func

    def register_output_comparator(self, type_name, comparator_func):
        self.output_comparators[type_name] = comparator_func

    def rerun(self):
        # Read setup code
        setup_code = "import type_analysis\ntype_analysis.setup_type_formatter()\n"

        nb = self.read_nb()

        client = NotebookClient(nb, timeout=60, kernel_name=self.kernel_name)

        with client.setup_kernel():
            # execute setup code
            client.kc.execute(setup_code, silent=False, store_history=False)
            self.register_analyzer_functions(client)

            # execute cell by cell
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    client.execute_cell(cell, cell_index=idx)

        # write output if needed
        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                nbformat.write(nb, f)

        return nb

    def compare(self):
        before_nb = self.nb
        after_nb = self.rerun_nb

        comparison_results = []

        # Code here. Ideally, we use registered output comparators based on types

        return comparison_results

    def get_cell_type_info(self, cell_or_idx):
        if isinstance(cell_or_idx, int):
            cell = self.rerun_nb.cells[cell_or_idx]
        else:
            cell = cell_or_idx
        type_infos = []
        if cell.cell_type == "code":
            for output in cell.outputs:
                if output.output_type == "execute_result":
                    type_info = json.loads(
                        output.get("data", {}).get("application/x-python-type")
                    )
                    if type_info:
                        type_infos.append(type_info)
        return type_infos

    def get_type_info(self):
        type_infos = {}
        for idx, cell in enumerate(self.rerun_nb.cells):
            if cell.cell_type == "code":
                type_infos[idx] = self.get_cell_type_info(cell)
        return type_infos

    def register_analyzer_functions(self, client):
        """Register analyzer by extracting source and name from function objects"""

        # we have to run this directly in the kernel, hence the strings
        for type_or_predicate, analyzer_func in self.type_analyzers.items():
            analyzer_name = analyzer_func.__name__
            analyzer_source = inspect.getsource(analyzer_func)

            if inspect.isclass(type_or_predicate):
                # It's a class - just pass it through
                type_name = type_or_predicate.__name__
                registration_code = f"""
# Define the analyzer function
{analyzer_source}

# Register with class (will be converted to predicate automatically)
type_analysis.register_analyzer({type_name}, {analyzer_name})
"""

            else:
                # It's a predicate function
                predicate_name = type_or_predicate.__name__
                predicate_source = inspect.getsource(type_or_predicate)

                registration_code = f"""
# Define the predicate function
{predicate_source}

# Define the analyzer function  
{analyzer_source}

# Register with predicate function
type_analysis.register_analyzer({predicate_name}, {analyzer_name})
"""
            client.kc.execute(registration_code, silent=False, store_history=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nb_repro_start.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    repro = NBRepro(notebook_path)
    repro.rerun()
    print(repro.compare())
