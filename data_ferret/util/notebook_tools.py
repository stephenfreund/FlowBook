import asyncio
import difflib
import json
from typing import Callable, ContextManager, Dict, List

from data_ferret.lsp.session import LspSession
from data_ferret.lsp.utils import (
    as_uri,
    cell_uri_to_code,
    cell_uri_to_index,
    last_position_in_string,
    last_position_of_symbol,
    abbrev_uri,
    cell_uri_to_id,
)
from dataclasses import dataclass
from pydantic import BaseModel, Field
from agents import Agent, FunctionTool, Runner, Tool, function_tool, RunContextWrapper
from pathlib import Path
from nbformat import read, NotebookNode
from data_ferret.util.dependencies import CellDependencies, analyze_notebook
from data_ferret.util.ferret_metadata import ProfileData, OptimizationPotential, FerretMetadata
from data_ferret.util.output import log, timer, error
import textwrap

import ast


class CellContents(BaseModel):
    id: str = Field(title="The id of the cell containing the code")
    code: str = Field(title="The code of the cell")

    def __str__(self):
        return f"CellContents(id={self.id}, code={self.code[:20]}...)"

class FunctionContents(BaseModel):
    cell_id: str = Field(title="The id of the cell containing the function")
    function_name: str = Field(title="The name of the function")
    code: str = Field(title="The code of the function")

class NotebookTools(LspSession):
    def __init__(
        self,
        notebook: NotebookNode,
        can_edit: bool = False,
        on_cell_modified: Callable[[str, str], None] | None = None,
    ):
        super().__init__(root=Path("/tmp/"))
        self.notebook_path = Path("/tmp/notebook.ipynb")
        self.notebook = notebook
        self.on_cell_modified = on_cell_modified
        self.can_edit = can_edit
        self.dependencies: Dict[str, CellDependencies] = {}
        self.run_analysis()

    def __enter__(self):
        result = super().__enter__()
        self.initialize()
        self.cell_uris = self.open_notebook_document(self.notebook_path, self.notebook)
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)


    def run_analysis(self):
        with timer(key="tool:run_analysis", message="run_analysis()"):
            self.dependencies = analyze_notebook(self.notebook)
            return self.dependencies

    def get_dependencies(self, cell_id: str) -> CellDependencies:
        with timer(key="tool:get_dependencies", message=f"get_dependencies({cell_id})"):
            if cell_id not in self.dependencies:
                raise ValueError(f"Cell {cell_id} not found")
            return self.dependencies[cell_id]

    def notebook_changed(self):
        log(f"notebook_changed")
        self.run_analysis()


    def tools(self, include_profile: bool = False) -> list[Tool]:

        @function_tool
        def get_source(cell_id: str) -> CellContents:
            """
            Returns the source code of the cell with the given id.
            """
            with timer(key="tool:get_source", message=f"get_source({cell_id})"):
                result = self.get_source(cell_id)
                if result is None:
                    raise ValueError(f"Cell {cell_id} not found")
                return result

        @function_tool
        def set_source(cell_id: str, source: str) -> CellContents:
            """
            Sets the source code of the cell with the given id.
            """
            with timer(key="tool:set_source", message=f"set_source({cell_id}, {repr(source)[:20]}...)"):
                source = source.rstrip()
                result = self.set_source(cell_id, source)
                if result is None:
                    raise ValueError(f"Cell {cell_id} not found")
                self.notebook_changed()
                return result

        @function_tool
        def get_cells_for_definition(cell_id: str, symbol: str) -> List[CellContents]:
            """
            Returns the source for the cell containing the definition of the given symbol access in the given cell.
            """
            with timer(key="tool:get_cells_for_definition", message=f"get_cells_for_definition({cell_id}, {symbol})"):
                result = self.get_cells_for_definition(cell_id, symbol)
                if result is None:
                    raise ValueError(f"Definition of {symbol} not found")
                log(
                    f"-> {[c.id for c in result]}"
                )
                return result

        @function_tool
        def get_profile(cell_id: str) -> ProfileData:
            """Get the Scalene profile report for a cell."""
            with timer(key="tool:get_profile", message=f"get_profile({cell_id})"):
                cell = self.get_cell_by_id(cell_id)
                if cell is None:
                    raise ValueError(f"Cell {cell_id} not found")
                data = FerretMetadata.from_cell(cell)
                if data.profile is None:
                    raise ValueError(f"There is no profile information for {cell_id}.")
                return data.profile

        @function_tool
        def get_optimization_potential(cell_id: str) -> OptimizationPotential:
            """Get the optimization potential for a cell, including the highest priority
            concrete suggestions for how to optimize the cell.
            """
            with timer(key="tool:get_optimization_potential", message=f"get_optimization_potential({cell_id})"):
                cell = self.get_cell_by_id(cell_id)
                if cell is None:
                    raise ValueError(f"Cell {cell_id} not found")
                data = FerretMetadata.from_cell(cell)
                if data.optimization_potential is None:
                    raise ValueError(f"There is no optimization potential information for {cell_id}.")
                return data.optimization_potential

        @function_tool
        def get_input_variables(cell_id: str) -> Dict[str, str]:
            """Get the input variables and their types for a cell."""
            with timer(key="tool:get_input_variables", message=f"get_input_variables({cell_id})"):
                cell = self.get_cell_by_id(cell_id)
                if cell is None:
                    raise ValueError(f"Cell {cell_id} not found")
                data = FerretMetadata.from_cell(cell)
                if data.profile is None:
                    raise ValueError(f"There is no variable information for {cell_id}.")
                return data.profile.env
 
        tools: List[Tool] = [
            get_source,
            get_cells_for_definition,
            # get_cell_ids,
        ]
        if self.can_edit:
            tools.append(set_source)
        if include_profile:
            tools.extend([get_profile, get_input_variables])
        return tools

    ###

    def get_cell_by_id(self, cell_id: str) -> NotebookNode | None:
        for cell in self.notebook["cells"]:
            if cell["id"] == cell_id:
                return cell
        return None

    def get_source(self, cell_id: str) -> CellContents | None:
        cell = self.get_cell_by_id(cell_id)
        if cell is None:
            return None
        source = "".join(cell["source"])
        return CellContents(id=cell["id"], code=source)

    def get_source_for_function(self, cell_id: str, function_name: str) -> CellContents | None:
        cell = self.get_cell_by_id(cell_id)
        if cell is None:
            return None

        source_code = "".join(cell["source"])
        try:
            parsed = ast.parse(source_code)
            func_code = None
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract source lines for the function definition
                    # ast nodes have "lineno" and (for python>=3.8) "end_lineno"
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        start = node.lineno - 1
                        end = node.end_lineno
                        lines = source_code.splitlines()
                        func_code = "\n".join(lines[start:end])
                    else:
                        # fallback, just return full source
                        func_code = ast.get_source_segment(source_code, node) or source_code
                    break
            if func_code is None:
                raise ValueError(f"Function {function_name} not found in cell {cell_id}")
            else:
                return FunctionContents(cell_id=cell["id"], function_name=function_name, code=func_code)
        except Exception:
            raise ValueError(f"Error parsing function {function_name} in cell {cell_id}")

    def sanity_check_source(self, cell: NotebookNode, source: str):
        """
        If the source uses joblib or Futures, raise an error.
        """
        return
        # if "import joblib" in source or "from joblib" in source:
        #     raise ValueError("Joblib is not allowed -- rewrite the code to not use joblib")

    def set_source(self, cell_id: str, source: str) -> CellContents | None:

        cell = self.get_cell_by_id(cell_id)
        if cell is None:
            return None

        self.sanity_check_source(cell, source)

        cell["source"] = source
        cell_uri = as_uri(self.notebook_path, cell_id)
        if self.on_cell_modified is not None:
            self.on_cell_modified(cell_id, source)
        payload = {
            "textDocument": {"uri": cell_uri, "version": 1},
            "contentChanges": [{"text": source}],
        }

        self.notify_did_change_text_document(payload)
        return CellContents(id=cell["id"], code=source)

    def get_cells_for_definition(
        self, cell_id: str, symbol: str
    ) -> List[CellContents] | None:
        """
        Returns the source for the cells containing the
        definition of the given symbol access in the given cell.
        """
        cell_uri = as_uri(self.notebook_path, cell_id)
        # index = cell_uri_to_index(self.notebook, cell_uri)
        # # set col, and line for first occurence of symbol
        # line, col = last_position_of_symbol(self.notebook, cell_uri, symbol)

        source = self.get_source(cell_id)
        log(f"get_source({cell_id}) -> ...")
        # for line, code in enumerate(source.code.split("\n")):
        #     error(f"{line:03d}: {code}")

        if source is None:
            return None
        line, col = last_position_in_string(source.code, symbol)
        log(f"last_position_in_string({cell_id}, {symbol}) -> {line}, {col}")

        if line == -1 or col == -1:
            return None
        # log(f"line: {line}, col: {col}")
        # log(self.notebook["cells"][index]["source"].split("\n")[line])
        # log(" " * col + "^")
        payload = {
            "textDocument": {"uri": cell_uri},
            "position": {"line": line, "character": col},
        }
        try:
            # with timer(message=f"text_document_definition"):
            # log(payload)
            result = self.text_document_definition(payload)
        except Exception as e:
            # error(e)
            # error(json.dumps(payload, indent=2))
            # cell = self.get_cell_by_id(cell_id)
            # for line, code in enumerate(cell["source"].split("\n")):
            #     error(f"{line:03d}: {code}")
            return None
        if result is None:
            # log("Not found")
            return None
        else:
            try:
                def_cells = []
                for item in result:
                    def_uri = item["uri"]
                    # with timer(message=f"cell_uri_to_code({def_uri})"):
                    def_cell = cell_uri_to_code(self.notebook, def_uri)
                    def_cells.append(
                        CellContents(id=cell_uri_to_id(def_uri), code=def_cell)
                    )
                return def_cells
            except Exception as e:
                # error(e)
                # error(json.dumps(payload, indent=2))
                # cell = self.get_cell_by_id(cell_id)
                # for line, code in enumerate(cell["source"].split("\n")):
                #     error(f"{line:03d}: {code}")
                return None


if __name__ == "__main__":
    from agents import Agent, Runner, Tool, function_tool, RunContextWrapper
    import nbformat

    async def test_bad():
        nb = nbformat.read(Path("examples/bad.ipynb"), as_version=4)
        with NotebookTools(nb) as tools:
            print(tools.get_cells_for_definition("f11e", "dropna"))

    async def test_get_cells_for_definition():

        source = [
            "params_cat = {'iterations': 746, 'learning_rate': 0.2702240852251232, 'depth': 7, 'l2_leaf_reg': 0.005010078257154434, 'border_count': 250, 'subsample': 0.6439427477473476, 'random_strength': 4.895400575086264}\n",
            "#Best MAPE: 0.00834734201702038\n",
            "\n",
            "# Define MAPE metric\n",
            "def mape(y_true, y_pred):\n",
            "    return mean_absolute_percentage_error(y_true, y_pred)*100\n",
            "\n",
            "# Cross-validation for CatBoostRegressor\n",
            "def cross_val_catboost_mape(X, y, test, n_splits=5, **params_cat):\n",
            "    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n",
            "    mape_scores = []\n",
            "    preds = []\n",
            "\n",
            "    for train_index, valid_index in kf.split(X):\n",
            "        # Ensure data types for indexing\n",
            "        if isinstance(X, pd.DataFrame):\n",
            "            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]\n",
            "            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]\n",
            "        else:\n",
            "            X_train, X_valid = X[train_index], X[valid_index]\n",
            "            y_train, y_valid = y[train_index], y[valid_index]\n",
            "\n",
            "        # Initialize and train the model\n",
            "        cat_model = CatBoostRegressor(random_state=42, silent=True, **params_cat)\n",
            "        cat_model.fit(X_train, y_train)\n",
            "\n",
            "        # Predictions and evaluation\n",
            "        y_pred = cat_model.predict(X_valid)\n",
            "        score = mape(y_valid, y_pred)\n",
            "        mape_scores.append(score)\n",
            "\n",
            "        # Predict on the test set\n",
            "        preds.append(cat_model.predict(test))\n",
            "\n",
            "    # Average predictions over all folds\n",
            "    test_preds_mean = np.mean(preds, axis=0)\n",
            "\n",
            "    return np.mean(mape_scores), test_preds_mean\n",
            "\n",
            "# Example usage\n",
            "model_params = {\n",
            '    "iterations": 500,\n',
            '    "learning_rate": 0.05,\n',
            '    "depth": 6,\n',
            '    "loss_function": "MAPE"\n',
            "}\n",
            "\n",
            "average_mape, catboost_preds = cross_val_catboost_mape(X, y, test, n_splits=5, **params_cat)\n",
            "\n",
            'print(f"Average MAPE across folds: {average_mape:.4f}")\n',
            "\n",
            "# Save predictions for submission\n",
            "submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1(catboost_preds).round()})\n",
            "print(submission.head())\n",
            "submission.to_csv('submission_catboost.csv', index=False)\n",
            "# take the average of the three models, write to submission.csv\n",
            "submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1((lgb_preds + xgb_preds + catboost_preds) / 3).round()})\n",
            "submission.to_csv('submission.csv', index=False)\n",
        ]

        source1 = [
            "from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n",
            "# Initialize LabelEncoder\n",
            "label_encoders = {col: LabelEncoder() for col in cat_cols}\n",
            "\n",
            "# Apply LabelEncoder to each categorical column\n",
            "for col in cat_cols:\n",
            "    train_data[col] = label_encoders[col].fit_transform(train_data[col])\n",
            "    test_data[col] = label_encoders[col].transform(test_data[col])\n",
        ]

        fake_notebook = NotebookNode(
            cells=[
                {
                    "cell_type": "code",
                    "id": "cell_0",
                    "source": "".join(source1),
                },
                {
                    "cell_type": "code",
                    "id": "cell_1",
                    "source": "".join(source),
                },
            ]
        )
        print(
            last_position_of_symbol(
                fake_notebook, "cell_1#cell_1", "cross_val_catboost_mape"
            )
        )

        with NotebookTools(fake_notebook) as tools:
            print(tools.get_cells_for_definition("cell_1", "cross_val_catboost_mape"))

    async def test_basic():

        notebook = NotebookNode(
            cells=[
                {
                    "cell_type": "code",
                    "id": "cell_1",
                    "source": "print('Hello, world!')\nx = 1\n",
                },
                {
                    "cell_type": "code",
                    "id": "cell_2",
                    "source": "print('Hello, world!')\nx\nx\n",
                },
            ]
        )

        with NotebookTools(notebook) as tools:

            print(tools.get_source("cell_1"))
            print(tools.get_cells_for_definition("cell_2", "x"))

            agent = Agent[str](
                name="agent",
                tools=tools.tools(),
                instructions="""\
        You are a data scientist working on a Jupyter notebook.  
        You can view and modify notebook cells.  
        You can call tools to get more information.  
        Examine all function calls and their definitions.
        """,
                model="gpt-4o-mini",
            )

            result = await Runner.run(
                agent, input="What is the source code of the second cell?"
            )
            print(result.final_output)

            result = await Runner.run(
                agent,
                input="Which cell has the definition of x used in the second cell?",
            )
            print(result.final_output)

    async def async_main():
        # await test_bad()
        await test_get_cells_for_definition()
        # await test_basic()

    def main():
        asyncio.run(async_main())

    main()
