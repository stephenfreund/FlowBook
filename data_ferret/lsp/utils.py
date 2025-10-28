"""Provides LSP client side utilities for easier testing."""

import os
import pathlib
import platform
import re
from random import choice
from nbformat import NotebookNode

from data_ferret.util.output import error


def normalizecase(path: str) -> str:
    """Fixes 'file' uri or path case for easier testing in windows."""
    if platform.system() == "Windows":
        return path.lower()
    return path


def as_uri(path: pathlib.Path, id: str | None = None) -> str:
    """Return 'file' uri as string."""
    return normalizecase(path.as_uri() + ("#" + id if id else ""))


def cell_id_to_index(notebook: NotebookNode, cell_id: str) -> int:
    for index, cell in enumerate(notebook["cells"]):
        if cell["id"] == cell_id:
            return index
    raise ValueError(f"Cell {cell_id} not found")


def cell_uri_to_index(notebook: NotebookNode, cell_uri: str) -> int:
    id = cell_uri.split("#")[-1]
    return cell_id_to_index(notebook, id)


def cell_uri_to_id(cell_uri: str) -> str:
    return cell_uri.split("#")[-1]


def cell_uri_to_code(notebook: NotebookNode, cell_uri: str) -> str:
    try:
        index = cell_uri_to_index(notebook, cell_uri)
        return "".join(notebook['cells'][index]["source"])
    except Exception as e:
        # error(f"Error getting code for {cell_uri}: {e}")
        return f"Can't get code for {cell_uri}"


def last_position_in_string(string: str, symbol: str) -> tuple[int, int]:
    # Use word boundary regex to match whole words only
    import re

    pattern = r"\b" + re.escape(symbol) + r"\b"

    lines = string.split("\n")
    for line_idx in range(len(lines) - 1, -1, -1):
        line = lines[line_idx]
        match = re.search(pattern, line)
        if match:
            return line_idx, match.start()
    return -1, -1


def last_position_of_symbol(
    notebook: NotebookNode, cell_uri: str, symbol: str
) -> tuple[int, int]:
    index = cell_uri_to_index(notebook, cell_uri)
    source = notebook["cells"][index]["source"]
    return last_position_in_string(source, symbol)


def abbrev_id(notebook: NotebookNode, cell_id: str) -> str:
    index = cell_id_to_index(notebook, cell_id)
    return f"'{index}/{cell_id[0:3]}...{cell_id[-3:]}'"


def abbrev_uri(notebook: NotebookNode, uri: str) -> str:
    id = uri.split("#")[1]
    return abbrev_id(notebook, id)


class StringPattern:
    """Matches string patterns."""

    def __init__(self, pattern):
        self.pattern = pattern

    def __eq__(self, compare):
        """Compares against pattern when possible."""
        if isinstance(compare, str):
            match = re.match(self.pattern, compare)
            return match is not None

        if isinstance(compare, StringPattern):
            return self.pattern == compare.pattern

        return False

    def match(self, test_str):
        """Returns matches if pattern matches are found in the test string."""
        return re.match(self.pattern, test_str)


class PythonFile:
    """Create python file on demand for testing."""

    def __init__(self, contents, root):
        self.contents = contents
        self.basename = "".join(
            choice("abcdefghijklmnopqrstuvwxyz") if i < 8 else ".py" for i in range(9)
        )
        self.fullpath = pathlib.Path(root) / self.basename

    def __enter__(self):
        """Creates a python file for  testing."""
        with open(self.fullpath, "w", encoding="utf8") as py_file:
            py_file.write(self.contents)
        return self

    def __exit__(self, typ, value, _tb):
        """Cleans up and deletes the python file."""
        os.unlink(self.fullpath)


if __name__ == "__main__":
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

    line, col = last_position_in_string("".join(source), "cross_val_catboost_mape")
    print(line, col)
    print(source[line])
    print(" " * (col) + "^")

    fake_notebook = NotebookNode(
        cells=[
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

