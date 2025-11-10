from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Literal, Union, get_type_hints
from pydantic import BaseModel
import numpy as np
import pandas as pd
import inspect
from collections.abc import Mapping, Sequence, Set

# ─── Type Models ───────────────────────────────────────────────────────────────

INSPECTION_LIMIT = 10


class AtomicType(BaseModel, frozen=True):
    kind: Literal["Atomic"]
    type_name: Literal[
        "int",
        "float",
        "bool",
        "str",
        "None",
        "Any",
        "int32",
        "int64",
        "float32",
        "float64",
        "str96",
    ]

    def __str__(self) -> str:
        return self.type_name


class ArrayType(BaseModel, frozen=True):
    kind: Literal["ndarray"]
    dtype: str
    shape: Tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.kind}[{self.dtype}, shape={self.shape}]"


class DataFrameColumn(BaseModel, frozen=True):
    name: str
    dtype: str


class DataFrameType(BaseModel, frozen=True):
    kind: Literal["DataFrame"]
    n_rows: int
    columns: List[DataFrameColumn]

    def __str__(self) -> str:
        cols = ", ".join(f"'{col.name}': {col.dtype}" for col in self.columns)
        return f"DataFrame[{self.n_rows} rows; {cols}]"


class SeriesType(BaseModel, frozen=True):
    kind: Literal["Series"]
    dtype: str
    length: int

    def __str__(self) -> str:
        return f"Series[{self.dtype}, length={self.length}]"


class DictType(BaseModel, frozen=True):
    kind: Literal["Dict"]
    key_types: List[TypeModel]
    value_types: List[TypeModel]

    def __str__(self) -> str:
        kt = str(self.key_types[0]) if self.key_types else "Unknown"
        vt = str(self.value_types[0]) if self.value_types else "Unknown"
        return f"Dict[{kt}, {vt}]"


class SequenceType(BaseModel, frozen=True):
    kind: Literal["List", "Tuple"]
    element_types: List[TypeModel]

    def __str__(self) -> str:
        et = str(self.element_types[0]) if self.element_types else "Unknown"
        return f"{self.kind}[{et}]"


class SetType(BaseModel, frozen=True):
    kind: Literal["Set"]
    element_types: List[TypeModel]

    def __str__(self) -> str:
        et = str(self.element_types[0]) if self.element_types else "Unknown"
        return f"Set[{et}]"


class ParameterModel(BaseModel, frozen=True):
    name: str
    annotation: Optional[TypeModel]
    default: Optional[Any]

    def __str__(self) -> str:
        part = self.name
        if self.annotation:
            part += f": {self.annotation}"
        if self.default is not None:
            part += f"={self.default!r}"
        return part


class FunctionType(BaseModel, frozen=True):
    kind: Literal["Function"]
    name: str
    parameters: List[ParameterModel]
    return_type: Optional[TypeModel]

    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.parameters)
        ret = str(self.return_type) if self.return_type else "None"
        return f"{self.name}({params}) -> {ret}"


class FallbackType(BaseModel, frozen=True):
    kind: Literal["Class"]
    class_name: str

    def __str__(self) -> str:
        return self.class_name


class UnionType(BaseModel, frozen=True):
    kind: Literal["Union"]
    types: List[TypeModel]

    def __str__(self) -> str:
        types = self.types
        if len(types) == 2:
            names = {str(t) for t in types}
            if "None" in names:
                others = [t for t in types if str(t) != "None"]
                if others:
                    return f"Optional[{others[0]}]"
        inner = ", ".join(str(t) for t in types)
        return f"Union[{inner}]"


TypeModel = Union[
    AtomicType,
    ArrayType,
    DataFrameType,
    SeriesType,
    DictType,
    SequenceType,
    SetType,
    FunctionType,
    FallbackType,
    UnionType,
]

# ─── Model Rebuilds ────────────────────────────────────────────────────────────
AtomicType.model_rebuild()
ArrayType.model_rebuild()
DataFrameType.model_rebuild()
SeriesType.model_rebuild()
DictType.model_rebuild()
SequenceType.model_rebuild()
SetType.model_rebuild()
FunctionType.model_rebuild()
FallbackType.model_rebuild()
UnionType.model_rebuild()

# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_union(models: List[TypeModel]) -> TypeModel:
    unique = []
    for m in models:
        if m not in unique:
            unique.append(m)

    if AtomicType(kind="Atomic", type_name="int64") in unique:
        if AtomicType(kind="Atomic", type_name="int") in unique:
            unique.remove(AtomicType(kind="Atomic", type_name="int64"))

    if AtomicType(kind="Atomic", type_name="float64") in unique:
        if AtomicType(kind="Atomic", type_name="float") in unique:
            unique.remove(AtomicType(kind="Atomic", type_name="float64"))

    if AtomicType(kind="Atomic", type_name="str96") in unique:
        if AtomicType(kind="Atomic", type_name="str") in unique:
            unique.remove(AtomicType(kind="Atomic", type_name="str96"))

    if len(unique) == 1:
        return unique[0]

    return UnionType(kind="Union", types=unique)


# ─── Main Logic ────────────────────────────────────────────────────────────────


def get_type_model(obj: Any) -> TypeModel:
    # --- Type annotations passed directly (e.g., from inspect) ---
    if isinstance(obj, type):
        # numpy scalar classes
        if issubclass(obj, np.generic):
            return AtomicType(kind="Atomic", type_name=np.dtype(obj).name)
        if obj is Any:
            return AtomicType(kind="Atomic", type_name="Any")
        if obj is None or obj is type(None):
            return AtomicType(kind="Atomic", type_name="None")
        if obj in {int, float, bool, str}:
            return AtomicType(kind="Atomic", type_name=obj.__name__)
        return FallbackType(kind="Class", class_name=obj.__name__)

    # --- Atomic instances (including numpy scalars) ---
    if isinstance(obj, np.generic):
        return AtomicType(kind="Atomic", type_name=str(obj.dtype.name))
    if obj is None:
        return AtomicType(kind="Atomic", type_name="None")
    if isinstance(obj, (int, float, bool, str)):
        return AtomicType(kind="Atomic", type_name=type(obj).__name__)

    # --- numpy arrays ---
    if isinstance(obj, np.ndarray):
        return ArrayType(kind="ndarray", dtype=str(obj.dtype), shape=obj.shape)

    # --- pandas DataFrame / Series ---
    if isinstance(obj, pd.DataFrame):
        cols = [
            DataFrameColumn(name=str(col), dtype=str(obj[col].dtype))
            for col in obj.columns
        ]
        return DataFrameType(kind="DataFrame", n_rows=len(obj), columns=cols)
    if isinstance(obj, pd.Series):
        return SeriesType(kind="Series", dtype=str(obj.dtype), length=len(obj))

    # --- mappings ---
    if isinstance(obj, Mapping):
        key_models = (
            [_make_union([get_type_model(k) for k in obj.keys()])] if obj else []
        )
        val_models = (
            [_make_union([get_type_model(v) for v in obj.values()])] if obj else []
        )
        return DictType(kind="Dict", key_types=key_models, value_types=val_models)

    # --- sequences (not str/bytes) ---
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        kind = "List" if isinstance(obj, list) else "Tuple"
        elems = [get_type_model(e) for e in obj[:INSPECTION_LIMIT]]
        elem_union = (
            _make_union(elems)
            if elems
            else FallbackType(kind="Class", class_name="Unknown")
        )
        return SequenceType(kind=kind, element_types=[elem_union])

    # --- sets ---
    if isinstance(obj, Set) and not isinstance(obj, (str, bytes)):
        # Sets don't support slicing, so convert to list first
        elems = [get_type_model(e) for e in list(obj)[:INSPECTION_LIMIT]]
        elem_union = (
            _make_union(elems)
            if elems
            else FallbackType(kind="Class", class_name="Unknown")
        )
        return SetType(kind="Set", element_types=[elem_union])

    # --- callables ---
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        sig = inspect.signature(obj)
        hints = get_type_hints(obj)
        params = []
        for name, param in sig.parameters.items():
            ann = hints.get(name)
            ann_model = get_type_model(ann) if ann is not None else None
            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            params.append(
                ParameterModel(name=name, annotation=ann_model, default=default)
            )
        ret_hint = hints.get("return")
        ret_model = get_type_model(ret_hint) if ret_hint is not None else None
        return FunctionType(
            kind="Function", name=obj.__name__, parameters=params, return_type=ret_model
        )

    # --- fallback ---
    return FallbackType(kind="Class", class_name=obj.__class__.__name__)


# ─── Examples ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # numpy array
    a = np.zeros((2, 3), dtype=np.int16)
    print(repr(get_type_model(a)))  # ArrayType
    print(str(get_type_model(a)))  # "ndarray[int16, shape=(2, 3)]"

    # pandas
    df = pd.DataFrame({"age": [30, 40], "score": [0.8, 0.9]})
    print(str(get_type_model(df)))  # "DataFrame[2 rows; age: int64, score: float64]"

    # Python primitives
    print(str(get_type_model(42)))  # "int"
    print(str(get_type_model(3.14)))  # "float"
    print(str(get_type_model(True)))  # "bool"
    print(str(get_type_model("hello")))  # "str"
    print(str(get_type_model(None)))  # "None"

    # numpy scalars
    print(str(get_type_model(np.int64(7))))  # "int64"
    print(str(get_type_model(np.float64(2.7))))  # "float64"
    print(str(get_type_model(np.str_("foo"))))  # "str"

    # Optional / Any
    print(str(get_type_model(Optional[int])))  # "Optional[int]"
    print(str(get_type_model(Any)))  # "Any"
