# Crash Handling in test_code - Implementation Summary

## Overview
Extended the `test_code` functionality to properly capture and communicate crashes that occur during execution of original or optimized code, including full error messages and stack traces.

## Changes Made

### 1. New BaseModels (`flowbook/kernel/types.py`)

#### ExecutionError
```python
class ExecutionError(BaseModel):
    error_type: str          # Exception type (e.g., "ValueError")
    error_message: str       # Exception message
    traceback: str          # Full formatted stack trace
    code_snippet: Optional[str]  # Code that crashed
```

#### TestCodeSuccess
```python
class TestCodeSuccess(BaseModel):
    status: Literal["success"] = "success"
    diff: DiffResult
    original_duration: float
    modified_duration: float
    speedup: float
```

#### TestCodeOriginalCrash
```python
class TestCodeOriginalCrash(BaseModel):
    status: Literal["original_crash"] = "original_crash"
    error: ExecutionError
    original_duration: Optional[float]
```

#### TestCodeModifiedCrash
```python
class TestCodeModifiedCrash(BaseModel):
    status: Literal["modified_crash"] = "modified_crash"
    error: ExecutionError
    original_duration: float
    modified_duration: Optional[float]
```

#### TestCodeResult (Union Type)
```python
TestCodeResult = Annotated[
    Union[TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash],
    Field(discriminator="status")
]
```

The `status` field is used as a discriminator for Pydantic to automatically deserialize to the correct type.

### 2. Kernel Changes (`flowbook/kernel/flowbook_kernel.py`)

#### Updated `test_code()` method (lines 365-463)
- Wraps original code execution in try-except
- Wraps modified code execution in try-except
- Returns `TestCodeOriginalCrash` if original code crashes
- Returns `TestCodeModifiedCrash` if modified code crashes (original succeeded)
- Returns `TestCodeSuccess` if both codes execute successfully
- Captures full traceback using `traceback.format_exc()`
- Checks `result.error_in_exec` to detect IPython execution errors

#### Updated `_test_code_comm_open()` handler (lines 514-539)
- Simplified error handling since `test_code()` now always returns a structured result
- Always sends `ok=True` to comm with the result
- The result's `status` field discriminates between success/crash types
- Outer try-except only catches unexpected comm parsing errors

### 3. Server Changes

#### `flowbook/server/commands/optimize.py`

**Updated imports** (lines 27-30):
- Added `TestCodeSuccess`, `TestCodeOriginalCrash`, `TestCodeModifiedCrash`

**Updated `ValidationHelper.validate_optimization()` (lines 242-324)**:
- Checks `isinstance(result.result, TestCodeSuccess)` for success case
- Checks `isinstance(result.result, TestCodeOriginalCrash)` for original code crash
- Checks `isinstance(result.result, TestCodeModifiedCrash)` for optimized code crash
- Returns detailed error messages with exception type, message, and formatted traceback
- Clearly distinguishes between "cannot optimize broken code" vs "optimization introduced a bug"

#### `flowbook/server/commands/validate_change.py`

**Updated imports** (lines 14-17):
- Added new crash types for type checking

**Updated result logging** (lines 265-282):
- Checks result type before accessing fields
- Logs appropriate message for each result type
- Extracts and displays error information for crash cases

### 4. Tests (`flowbook/kernel/test_test_code_crashes.py`)

Created comprehensive test suite with 11 tests:

**Crash Scenario Tests**:
1. `test_both_codes_succeed` - Success case
2. `test_original_code_crashes` - Original code crashes (ZeroDivisionError)
3. `test_modified_code_crashes` - Modified code crashes, original succeeds (ValueError)
4. `test_original_code_syntax_error` - Original code has syntax error
5. `test_modified_code_name_error` - Modified code has NameError

**Model Tests**:
6. `test_execution_error_creation` - ExecutionError model creation
7. `test_execution_error_optional_code_snippet` - Optional fields work
8. `test_test_code_success_discriminator` - Success discriminator
9. `test_test_code_original_crash_discriminator` - Original crash discriminator
10. `test_test_code_modified_crash_discriminator` - Modified crash discriminator
11. `test_serialization_deserialization` - Models serialize/deserialize correctly

**All tests pass** ✓

## Benefits

1. **Type Safety**: Discriminated union ensures proper handling of each case at compile time
2. **Clear Semantics**: Each result type has only relevant fields (no optional fields that may or may not exist)
3. **Full Context**: Distinguishes between original crash vs modified crash vs success
4. **Debugging**: Full stack traces and error details preserved
5. **User Experience**: Clear error messages explain what went wrong and where

## Example Usage

### Original Code Crashes
```json
{
    "status": "original_crash",
    "error": {
        "error_type": "ZeroDivisionError",
        "error_message": "division by zero",
        "traceback": "Traceback (most recent call last):\n  File ...",
        "code_snippet": "x = 1/0"
    },
    "original_duration": 0.001
}
```

### Modified Code Crashes
```json
{
    "status": "modified_crash",
    "error": {
        "error_type": "ValueError",
        "error_message": "invalid literal for int()",
        "traceback": "Traceback (most recent call last):\n  File ...",
        "code_snippet": "x = int('invalid')"
    },
    "original_duration": 1.0,
    "modified_duration": 0.01
}
```

### Success
```json
{
    "status": "success",
    "diff": {"differences": {}},
    "original_duration": 1.0,
    "modified_duration": 0.5,
    "speedup": 2.0
}
```

## Pattern Matching in Client Code

```python
if isinstance(result.result, TestCodeSuccess):
    # Both codes work - check diff
    if result.result.diff.differences:
        print("Outputs differ!")
    else:
        print(f"Success! Speedup: {result.result.speedup}x")

elif isinstance(result.result, TestCodeOriginalCrash):
    # Original code is broken
    error = result.result.error
    print(f"Cannot optimize - original crashes: {error.error_type}: {error.error_message}")

elif isinstance(result.result, TestCodeModifiedCrash):
    # Optimization broke the code
    error = result.result.error
    print(f"Optimization introduced bug: {error.error_type}: {error.error_message}")
```

## Files Modified

1. `flowbook/kernel/types.py` - Added 4 new models + union type
2. `flowbook/kernel/flowbook_kernel.py` - Updated test_code() and _test_code_comm_open()
3. `flowbook/server/commands/optimize.py` - Updated validation logic
4. `flowbook/server/commands/validate_change.py` - Updated result logging
5. `flowbook/kernel/test_test_code_crashes.py` - New test file (11 tests)
