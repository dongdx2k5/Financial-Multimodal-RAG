"""Safe Python execution sandbox for Program-of-Thought outputs.

Runs LLM-generated Python code in a restricted namespace with:
- Blocklist validation (no imports, file I/O, exec/eval, system access)
- Whitelist of allowed builtins (math operations only)
- Timeout protection via threading
"""

from __future__ import annotations

import re
import threading
from typing import Any


# ---------------------------------------------------------------------------
# Blocklisted patterns — code containing any of these is REJECTED
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bimport\s+\w+"),             # import statements
    re.compile(r"\bfrom\s+\w+\s+import\b"),    # from X import Y
    re.compile(r"\b__\w+__\b"),                # dunder attributes (__builtins__, etc.)
    re.compile(r"\bexec\s*\("),                # exec()
    re.compile(r"\beval\s*\("),                # eval()
    re.compile(r"\bcompile\s*\("),             # compile()
    re.compile(r"\bopen\s*\("),                # open()
    re.compile(r"\bos\.\w+"),                  # os module access
    re.compile(r"\bsys\.\w+"),                 # sys module access
    re.compile(r"\bsubprocess\b"),             # subprocess
    re.compile(r"\bglobals\s*\("),             # globals()
    re.compile(r"\blocals\s*\("),              # locals()
    re.compile(r"\bgetattr\s*\("),             # getattr()
    re.compile(r"\bsetattr\s*\("),             # setattr()
    re.compile(r"\bdelattr\s*\("),             # delattr()
    re.compile(r"\bbreakpoint\s*\("),          # breakpoint()
    re.compile(r"\binput\s*\("),               # input()
    re.compile(r"\bprint\s*\("),               # print() — unnecessary in sandbox
    re.compile(r"\bexit\s*\("),                # exit()
    re.compile(r"\bquit\s*\("),                # quit()
]

# Whitelist of safe builtins for financial calculations
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "pow": pow,
    "divmod": divmod,
    "True": True,
    "False": False,
    "None": None,
}


def validate_code(code: str) -> tuple[bool, str]:
    """Check code against the blocklist.

    Returns (is_safe, reason).
    """
    if not code or not code.strip():
        return False, "Empty code"

    for pattern in _BLOCKED_PATTERNS:
        match = pattern.search(code)
        if match:
            return False, f"Blocked pattern detected: '{match.group(0)}'"

    return True, "OK"


def execute_python_safely(
    code: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed namespace.

    Parameters
    ----------
    code : str
        Python source code to execute.  The code should assign its final
        result to a variable named ``ans``.
    timeout : int
        Maximum execution time in seconds.

    Returns
    -------
    dict with keys:
        - ``success`` (bool)
        - ``result`` — the value of ``ans`` if success
        - ``variables`` — all non-builtin variables after execution
        - ``error`` — error message if failed
    """
    # 1. Validate
    is_safe, reason = validate_code(code)
    if not is_safe:
        return {"success": False, "error": f"Code validation failed: {reason}", "result": None, "variables": {}}

    # 2. Prepare restricted namespace
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

    # 3. Execute with timeout
    exec_error: list[Exception] = []

    def _run() -> None:
        try:
            exec(code, namespace)  # noqa: S102 — intentional sandboxed exec
        except Exception as e:
            exec_error.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return {
            "success": False,
            "error": f"Execution timed out after {timeout}s",
            "result": None,
            "variables": {},
        }

    if exec_error:
        return {
            "success": False,
            "error": str(exec_error[0]),
            "result": None,
            "variables": {},
        }

    # 4. Extract results
    # Filter out builtins and internal keys
    user_vars = {
        k: v for k, v in namespace.items()
        if k != "__builtins__" and not k.startswith("_")
    }

    result = namespace.get("ans", None)

    return {
        "success": True,
        "result": result,
        "variables": user_vars,
        "error": None,
    }
