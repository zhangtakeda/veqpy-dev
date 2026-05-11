import ast
from pathlib import Path


def test_psin_coordinate_update_uses_base_accumulator():
    """Route kernels should integrate psin_r with the base grid integration matrix."""

    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_args: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_update_psin_coordinate":
            continue
        assert len(node.args) == 3
        call_args.append(ast.unparse(node.args[2]))

    assert call_args
    assert set(call_args) == {"accumulator"}
