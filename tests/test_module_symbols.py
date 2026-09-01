from __future__ import annotations

import builtins
from pathlib import Path
import symtable


def _nested_tables(table: symtable.SymbolTable):
    for child in table.get_children():
        yield child
        yield from _nested_tables(child)


def test_studio_modules_have_no_unresolved_global_names() -> None:
    """Catch imports lost while moving delayed GUI callbacks between modules."""
    allowed = set(dir(builtins)) | {
        "__annotations__",
        "__file__",
        "__name__",
        "__package__",
    }
    failures: list[str] = []
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "studio").rglob("*.py")):
        table = symtable.symtable(path.read_text(), str(path), "exec")
        definitions = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
        }
        missing: set[str] = set()
        for child in _nested_tables(table):
            for symbol in child.get_symbols():
                if (
                    symbol.is_global()
                    and symbol.is_referenced()
                    and symbol.get_name() not in definitions | allowed
                ):
                    missing.add(symbol.get_name())
        if missing:
            failures.append(f"{path.relative_to(root)}: {sorted(missing)}")
    assert not failures, "\n".join(failures)
