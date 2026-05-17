from pathlib import Path


def test_source_projection_interface_is_removed() -> None:
    forbidden_fragments = (
        "Source" + "Projection" + "Policy",
        "has_" + "projection" + "_policy",
        "heat_" + "projection" + "_coeff",
        "current_" + "projection" + "_coeff",
        "materialize_" + "projected" + "_source_inputs",
        "VEQPY_DISABLE_SOURCE_" + "PROJECTION",
    )
    source_root = Path(__file__).resolve().parents[1] / "veqpy"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text()
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(source_root)}: {fragment}")
    assert offenders == []


def test_layout_stage_binding_shim_has_no_binding_logic() -> None:
    import veqpy.layout.stage_binding as stage_binding
    from veqpy.layout.geometry_binding import build_geometry_stage_runner
    from veqpy.layout.source_binding import build_bound_source_stage_runner

    assert stage_binding.build_geometry_stage_runner is build_geometry_stage_runner
    assert stage_binding.build_bound_source_stage_runner is build_bound_source_stage_runner
