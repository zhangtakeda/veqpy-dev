from veqpy.orchestration import (
    SOURCE_PROJECTION_DISABLE_ENV,
    _resolve_source_projection_policy,
    source_projection_enabled,
)


def test_source_projection_disable_env_turns_off_all_configured_policies(monkeypatch):
    monkeypatch.setenv(SOURCE_PROJECTION_DISABLE_ENV, "1")

    assert not source_projection_enabled()
    for route in ("PI", "PJ1", "PJ2"):
        assert (
            _resolve_source_projection_policy(
                route,
                "psin",
                "uniform",
                has_ip_constraint=False,
                has_beta_constraint=False,
            )
            is None
        )


def test_source_projection_disable_env_is_opt_in(monkeypatch):
    monkeypatch.delenv(SOURCE_PROJECTION_DISABLE_ENV, raising=False)

    assert source_projection_enabled()
    assert _resolve_source_projection_policy(
        "PJ1",
        "psin",
        "uniform",
        has_ip_constraint=False,
        has_beta_constraint=False,
    ) is not None
