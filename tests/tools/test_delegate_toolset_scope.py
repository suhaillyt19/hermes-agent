"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace

from tools.delegate_tool import _strip_blocked_tools, _emit_parent_console

class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

class TestPerTaskToolsetsForwarding:
    def test_resolve_child_toolsets_intersects_explicit_toolsets(self):
        from tools.delegate_tool_toolsets import _resolve_child_toolsets

        parent = SimpleNamespace(enabled_toolsets=["file", "web", "terminal"], disabled_toolsets=[])
        enabled, _disabled = _resolve_child_toolsets(parent, ["file", "web"], "leaf")
        assert enabled == ["file", "web"]

    def test_resolve_child_toolsets_explicit_empty_stays_empty(self):
        from tools.delegate_tool_toolsets import _resolve_child_toolsets

        parent = SimpleNamespace(enabled_toolsets=["file", "web", "terminal"], disabled_toolsets=[])
        enabled, _disabled = _resolve_child_toolsets(parent, [], "leaf")
        assert enabled == []

    def test_normalize_single_goal_carries_top_level_toolsets(self):
        from tools.delegate_tool_tasks import _normalize_task_list

        task_list, err = _normalize_task_list("do the thing", None, None, None, "leaf", 3, toolsets=["file"])
        assert err is None
        assert task_list[0]["toolsets"] == ["file"]

    def test_build_children_forwards_per_task_toolsets(self, monkeypatch):
        from tools import delegate_tool

        seen = {}

        def _fake_build(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", _fake_build)
        monkeypatch.setattr(delegate_tool, "pin_child_service_tier", lambda *_a, **_kw: None)
        monkeypatch.setattr(delegate_tool, "_route_overrides", lambda _route: {})
        route = SimpleNamespace(
            error=None, creds={"model": "m", "provider": "p"}, routing_cfg=None,
            reasoning_effort=None, profile_home=None, report=lambda: None,
        )
        children, err = delegate_tool._build_children(
            [{"goal": "do a sufficiently long thing", "toolsets": ["file"]}],
            [None], [route], top_role="leaf", max_iterations=10,
            parent_agent=SimpleNamespace(), live_deleg_id=None, live_writers=[None],
        )
        assert err is None
        assert len(children) == 1
        assert seen["toolsets"] == ["file"]


class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")

        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""
