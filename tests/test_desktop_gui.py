"""Logic and smoke tests for the PySide6 desktop interface."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from stable_set_game.desktop_gui import (
    DesktopConfig,
    GamePage,
    MainWindow,
    SetupPage,
    build_desktop_graph,
    graph_positions,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.mark.parametrize(
    ("graph_type", "n", "expected_nodes"),
    [
        ("Path", 7, 7),
        ("Cycle", 7, 7),
        ("Complete", 5, 5),
        ("Star", 6, 7),
        ("Erdos-Renyi", 8, 8),
        ("3-Regular", 8, 8),
        ("Fork", 7, 8),
    ],
)
def test_build_all_graph_families(graph_type, n, expected_nodes):
    graph, name, layout = build_desktop_graph(
        DesktopConfig(graph_type=graph_type, n=n)
    )
    assert graph.number_of_nodes() == expected_nodes
    assert name
    assert layout


@pytest.mark.parametrize(
    "config",
    [
        DesktopConfig(graph_type="Path", n=1),
        DesktopConfig(graph_type="Cycle", n=2),
        DesktopConfig(graph_type="3-Regular", n=5),
        DesktopConfig(graph_type="Fork", n=2),
        DesktopConfig(n=7, depth=13),
    ],
)
def test_invalid_configs_are_rejected(config):
    with pytest.raises(ValueError):
        build_desktop_graph(config)


def test_positions_cover_every_node_and_are_normalised():
    graph, _name, layout = build_desktop_graph(
        DesktopConfig(graph_type="Fork", n=9)
    )
    positions = graph_positions(graph, layout)
    assert set(positions) == set(graph.nodes)
    assert all(-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 for x, y in positions.values())


def test_setup_defaults(app):
    page = SetupPage()
    assert page.current_config() == DesktopConfig()
    assert page.preview_canvas.graph.number_of_nodes() == 7
    page.close()


def test_main_window_has_setup_and_game_pages(app):
    window = MainWindow()
    assert window.stack.count() == 2
    assert window.stack.currentWidget() is window.setup_page
    window.start_game(DesktopConfig(mode="human_vs_human"))
    assert window.stack.currentWidget() is window.game_page
    window.show_setup()
    assert window.stack.currentWidget() is window.setup_page
    window.close()


def test_human_game_move_undo_and_restart(app):
    page = GamePage()
    page.start_game(
        DesktopConfig(graph_type="Path", n=7, mode="human_vs_human")
    )
    page.play_human_move(0)
    assert page.game.history == [(1, 0)]
    assert page.game.current_player == 2
    page.undo()
    assert page.game.history == []
    assert page.game.current_player == 1
    page.play_human_move(1)
    page.restart()
    assert page.game.history == []
    page.close()


def test_human_side_mapping(app):
    page = GamePage()
    page.start_game(
        DesktopConfig(mode="human_vs_ai", human_player=2, depth=2)
    )
    assert not page._is_human(1)
    assert page._is_human(2)
    assert page._ai_for(1) is page.ai1
    assert page._ai_for(2) is None
    page.close()


def test_ai_vs_ai_uses_step_control(app):
    page = GamePage()
    page.start_game(DesktopConfig(mode="ai_vs_ai", depth=2))
    assert page.next_button.isVisible() is False  # parent page has not been shown
    assert page.next_button.isHidden() is False
    assert not page._is_human(1)
    assert not page._is_human(2)
    page.close()
