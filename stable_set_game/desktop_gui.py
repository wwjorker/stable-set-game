"""Professional PySide6 desktop interface for the Stable Set Game.

The research engine, AI, graph generators and experiments remain independent
of this module.  The older Matplotlib interface is still available from
``stable_set_game.gui`` as a lightweight fallback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx
from PySide6.QtCore import QObject, QPointF, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from stable_set_game.ai import AIPlayer, SearchResult
from stable_set_game.engine import StableSetGame
from stable_set_game.generators import (
    complete_graph,
    cycle_graph,
    erdos_renyi_graph,
    fork_graph,
    path_graph,
    random_regular_graph,
    star_graph,
)


GRAPH_INFO = {
    "Path": ("Path", "A linear chain of vertices", "Pₙ"),
    "Cycle": ("Cycle", "A closed ring", "Cₙ"),
    "Complete": ("Complete", "Every pair is connected", "Kₙ"),
    "Star": ("Star", "One centre with n leaves", "Sₙ"),
    "Erdos-Renyi": ("Erdős–Rényi", "Random edges, configurable p", "G(n,p)"),
    "3-Regular": ("3-Regular", "Every vertex has degree three", "G₃"),
    "Fork": ("Fork", "A path with a two-prong fork", "Fₙ"),
}

MODE_LABELS = {
    "human_vs_human": "Human vs Human",
    "human_vs_ai": "Human vs AI",
    "ai_vs_ai": "AI vs AI",
}


@dataclass(frozen=True)
class DesktopConfig:
    graph_type: str = "Path"
    n: int = 7
    probability: float = 0.4
    mode: str = "human_vs_ai"
    human_player: int = 1
    depth: int = 5


def build_desktop_graph(config: DesktopConfig) -> Tuple[nx.Graph, str, str]:
    """Validate *config* and return ``(graph, display name, layout kind)``."""
    if config.graph_type not in GRAPH_INFO:
        raise ValueError(f"Unknown graph type: {config.graph_type}.")
    if config.mode not in MODE_LABELS:
        raise ValueError(f"Unknown game mode: {config.mode}.")
    if config.human_player not in (1, 2):
        raise ValueError("Human side must be Player 1 or Player 2.")
    if not 1 <= config.n <= 60:
        raise ValueError("Graph size must be between 1 and 60.")
    if not 0.0 <= config.probability <= 1.0:
        raise ValueError("Edge probability p must be between 0 and 1.")
    if not 1 <= config.depth <= 12:
        raise ValueError("AI search depth must be between 1 and 12.")

    n = config.n
    graph_type = config.graph_type
    if graph_type == "Path":
        if n < 2:
            raise ValueError("A path requires at least 2 vertices.")
        return path_graph(n), f"Path P{n}", "path"
    if graph_type == "Cycle":
        if n < 3:
            raise ValueError("A cycle requires at least 3 vertices.")
        return cycle_graph(n), f"Cycle C{n}", "circular"
    if graph_type == "Complete":
        return complete_graph(n), f"Complete K{n}", "circular"
    if graph_type == "Star":
        return star_graph(n), f"Star S{n} ({n} leaves)", "star"
    if graph_type == "Erdos-Renyi":
        graph = erdos_renyi_graph(n, config.probability, seed=42)
        return graph, f"Erdős–Rényi G({n}, {config.probability:g})", "spring"
    if graph_type == "3-Regular":
        if n < 4 or n % 2:
            raise ValueError("A 3-regular graph requires an even n of at least 4.")
        graph = random_regular_graph(n, 3, seed=42)
        return graph, f"3-Regular graph ({n} vertices)", "spring"
    if n < 3:
        raise ValueError("A fork graph requires n of at least 3.")
    return fork_graph(n), f"Fork F{n}", "fork"


def graph_positions(graph: nx.Graph, layout_kind: str) -> Dict[object, Tuple[float, float]]:
    """Return deterministic, normalised positions for drawing *graph*."""
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}

    if layout_kind == "path":
        count = len(nodes)
        raw = {node: (index / max(1, count - 1), 0.0) for index, node in enumerate(nodes)}
    elif layout_kind == "fork":
        n = max(nodes)
        branch = n - 2
        raw = {}
        for node in range(branch + 1):
            raw[node] = (node / max(1, branch), 0.0)
        raw[n - 1] = (1.18, -0.24)
        raw[n] = (1.18, 0.24)
    elif layout_kind == "star":
        raw_array = nx.shell_layout(graph, nlist=[[0], [node for node in nodes if node != 0]])
        raw = {node: tuple(raw_array[node]) for node in nodes}
    elif layout_kind == "circular":
        raw_array = nx.circular_layout(graph)
        raw = {node: tuple(raw_array[node]) for node in nodes}
    else:
        raw_array = nx.spring_layout(graph, seed=42, iterations=120)
        raw = {node: tuple(raw_array[node]) for node in nodes}

    xs = [float(point[0]) for point in raw.values()]
    ys = [float(point[1]) for point in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    return {
        node: (
            ((float(point[0]) - min_x) / span_x) * 2.0 - 1.0,
            ((float(point[1]) - min_y) / span_y) * 2.0 - 1.0
            if max_y != min_y else 0.0,
        )
        for node, point in raw.items()
    }


class ChoiceButton(QPushButton):
    """Checkable two-line card used by the setup page."""

    def __init__(self, title: str, subtitle: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(f"{title}\n{subtitle}", parent)
        self.setCheckable(True)
        self.setObjectName("choiceCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class GraphCanvas(QWidget):
    """Antialiased graph renderer used for setup previews and live play."""

    node_clicked = Signal(object)

    def __init__(self, interactive: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 330)
        self.setMouseTracking(interactive)
        self.interactive = interactive
        self.graph: Optional[nx.Graph] = None
        self.layout_kind = "spring"
        self.positions: Dict[object, Tuple[float, float]] = {}
        self.game: Optional[StableSetGame] = None
        self.hover_node = None
        self.busy = False
        self.empty_message = "Choose a graph to preview it"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_graph(self, graph: nx.Graph, layout_kind: str) -> None:
        self.graph = graph.copy()
        self.layout_kind = layout_kind
        self.positions = graph_positions(self.graph, layout_kind)
        self.game = None
        self.hover_node = None
        self.update()

    def set_game(self, game: StableSetGame, layout_kind: str) -> None:
        graph_changed = (
            self.graph is None
            or set(self.graph.nodes) != set(game.graph.nodes)
            or {frozenset(edge) for edge in self.graph.edges}
            != {frozenset(edge) for edge in game.graph.edges}
            or self.layout_kind != layout_kind
        )
        self.graph = game.graph
        self.layout_kind = layout_kind
        if graph_changed or not self.positions:
            self.positions = graph_positions(self.graph, layout_kind)
        self.game = game
        self.update()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.update()

    def _node_radius(self) -> float:
        count = self.graph.number_of_nodes() if self.graph is not None else 0
        if count <= 12:
            return 23.0
        if count <= 24:
            return 18.0
        if count <= 40:
            return 13.0
        return 9.5

    def _screen_positions(self) -> Dict[object, QPointF]:
        margin = max(48.0, self._node_radius() + 20.0)
        width = max(1.0, self.width() - 2 * margin)
        height = max(1.0, self.height() - 2 * margin)
        return {
            node: QPointF(
                margin + (x + 1.0) * 0.5 * width,
                margin + (1.0 - (y + 1.0) * 0.5) * height,
            )
            for node, (x, y) in self.positions.items()
        }

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        if self.graph is None:
            painter.setPen(QColor("#64748B"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_message)
            return

        screen = self._screen_positions()
        painter.setPen(QPen(QColor("#CBD5E1"), 1.8))
        for left, right in self.graph.edges:
            painter.drawLine(screen[left], screen[right])

        legal = set(self.game.legal_moves()) if self.game is not None else set(self.graph.nodes)
        selected_by = {}
        if self.game is not None:
            selected_by = {vertex: player for player, vertex in self.game.history}

        radius = self._node_radius()
        for node in self.graph.nodes:
            point = screen[node]
            if node in selected_by:
                fill = QColor("#2563EB" if selected_by[node] == 1 else "#E5484D")
                border = fill.darker(108)
                text = QColor("#FFFFFF")
            elif node in legal:
                fill = QColor("#ECFDF5")
                border = QColor("#10B981")
                text = QColor("#065F46")
            else:
                fill = QColor("#E2E8F0")
                border = QColor("#CBD5E1")
                text = QColor("#64748B")

            if node == self.hover_node and node in legal and self.interactive and not self.busy:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(37, 99, 235, 35))
                painter.drawEllipse(point, radius + 8, radius + 8)
                border = QColor("#2563EB")

            painter.setPen(QPen(border, 2.2))
            painter.setBrush(fill)
            painter.drawEllipse(point, radius, radius)

            if self.graph.number_of_nodes() <= 32:
                font_size = 9 if radius >= 18 else 7
                painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
                painter.setPen(text)
                painter.drawText(
                    int(point.x() - radius),
                    int(point.y() - radius),
                    int(radius * 2),
                    int(radius * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    str(node),
                )

        if self.busy:
            painter.fillRect(self.rect(), QColor(15, 23, 42, 76))
            box_width, box_height = 230, 72
            x = (self.width() - box_width) / 2
            y = (self.height() - box_height) / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawRoundedRect(x, y, box_width, box_height, 12, 12)
            painter.setPen(QColor("#0F172A"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.drawText(
                int(x), int(y), box_width, box_height,
                Qt.AlignmentFlag.AlignCenter,
                "AI is thinking…",
            )
        elif self.game is not None and self.game.is_game_over():
            winner = self.game.get_winner()
            painter.fillRect(self.rect(), QColor(255, 255, 255, 184))
            painter.setPen(QColor("#0F172A"))
            painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            painter.drawText(
                self.rect().adjusted(0, -24, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                f"Player {winner} wins",
            )
            painter.setPen(QColor("#64748B"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(
                self.rect().adjusted(0, 32, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                f"Game complete in {len(self.game.history)} moves",
            )

    def _node_at(self, point: QPointF):
        if self.graph is None:
            return None
        radius = self._node_radius() + 6
        for node, node_point in self._screen_positions().items():
            dx = point.x() - node_point.x()
            dy = point.y() - node_point.y()
            if dx * dx + dy * dy <= radius * radius:
                return node
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self.interactive or self.busy:
            return
        node = self._node_at(event.position())
        legal = set(self.game.legal_moves()) if self.game is not None else set()
        hover = node if node in legal else None
        if hover != self.hover_node:
            self.hover_node = hover
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hover is not None
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802 - Qt API
        self.hover_node = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self.interactive or self.busy or self.game is None:
            return
        node = self._node_at(event.position())
        if node is not None and self.game.is_legal_move(node):
            self.node_clicked.emit(node)


class SetupPage(QWidget):
    start_requested = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.graph_group = QButtonGroup(self)
        self.mode_group = QButtonGroup(self)
        self.side_group = QButtonGroup(self)
        self.graph_buttons: Dict[str, ChoiceButton] = {}
        self.mode_buttons: Dict[str, QPushButton] = {}
        self.side_buttons: Dict[int, QPushButton] = {}
        self._build()
        self._connect()
        self.graph_buttons["Path"].setChecked(True)
        self.side_buttons[1].setChecked(True)
        self.mode_buttons["human_vs_ai"].setChecked(True)
        self._update_preview()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(44, 28, 44, 36)
        root.setSpacing(24)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title = QLabel("Stable Set Game")
        title.setObjectName("appTitle")
        subtitle = QLabel("Interactive research platform for graph-based impartial games")
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        badge = QLabel("MSc Research Project")
        badge.setObjectName("headerBadge")
        header.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(24)
        root.addLayout(body, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("setupScroll")
        setup_card = QFrame()
        setup_card.setObjectName("card")
        setup_card.setMinimumWidth(500)
        setup_layout = QVBoxLayout(setup_card)
        setup_layout.setContentsMargins(28, 26, 28, 28)
        setup_layout.setSpacing(15)

        heading = QLabel("Configure a new game")
        heading.setObjectName("sectionTitle")
        setup_layout.addWidget(heading)
        intro = QLabel("Choose a graph family, players and search settings.")
        intro.setObjectName("mutedText")
        setup_layout.addWidget(intro)

        setup_layout.addWidget(self._label("1  Graph family"))
        graph_grid = QGridLayout()
        graph_grid.setHorizontalSpacing(10)
        graph_grid.setVerticalSpacing(10)
        for index, (key, (title_text, subtitle_text, _symbol)) in enumerate(GRAPH_INFO.items()):
            button = ChoiceButton(title_text, subtitle_text)
            self.graph_group.addButton(button)
            self.graph_buttons[key] = button
            row, column = divmod(index, 3)
            if index == len(GRAPH_INFO) - 1:
                graph_grid.addWidget(button, row, 0, 1, 3)
            else:
                graph_grid.addWidget(button, row, column)
        setup_layout.addLayout(graph_grid)

        setup_layout.addWidget(self._label("2  Game mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        for key, label in MODE_LABELS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setObjectName("segmentButton")
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            mode_row.addWidget(button)
        setup_layout.addLayout(mode_row)

        self.side_container = QWidget()
        side_layout = QVBoxLayout(self.side_container)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        side_layout.addWidget(self._label("Human plays as"))
        side_row = QHBoxLayout()
        side_row.setSpacing(8)
        for player, label in ((1, "Player 1 · moves first"), (2, "Player 2 · moves second")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setObjectName("optionButton")
            self.side_group.addButton(button)
            self.side_buttons[player] = button
            side_row.addWidget(button)
        side_layout.addLayout(side_row)
        setup_layout.addWidget(self.side_container)

        setup_layout.addWidget(self._label("3  Parameters"))
        parameter_row = QHBoxLayout()
        parameter_row.setSpacing(12)
        n_box, self.n_spin = self._spin_card("Graph size", 1, 60, 7)
        depth_box, self.depth_spin = self._spin_card("AI search depth", 1, 12, 5)
        self.probability_box, self.probability_spin = self._probability_card()
        parameter_row.addWidget(n_box)
        parameter_row.addWidget(depth_box)
        parameter_row.addWidget(self.probability_box)
        setup_layout.addLayout(parameter_row)

        self.error_label = QLabel("")
        self.error_label.setObjectName("errorText")
        self.error_label.setWordWrap(True)
        self.error_label.setMinimumHeight(20)
        setup_layout.addWidget(self.error_label)

        self.start_button = QPushButton("Start game")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setMinimumHeight(50)
        setup_layout.addWidget(self.start_button)
        scroll.setWidget(setup_card)
        body.addWidget(scroll, 5)

        preview_card = QFrame()
        preview_card.setObjectName("card")
        preview_card.setMinimumWidth(470)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(26, 24, 26, 26)
        preview_layout.setSpacing(14)
        preview_top = QHBoxLayout()
        preview_heading = QLabel("Live preview")
        preview_heading.setObjectName("sectionTitle")
        self.preview_symbol = QLabel("Pₙ")
        self.preview_symbol.setObjectName("symbolBadge")
        preview_top.addWidget(preview_heading)
        preview_top.addStretch()
        preview_top.addWidget(self.preview_symbol)
        preview_layout.addLayout(preview_top)

        self.preview_canvas = GraphCanvas(interactive=False)
        self.preview_canvas.setObjectName("graphCanvas")
        preview_layout.addWidget(self.preview_canvas, 1)

        self.preview_name = QLabel("Path P7")
        self.preview_name.setObjectName("previewTitle")
        preview_layout.addWidget(self.preview_name)
        self.preview_description = QLabel("")
        self.preview_description.setObjectName("mutedText")
        self.preview_description.setWordWrap(True)
        preview_layout.addWidget(self.preview_description)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.nodes_metric = self._metric("VERTICES", "7")
        self.edges_metric = self._metric("EDGES", "6")
        self.mode_metric = self._metric("MODE", "Human vs AI")
        metrics.addWidget(self.nodes_metric[0])
        metrics.addWidget(self.edges_metric[0])
        metrics.addWidget(self.mode_metric[0])
        preview_layout.addLayout(metrics)

        rule_box = QFrame()
        rule_box.setObjectName("infoBox")
        rule_layout = QVBoxLayout(rule_box)
        rule_layout.setContentsMargins(16, 14, 16, 14)
        rule_title = QLabel("How to play")
        rule_title.setObjectName("smallHeading")
        rule_text = QLabel(
            "Select a legal vertex. It and all adjacent vertices become unavailable. "
            "The player with no legal move loses."
        )
        rule_text.setObjectName("mutedText")
        rule_text.setWordWrap(True)
        rule_layout.addWidget(rule_title)
        rule_layout.addWidget(rule_text)
        preview_layout.addWidget(rule_box)
        body.addWidget(preview_card, 5)

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _spin_card(title: str, minimum: int, maximum: int, value: int):
        frame = QFrame()
        frame.setObjectName("parameterCard")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 12, 10)
        label = QLabel(title)
        label.setObjectName("parameterLabel")
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setObjectName("numberInput")
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(spin)
        return frame, spin

    @staticmethod
    def _probability_card():
        frame = QFrame()
        frame.setObjectName("parameterCard")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 12, 10)
        label = QLabel("Edge probability p")
        label.setObjectName("parameterLabel")
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(0.4)
        spin.setObjectName("numberInput")
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(spin)
        return frame, spin

    @staticmethod
    def _metric(title: str, value: str):
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame, value_label

    def _connect(self) -> None:
        for button in self.graph_buttons.values():
            button.toggled.connect(self._update_preview)
        for button in self.mode_buttons.values():
            button.toggled.connect(self._on_mode_changed)
        for button in self.side_buttons.values():
            button.toggled.connect(self._update_preview)
        self.n_spin.valueChanged.connect(self._update_preview)
        self.depth_spin.valueChanged.connect(self._update_preview)
        self.probability_spin.valueChanged.connect(self._update_preview)
        self.start_button.clicked.connect(self._start)

    def _selected_graph_type(self) -> str:
        return next(key for key, button in self.graph_buttons.items() if button.isChecked())

    def _selected_mode(self) -> str:
        return next(key for key, button in self.mode_buttons.items() if button.isChecked())

    def _selected_side(self) -> int:
        return next(
            (player for player, button in self.side_buttons.items() if button.isChecked()),
            1,
        )

    def current_config(self) -> DesktopConfig:
        return DesktopConfig(
            graph_type=self._selected_graph_type(),
            n=self.n_spin.value(),
            probability=self.probability_spin.value(),
            mode=self._selected_mode(),
            human_player=self._selected_side(),
            depth=self.depth_spin.value(),
        )

    def _on_mode_changed(self) -> None:
        if not any(button.isChecked() for button in self.mode_buttons.values()):
            return
        self.side_container.setVisible(self._selected_mode() == "human_vs_ai")
        self._update_preview()

    def _update_preview(self) -> None:
        if not any(button.isChecked() for button in self.graph_buttons.values()):
            return
        if not any(button.isChecked() for button in self.mode_buttons.values()):
            return
        try:
            config = self.current_config()
            self.probability_box.setVisible(config.graph_type == "Erdos-Renyi")
            graph, name, layout_kind = build_desktop_graph(config)
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return

        self.error_label.setText("")
        self.preview_canvas.set_graph(graph, layout_kind)
        title, description, symbol = GRAPH_INFO[config.graph_type]
        self.preview_symbol.setText(symbol)
        self.preview_name.setText(name)
        detail = description
        if config.graph_type == "Erdos-Renyi":
            detail = f"Random edges, p = {config.probability:g}."
        if config.graph_type in {"Erdos-Renyi", "3-Regular"}:
            detail += " A fixed seed keeps demonstrations reproducible."
        self.preview_description.setText(detail)
        self.nodes_metric[1].setText(str(graph.number_of_nodes()))
        self.edges_metric[1].setText(str(graph.number_of_edges()))
        self.mode_metric[1].setText(MODE_LABELS[config.mode])

    def _start(self) -> None:
        try:
            config = self.current_config()
            build_desktop_graph(config)
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        self.error_label.setText("")
        self.start_requested.emit(config)


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class AIWorker(QRunnable):
    def __init__(self, ai: AIPlayer, game: StableSetGame) -> None:
        super().__init__()
        self.ai = ai
        self.game = game.copy()
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.ai.choose_move(self.game)
        except Exception as exc:  # surface worker failures in the GUI
            self.signals.error.emit(str(exc))
            return
        self.signals.result.emit(result)


class GamePage(QWidget):
    back_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config: Optional[DesktopConfig] = None
        self.game: Optional[StableSetGame] = None
        self.graph_name = ""
        self.layout_kind = "spring"
        self.ai1: Optional[AIPlayer] = None
        self.ai2: Optional[AIPlayer] = None
        self.busy = False
        self.thread_pool = QThreadPool.globalInstance()
        self._worker: Optional[AIWorker] = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 22, 32, 28)
        root.setSpacing(18)

        header = QHBoxLayout()
        self.back_button = QPushButton("  New setup")
        self.back_button.setObjectName("secondaryButton")
        self.back_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self.back_button)
        header.addSpacing(14)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.page_title = QLabel("Stable Set Game")
        self.page_title.setObjectName("gameTitle")
        self.page_meta = QLabel("")
        self.page_meta.setObjectName("mutedText")
        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_meta)
        header.addLayout(title_box)
        header.addStretch()
        self.busy_pill = QLabel("Ready")
        self.busy_pill.setObjectName("readyPill")
        header.addWidget(self.busy_pill)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(20)
        root.addLayout(content, 1)

        board_card = QFrame()
        board_card.setObjectName("card")
        board_layout = QVBoxLayout(board_card)
        board_layout.setContentsMargins(22, 18, 22, 20)
        board_layout.setSpacing(14)
        board_header = QHBoxLayout()
        board_title = QLabel("Game board")
        board_title.setObjectName("sectionTitle")
        board_header.addWidget(board_title)
        board_header.addStretch()
        legend = QLabel(
            "●  Legal     <span style='color:#2563EB'>●</span>  Player 1     "
            "<span style='color:#E5484D'>●</span>  Player 2     "
            "<span style='color:#94A3B8'>●</span>  Blocked"
        )
        legend.setObjectName("legendText")
        board_header.addWidget(legend)
        board_layout.addLayout(board_header)
        self.canvas = GraphCanvas(interactive=True)
        self.canvas.setObjectName("graphCanvas")
        board_layout.addWidget(self.canvas, 1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.undo_button = self._action_button(
            "Undo", QStyle.StandardPixmap.SP_ArrowBack
        )
        self.restart_button = self._action_button(
            "Restart", QStyle.StandardPixmap.SP_BrowserReload
        )
        self.next_button = self._action_button(
            "Next AI move", QStyle.StandardPixmap.SP_MediaPlay, primary=True
        )
        actions.addWidget(self.undo_button)
        actions.addWidget(self.restart_button)
        actions.addStretch()
        actions.addWidget(self.next_button)
        board_layout.addLayout(actions)
        content.addWidget(board_card, 7)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(14)
        sidebar.setContentsMargins(0, 0, 0, 0)

        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 18, 20, 20)
        status_layout.setSpacing(10)
        status_layout.addWidget(self._eyebrow("CURRENT GAME"))
        self.status_title = QLabel("Player 1 to move")
        self.status_title.setObjectName("statusTitle")
        status_layout.addWidget(self.status_title)
        self.status_detail = QLabel("Select a legal vertex to begin.")
        self.status_detail.setObjectName("mutedText")
        self.status_detail.setWordWrap(True)
        status_layout.addWidget(self.status_detail)

        meta_grid = QGridLayout()
        meta_grid.setHorizontalSpacing(8)
        meta_grid.setVerticalSpacing(8)
        self.mode_value = self._meta_value("Mode")
        self.turn_value = self._meta_value("Turn")
        self.legal_value = self._meta_value("Legal moves")
        self.depth_value = self._meta_value("AI depth")
        for index, widget in enumerate(
            (self.mode_value[0], self.turn_value[0], self.legal_value[0], self.depth_value[0])
        ):
            meta_grid.addWidget(widget, index // 2, index % 2)
        status_layout.addLayout(meta_grid)
        sidebar.addWidget(status_card)

        history_card = QFrame()
        history_card.setObjectName("card")
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(20, 18, 20, 16)
        history_layout.setSpacing(10)
        history_top = QHBoxLayout()
        history_top.addWidget(self._eyebrow("MOVE HISTORY"))
        history_top.addStretch()
        self.move_count = QLabel("0 moves")
        self.move_count.setObjectName("mutedText")
        history_top.addWidget(self.move_count)
        history_layout.addLayout(history_top)
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.setAlternatingRowColors(False)
        history_layout.addWidget(self.history_list, 1)
        sidebar.addWidget(history_card, 1)

        ai_card = QFrame()
        ai_card.setObjectName("infoBox")
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(16, 14, 16, 14)
        ai_layout.addWidget(self._eyebrow("LAST AI SEARCH"))
        self.ai_stats = QLabel("No search has been run yet.")
        self.ai_stats.setObjectName("mutedText")
        self.ai_stats.setWordWrap(True)
        ai_layout.addWidget(self.ai_stats)
        sidebar.addWidget(ai_card)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMinimumWidth(330)
        sidebar_widget.setMaximumWidth(380)
        content.addWidget(sidebar_widget, 3)

        self.back_button.clicked.connect(self._back)
        self.undo_button.clicked.connect(self.undo)
        self.restart_button.clicked.connect(self.restart)
        self.next_button.clicked.connect(self.next_ai_move)
        self.canvas.node_clicked.connect(self.play_human_move)

    def _action_button(self, text: str, icon, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon))
        button.setObjectName("primaryAction" if primary else "actionButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(42)
        return button

    @staticmethod
    def _eyebrow(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("eyebrow")
        return label

    @staticmethod
    def _meta_value(title: str):
        frame = QFrame()
        frame.setObjectName("miniMetric")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("miniMetricTitle")
        value_label = QLabel("—")
        value_label.setObjectName("miniMetricValue")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame, value_label

    def start_game(self, config: DesktopConfig) -> None:
        self.config = config
        graph, self.graph_name, self.layout_kind = build_desktop_graph(config)
        self.game = StableSetGame(graph)
        self.ai1 = AIPlayer(max_depth=config.depth, name="AI Player 1")
        self.ai2 = AIPlayer(max_depth=config.depth, name="AI Player 2")
        self.ai_stats.setText("No search has been run yet.")
        self.page_title.setText(self.graph_name)
        self.page_meta.setText(
            f"{MODE_LABELS[config.mode]}  ·  {graph.number_of_nodes()} vertices  ·  "
            f"{graph.number_of_edges()} edges"
        )
        self.next_button.setVisible(config.mode == "ai_vs_ai")
        self._set_busy(False)
        self.refresh()
        QTimer.singleShot(120, self._maybe_start_ai)

    def _is_human(self, player: int) -> bool:
        if self.config is None:
            return False
        if self.config.mode == "human_vs_human":
            return True
        if self.config.mode == "human_vs_ai":
            return player == self.config.human_player
        return False

    def _ai_for(self, player: int) -> Optional[AIPlayer]:
        if self.config is None or self.config.mode == "human_vs_human":
            return None
        if self.config.mode == "human_vs_ai" and self._is_human(player):
            return None
        return self.ai1 if player == 1 else self.ai2

    def play_human_move(self, vertex) -> None:
        if self.busy or self.game is None or self.game.is_game_over():
            return
        if not self._is_human(self.game.current_player):
            return
        if not self.game.is_legal_move(vertex):
            return
        self.game.make_move(vertex)
        self.refresh()
        QTimer.singleShot(100, self._maybe_start_ai)

    def _maybe_start_ai(self) -> None:
        if self.busy or self.game is None or self.game.is_game_over():
            return
        if self.config is not None and self.config.mode == "ai_vs_ai":
            return
        ai = self._ai_for(self.game.current_player)
        if ai is not None:
            self._start_ai(ai)

    def next_ai_move(self) -> None:
        if (
            self.busy or self.game is None or self.game.is_game_over()
            or self.config is None or self.config.mode != "ai_vs_ai"
        ):
            return
        ai = self._ai_for(self.game.current_player)
        if ai is not None:
            self._start_ai(ai)

    def _start_ai(self, ai: AIPlayer) -> None:
        if self.game is None:
            return
        self._set_busy(True)
        self.refresh()
        worker = AIWorker(ai, self.game)
        worker.signals.result.connect(self._on_ai_result)
        worker.signals.error.connect(self._on_ai_error)
        self._worker = worker
        self.thread_pool.start(worker)

    def _on_ai_result(self, result: SearchResult) -> None:
        if self.game is not None and result.best_move is not None and not self.game.is_game_over():
            self.game.make_move(result.best_move)
        self.ai_stats.setText(
            f"Move {result.best_move}  ·  score {result.score:g}\n"
            f"{result.nodes_explored:,} nodes explored  ·  depth {result.depth_reached}"
        )
        self._worker = None
        self._set_busy(False)
        self.refresh()

    def _on_ai_error(self, message: str) -> None:
        self._worker = None
        self._set_busy(False)
        QMessageBox.critical(self, "AI search failed", message)
        self.refresh()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.canvas.set_busy(busy)
        self.busy_pill.setText("AI thinking…" if busy else "Ready")
        self.busy_pill.setObjectName("busyPill" if busy else "readyPill")
        self.busy_pill.style().unpolish(self.busy_pill)
        self.busy_pill.style().polish(self.busy_pill)
        self.undo_button.setEnabled(not busy)
        self.restart_button.setEnabled(not busy)
        self.next_button.setEnabled(not busy)
        self.back_button.setEnabled(not busy)

    def refresh(self) -> None:
        if self.game is None or self.config is None:
            return
        self.canvas.set_game(self.game, self.layout_kind)
        legal_count = len(self.game.legal_moves())
        if self.game.is_game_over():
            winner = self.game.get_winner()
            self.status_title.setText(f"Player {winner} wins")
            self.status_detail.setText(
                f"Player {self.game.current_player} has no legal move. "
                "Restart to play this graph again."
            )
        elif self.busy:
            self.status_title.setText(f"Player {self.game.current_player} is thinking")
            self.status_detail.setText("The AI is evaluating the remaining game tree.")
        else:
            player = self.game.current_player
            self.status_title.setText(f"Player {player} to move")
            if self._is_human(player):
                self.status_detail.setText("Select any green vertex on the board.")
            elif self.config.mode == "ai_vs_ai":
                self.status_detail.setText("Press Next AI move to advance the match.")
            else:
                self.status_detail.setText("The AI will move automatically.")

        self.mode_value[1].setText(MODE_LABELS[self.config.mode])
        self.turn_value[1].setText(f"Player {self.game.current_player}")
        self.legal_value[1].setText(str(legal_count))
        self.depth_value[1].setText(str(self.config.depth))
        can_undo = bool(self.game.history) and not self.busy
        if (
            self.config.mode == "human_vs_ai"
            and self.config.human_player == 2
            and len(self.game.history) < 2
        ):
            can_undo = False
        self.undo_button.setEnabled(can_undo)
        self.next_button.setEnabled(
            self.config.mode == "ai_vs_ai"
            and not self.busy
            and not self.game.is_game_over()
        )
        self._refresh_history()

    def _refresh_history(self) -> None:
        self.history_list.clear()
        if self.game is None:
            return
        for index, (player, vertex) in enumerate(self.game.history, start=1):
            item = QListWidgetItem(f"{index:02d}     Player {player}     Vertex {vertex}")
            item.setForeground(QColor("#1D4ED8" if player == 1 else "#C24146"))
            item.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            item.setSizeHint(QSize(item.sizeHint().width(), 34))
            self.history_list.addItem(item)
        count = len(self.game.history)
        self.move_count.setText(f"{count} move" + ("" if count == 1 else "s"))
        if count:
            self.history_list.scrollToBottom()

    def undo(self) -> None:
        if self.busy or self.game is None or not self.game.history:
            return
        if self.config is not None and self.config.mode == "human_vs_ai" and len(self.game.history) >= 2:
            self.game.undo_move()
            self.game.undo_move()
        else:
            self.game.undo_move()
        self.ai_stats.setText("The last move was undone.")
        self.refresh()

    def restart(self) -> None:
        if self.busy or self.config is None:
            return
        self.start_game(self.config)

    def _back(self) -> None:
        if self.busy:
            return
        self.back_requested.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stable Set Game")
        self.setMinimumSize(1180, 760)
        self.resize(1360, 860)
        self.stack = QStackedWidget()
        self.setup_page = SetupPage()
        self.game_page = GamePage()
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.game_page)
        self.setCentralWidget(self.stack)
        self.setup_page.start_requested.connect(self.start_game)
        self.game_page.back_requested.connect(self.show_setup)

    def start_game(self, config: DesktopConfig) -> None:
        self.game_page.start_game(config)
        self.stack.setCurrentWidget(self.game_page)

    def show_setup(self) -> None:
        self.stack.setCurrentWidget(self.setup_page)


APP_STYLESHEET = """
* {
    font-family: "Segoe UI";
    color: #0F172A;
}
QMainWindow, QWidget {
    background: #F3F6FA;
}
QLabel {
    background: transparent;
}
QLabel#appTitle {
    font-size: 28px;
    font-weight: 700;
    color: #0F172A;
}
QLabel#gameTitle {
    font-size: 22px;
    font-weight: 700;
}
QLabel#pageSubtitle, QLabel#mutedText {
    color: #64748B;
    font-size: 13px;
}
QLabel#headerBadge, QLabel#symbolBadge {
    color: #1D4ED8;
    background: #E8F0FF;
    border: 1px solid #C7D8FF;
    border-radius: 13px;
    padding: 6px 12px;
    font-weight: 600;
}
QFrame#card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}
QLabel#sectionTitle {
    font-size: 17px;
    font-weight: 700;
}
QLabel#fieldLabel {
    color: #334155;
    font-size: 13px;
    font-weight: 700;
    padding-top: 3px;
}
QPushButton#choiceCard {
    background: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 11px;
    padding: 10px 14px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    color: #334155;
}
QPushButton#choiceCard:hover {
    border: 1px solid #93B4F4;
    background: #F8FAFF;
}
QPushButton#choiceCard:checked {
    border: 2px solid #2563EB;
    background: #EEF4FF;
    color: #1D4ED8;
}
QPushButton#segmentButton {
    min-height: 38px;
    background: #F1F5F9;
    border: 1px solid #DCE3EC;
    padding: 0 12px;
    font-weight: 600;
    color: #475569;
    border-radius: 8px;
}
QPushButton#segmentButton:hover { background: #E8EEF6; }
QPushButton#segmentButton:checked {
    background: #0F172A;
    color: #FFFFFF;
    border-color: #0F172A;
}
QPushButton#optionButton {
    min-height: 38px;
    background: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 9px;
    padding: 0 12px;
    font-weight: 600;
    color: #475569;
}
QPushButton#optionButton:checked {
    background: #EEF4FF;
    border: 2px solid #2563EB;
    color: #1D4ED8;
}
QFrame#parameterCard, QFrame#metricCard, QFrame#miniMetric {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}
QLabel#parameterLabel {
    color: #475569;
    font-weight: 600;
}
QSpinBox#numberInput, QDoubleSpinBox#numberInput {
    min-width: 64px;
    min-height: 34px;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    background: #FFFFFF;
    font-size: 14px;
    font-weight: 700;
    padding: 0 5px;
}
QPushButton#primaryButton, QPushButton#primaryAction {
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    padding: 0 22px;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#primaryButton:hover, QPushButton#primaryAction:hover { background: #1D4ED8; }
QPushButton#primaryButton:pressed, QPushButton#primaryAction:pressed { background: #1E40AF; }
QPushButton#secondaryButton, QPushButton#actionButton {
    background: #FFFFFF;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
    min-height: 40px;
    padding: 0 16px;
    font-weight: 600;
}
QPushButton#secondaryButton:hover, QPushButton#actionButton:hover { background: #F8FAFC; border-color: #94A3B8; }
QPushButton:disabled { color: #94A3B8; background: #E2E8F0; }
QLabel#errorText { color: #B42318; font-size: 12px; }
QLabel#previewTitle { font-size: 18px; font-weight: 700; }
QLabel#metricTitle, QLabel#miniMetricTitle, QLabel#eyebrow {
    color: #64748B;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#metricValue { font-size: 14px; font-weight: 700; }
QLabel#miniMetricValue { font-size: 12px; font-weight: 700; color: #334155; }
QFrame#infoBox {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 11px;
}
QLabel#smallHeading { font-weight: 700; color: #334155; }
QWidget#graphCanvas {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}
QLabel#legendText { color: #64748B; font-size: 11px; }
QLabel#statusTitle { font-size: 20px; font-weight: 700; }
QLabel#readyPill {
    color: #047857;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 13px;
    padding: 6px 12px;
    font-weight: 700;
}
QLabel#busyPill {
    color: #1D4ED8;
    background: #EEF4FF;
    border: 1px solid #BFDBFE;
    border-radius: 13px;
    padding: 6px 12px;
    font-weight: 700;
}
QListWidget#historyList {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget#historyList::item {
    border-bottom: 1px solid #EEF2F7;
    padding: 5px 2px;
}
QListWidget#historyList::item:selected { background: #EEF4FF; }
QScrollArea#setupScroll { background: transparent; }
QScrollArea#setupScroll > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def main() -> None:
    owns_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Stable Set Game")
    app.setOrganizationName("MSc Research Project")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
