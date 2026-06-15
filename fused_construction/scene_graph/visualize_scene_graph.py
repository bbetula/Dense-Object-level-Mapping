#!/usr/bin/env python3
"""Visualize generated scene graph.

Default output:
  - scene_graph_view.html: dependency-free interactive HTML

Optional debug/export outputs:
  - scene_graph_topdown.svg
  - scene_graph_relation.svg
  - scene_graph_centers_edges.ply
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene_graph_config import SCENE_GRAPH_OUT_DIR, VISUALIZATION_TITLE  # noqa: E402


RELATION_COLORS = {
    "on": "#1b7f3f",
    "adjacent_to": "#2367c9",
    "near": "#7c5cc4",
    "inside": "#c05a00",
    "above": "#b21d38",
    "below": "#008f8f",
}

THREE_VERSION = "0.160.0"
THREE_DEPS = {
    "three.module.js": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
    "OrbitControls.js": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js",
}


def ensure_three_dependencies(out_dir: Path) -> bool:
    """Cache Three.js dependencies next to the HTML so local viewing is stable."""
    import urllib.request

    ok = True
    for name, url in THREE_DEPS.items():
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                dst.write_bytes(resp.read())
        except Exception as exc:  # pragma: no cover - network fallback path
            print(f"[WARN] Could not cache {name}: {exc}. HTML will use CDN fallback.")
            ok = False
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualizations for scene_graph.json.")
    parser.add_argument("--scene-graph", default=SCENE_GRAPH_OUT_DIR / "scene_graph.json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=SCENE_GRAPH_OUT_DIR / "visualization")
    parser.add_argument("--title", default=VISUALIZATION_TITLE)
    parser.add_argument("--max-label-len", default=28, type=int)
    parser.add_argument("--write-svg", action="store_true", help="Also write standalone top-down and relation SVG files.")
    parser.add_argument("--write-ply", action="store_true", help="Also write node-center edge PLY for 3D viewers.")
    return parser.parse_args()


def load_graph(path: Path) -> tuple[dict, list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload.get("nodes", []), payload.get("edges", [])


def rgb_to_hex(rgb: list[int] | tuple[int, int, int]) -> str:
    vals = [max(0, min(255, int(v))) for v in rgb[:3]]
    return "#%02x%02x%02x" % tuple(vals)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def bbox(nodes: list[dict]) -> tuple[float, float, float, float]:
    xs = [float(node["centroid"][0]) for node in nodes]
    ys = [float(node["centroid"][1]) for node in nodes]
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-6:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-6:
        min_y -= 1.0
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def project_xy(x: float, y: float, bounds: tuple[float, float, float, float], width: int, height: int, pad: int) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    px = pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)
    # SVG y axis points downward; flip map y for a natural top-down view.
    py = height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)
    return px, py


def node_radius(node: dict) -> float:
    count = max(1, int(node.get("point_count", 1)))
    return max(7.0, min(24.0, 5.0 + math.log10(count + 1) * 5.0))


def truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def write_topdown_svg(path: Path, title: str, nodes: list[dict], edges: list[dict], max_label_len: int) -> None:
    width, height, pad = 1400, 900, 72
    bounds = bbox(nodes)
    nodes_by_id = {node["id"]: node for node in nodes}
    positions = {
        node["id"]: project_xy(float(node["centroid"][0]), float(node["centroid"][1]), bounds, width, height, pad)
        for node in nodes
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC',sans-serif}.title{font-size:30px;font-weight:700;fill:#1f2937}.label{font-size:13px;fill:#111827}.meta{font-size:14px;fill:#6b7280}.edge{fill:none;stroke-width:2.2;opacity:.72}.node{stroke:#111827;stroke-width:1.4}.bbox{fill:none;stroke:#9ca3af;stroke-dasharray:5 5;opacity:.45}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text class="title" x="36" y="46">{esc(title)} - Top-down Spatial View</text>',
        f'<text class="meta" x="36" y="76">nodes={len(nodes)}, edges={len(edges)}. Node position = centroid(x,y), size = log(point_count).</text>',
    ]

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        relation = str(edge.get("relation", "relation"))
        color = RELATION_COLORS.get(relation, "#64748b")
        parts.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}"><title>{esc(src)} {esc(relation)} {esc(tgt)} conf={esc(edge.get("confidence",""))}</title></line>')

    for node in nodes:
        x, y = positions[node["id"]]
        r = node_radius(node)
        color = rgb_to_hex(node.get("color_rgb", [220, 220, 220]))
        label = truncate(f'{node.get("label")}:{node.get("id")}', max_label_len)
        parts.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"><title>{esc(json.dumps(node, ensure_ascii=False))}</title></circle>')
        parts.append(f'<text class="label" x="{x + r + 4:.1f}" y="{y + 4:.1f}">{esc(label)}</text>')

    # Legend
    lx, ly = 36, height - 154
    parts.append(f'<rect x="{lx}" y="{ly}" width="360" height="118" rx="10" fill="#ffffff" stroke="#d1d5db"/>')
    parts.append(f'<text class="meta" x="{lx + 16}" y="{ly + 28}">Relation legend</text>')
    for idx, (rel, color) in enumerate(RELATION_COLORS.items()):
        x = lx + 18 + (idx % 3) * 112
        y = ly + 58 + (idx // 3) * 34
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text class="meta" x="{x + 36}" y="{y + 5}">{esc(rel)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def relation_layout(nodes: list[dict], width: int, height: int) -> dict[str, tuple[float, float]]:
    labels = sorted(set(str(node.get("label", "")) for node in nodes))
    label_to_angle = {label: 2 * math.pi * idx / max(1, len(labels)) for idx, label in enumerate(labels)}
    label_counts: dict[str, int] = {}
    radius = min(width, height) * 0.36
    cx, cy = width * 0.5, height * 0.52
    positions = {}
    for node in nodes:
        label = str(node.get("label", ""))
        label_counts[label] = label_counts.get(label, 0) + 1
        local_idx = label_counts[label] - 1
        angle = label_to_angle[label]
        jitter = (local_idx - 1.5) * 22
        x = cx + math.cos(angle) * (radius + jitter)
        y = cy + math.sin(angle) * (radius + jitter)
        positions[node["id"]] = (x, y)
    return positions


def write_relation_svg(path: Path, title: str, nodes: list[dict], edges: list[dict], max_label_len: int) -> None:
    width, height = 1400, 900
    positions = relation_layout(nodes, width, height)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC',sans-serif}.title{font-size:30px;font-weight:700;fill:#1f2937}.label{font-size:13px;fill:#111827}.edge_label{font-size:12px;fill:#374151}.meta{font-size:14px;fill:#6b7280}.edge{fill:none;stroke-width:2;opacity:.68}.node{stroke:#111827;stroke-width:1.3}",
        "</style>",
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text class="title" x="36" y="46">{esc(title)} - Relation Graph</text>',
        f'<text class="meta" x="36" y="76">Grouped roughly by semantic label. Edge labels show relation type.</text>',
    ]
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        rel = str(edge.get("relation", "relation"))
        color = RELATION_COLORS.get(rel, "#64748b")
        parts.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" marker-end="url(#arrow)"/>')
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        parts.append(f'<text class="edge_label" x="{mx:.1f}" y="{my:.1f}">{esc(rel)}</text>')
    for node in nodes:
        x, y = positions[node["id"]]
        r = node_radius(node)
        color = rgb_to_hex(node.get("color_rgb", [220, 220, 220]))
        label = truncate(f'{node.get("label")}:{node.get("id")}', max_label_len)
        parts.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"><title>{esc(json.dumps(node, ensure_ascii=False))}</title></circle>')
        parts.append(f'<text class="label" text-anchor="middle" x="{x:.1f}" y="{y + r + 16:.1f}">{esc(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_text_topdown(title: str, nodes: list[dict], edges: list[dict], max_label_len: int) -> str:
    width, height, pad = 1400, 900, 72
    bounds = bbox(nodes)
    positions = {
        node["id"]: project_xy(float(node["centroid"][0]), float(node["centroid"][1]), bounds, width, height, pad)
        for node in nodes
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC',sans-serif}.title{font-size:30px;font-weight:700;fill:#1f2937}.label{font-size:13px;fill:#111827}.meta{font-size:14px;fill:#6b7280}.edge{fill:none;stroke-width:2.2;opacity:.72}.node{stroke:#111827;stroke-width:1.4}.bbox{fill:none;stroke:#9ca3af;stroke-dasharray:5 5;opacity:.45}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text class="title" x="36" y="46">{esc(title)} - Top-down Spatial View</text>',
        f'<text class="meta" x="36" y="76">nodes={len(nodes)}, edges={len(edges)}. Node position = centroid(x,y), size = log(point_count).</text>',
    ]

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        relation = str(edge.get("relation", "relation"))
        color = RELATION_COLORS.get(relation, "#64748b")
        parts.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}"><title>{esc(src)} {esc(relation)} {esc(tgt)} conf={esc(edge.get("confidence",""))}</title></line>')

    for node in nodes:
        x, y = positions[node["id"]]
        r = node_radius(node)
        color = rgb_to_hex(node.get("color_rgb", [220, 220, 220]))
        label = truncate(f'{node.get("label")}:{node.get("id")}', max_label_len)
        parts.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"><title>{esc(json.dumps(node, ensure_ascii=False))}</title></circle>')
        parts.append(f'<text class="label" x="{x + r + 4:.1f}" y="{y + 4:.1f}">{esc(label)}</text>')

    lx, ly = 36, height - 154
    parts.append(f'<rect x="{lx}" y="{ly}" width="360" height="118" rx="10" fill="#ffffff" stroke="#d1d5db"/>')
    parts.append(f'<text class="meta" x="{lx + 16}" y="{ly + 28}">Relation legend</text>')
    for idx, (rel, color) in enumerate(RELATION_COLORS.items()):
        x = lx + 18 + (idx % 3) * 112
        y = ly + 58 + (idx // 3) * 34
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text class="meta" x="{x + 36}" y="{y + 5}">{esc(rel)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_text_relation(title: str, nodes: list[dict], edges: list[dict], max_label_len: int) -> str:
    width, height = 1400, 900
    positions = relation_layout(nodes, width, height)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC',sans-serif}.title{font-size:30px;font-weight:700;fill:#1f2937}.label{font-size:13px;fill:#111827}.edge_label{font-size:12px;fill:#374151}.meta{font-size:14px;fill:#6b7280}.edge{fill:none;stroke-width:2;opacity:.68}.node{stroke:#111827;stroke-width:1.3}",
        "</style>",
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text class="title" x="36" y="46">{esc(title)} - Relation Graph</text>',
        f'<text class="meta" x="36" y="76">Grouped roughly by semantic label. Edge labels show relation type.</text>',
    ]
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        rel = str(edge.get("relation", "relation"))
        color = RELATION_COLORS.get(rel, "#64748b")
        parts.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" marker-end="url(#arrow)"/>')
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        parts.append(f'<text class="edge_label" x="{mx:.1f}" y="{my:.1f}">{esc(rel)}</text>')
    for node in nodes:
        x, y = positions[node["id"]]
        r = node_radius(node)
        color = rgb_to_hex(node.get("color_rgb", [220, 220, 220]))
        label = truncate(f'{node.get("label")}:{node.get("id")}', max_label_len)
        parts.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"><title>{esc(json.dumps(node, ensure_ascii=False))}</title></circle>')
        parts.append(f'<text class="label" text-anchor="middle" x="{x:.1f}" y="{y + r + 16:.1f}">{esc(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_html(path: Path, title: str, payload: dict) -> None:
    graph_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    relation_colors_json = json.dumps(RELATION_COLORS, ensure_ascii=False)
    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light; --bg:#f5f7fa; --panel:#ffffff; --line:#d8dee8; --text:#172033; --muted:#667085; --accent:#2563eb; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: Arial, "Noto Sans CJK SC", sans-serif; color:var(--text); background:var(--bg); }
    header { height:58px; display:flex; align-items:center; gap:18px; padding:0 18px; background:#111827; color:#f9fafb; }
    h1 { margin:0; font-size:19px; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    header .meta { color:#cbd5e1; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    main { padding:12px; }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:10px; }
    .toolbar input, .toolbar select, .toolbar button { height:32px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); padding:0 9px; font-size:13px; }
    .toolbar button { cursor:pointer; }
    .toolbar button.active { border-color:var(--accent); color:var(--accent); }
    .switch { display:inline-flex; align-items:center; gap:6px; height:32px; padding:0 9px; border:1px solid var(--line); border-radius:6px; background:#fff; font-size:13px; color:var(--muted); }
    .layout { display:grid; grid-template-columns:minmax(0, 1fr) 340px; gap:12px; align-items:start; }
    .viewer { position:relative; border:1px solid var(--line); border-radius:8px; background:#0b1020; overflow:hidden; min-height:780px; }
    #scene3d { width:100%; height:780px; display:block; }
    .overlay { position:absolute; left:12px; bottom:12px; display:flex; gap:8px; flex-wrap:wrap; pointer-events:none; }
    .legend { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:8px; padding:8px 10px; box-shadow:0 4px 16px rgba(15,23,42,.18); font-size:12px; color:var(--muted); }
    #hoverTip { position:absolute; display:none; pointer-events:none; background:rgba(17,24,39,.92); color:#fff; border-radius:6px; padding:6px 8px; font-size:12px; max-width:360px; z-index:3; }
    .side { display:flex; flex-direction:column; gap:12px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .panel h2 { margin:0 0 10px; font-size:15px; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; max-height:190px; overflow:auto; }
    .chip { border:1px solid var(--line); border-radius:999px; padding:5px 9px; background:#fff; font-size:12px; cursor:pointer; display:inline-flex; align-items:center; gap:6px; }
    .chip.off { opacity:.35; }
    .dot { width:10px; height:10px; border-radius:999px; border:1px solid #111827; display:inline-block; }
    .detail { min-height:132px; font-size:13px; color:var(--muted); line-height:1.55; }
    .detail strong { color:var(--text); }
    .tables { margin-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    table { border-collapse:collapse; width:100%; font-size:12px; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    th, td { border-bottom:1px solid #eef1f6; padding:7px 8px; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    th { background:#f1f5f9; color:#475467; position:sticky; top:0; }
    tbody { display:block; max-height:330px; overflow:auto; }
    thead, tbody tr { display:table; width:100%; table-layout:fixed; }
    tr { cursor:pointer; }
    tr.selected { background:#eff6ff; }
    .muted { color:var(--muted); }
    @media (max-width: 1150px) {
      .layout { grid-template-columns:1fr; }
      .side { display:grid; grid-template-columns:1fr 1fr; }
      .tables { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="meta" id="headerMeta"></div>
  </header>
  <main>
    <div class="toolbar">
      <input id="nodeSearch" type="search" placeholder="查询节点/类别" />
      <input id="edgeSearch" type="search" placeholder="查询边：源/关系/目标" />
      <select id="relationFilter"><option value="">全部关系</option></select>
      <select id="labelMode">
        <option value="selected">选中标签</option>
        <option value="major">主要标签</option>
        <option value="all">全部标签</option>
        <option value="none">隐藏标签</option>
      </select>
      <label class="switch"><input id="edgeToggle" type="checkbox" checked /> 显示边</label>
      <button id="fitBtn">适配3D</button>
      <button id="clearBtn">清空选择</button>
      <span class="muted" id="countText"></span>
    </div>

    <div class="layout">
      <div class="viewer">
        <div id="scene3d"></div>
        <div id="hoverTip"></div>
        <div class="overlay"><div class="legend" id="legendText"></div></div>
      </div>
      <aside class="side">
        <section class="panel">
          <h2>类别过滤</h2>
          <div class="chips" id="chips"></div>
        </section>
        <section class="panel">
          <h2>选择详情</h2>
          <div class="detail" id="detail"></div>
        </section>
      </aside>
    </div>

    <div class="tables">
      <section>
        <table>
          <thead><tr><th>ID</th><th>类别</th><th>类型</th><th>点数</th></tr></thead>
          <tbody id="nodeTable"></tbody>
        </table>
      </section>
      <section>
        <table>
          <thead><tr><th>源节点</th><th>关系</th><th>目标节点</th><th>置信度</th></tr></thead>
          <tbody id="edgeTable"></tbody>
        </table>
      </section>
    </div>
  </main>
  <script type="importmap">
    {"imports":{"three":"./three.module.js","three/addons/controls/OrbitControls.js":"./OrbitControls.js"}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const graph = __GRAPH_JSON__;
    const relationColors = __RELATION_COLORS__;
    const nodes = graph.nodes || [];
    const edges = graph.edges || [];
    const byId = new Map(nodes.map(n => [n.id, n]));
    const relations = [...new Set(edges.map(e => e.relation))].sort();
    const countsByLabel = {};
    for (const n of nodes) countsByLabel[n.label] = (countsByLabel[n.label] || 0) + 1;
    const labels = Object.keys(countsByLabel).sort((a, b) => countsByLabel[b] - countsByLabel[a] || a.localeCompare(b));
    const hiddenLabels = new Set();
    const state = { nodeSearch:'', edgeSearch:'', relation:'', labelMode:'selected', showEdges:true, selectedNode:null, selectedEdge:null, hover:null };
    const nodeMeshes = new Map();
    const edgeObjects = new Map();
    const labelSprites = new Map();

    const container = document.getElementById('scene3d');
    const tip = document.getElementById('hoverTip');
    document.getElementById('headerMeta').textContent = `scene=${graph.scene_name || ''} | nodes=${nodes.length} | edges=${edges.length} | source=${graph.source_pcd || ''}`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false, preserveDrawingBuffer:true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.72));
    const dir = new THREE.DirectionalLight(0xffffff, 0.62);
    dir.position.set(20, 30, 40);
    scene.add(dir);

    const root = new THREE.Group();
    scene.add(root);
    const grid = new THREE.GridHelper(120, 24, 0xcbd5e1, 0xe5e7eb);
    grid.position.y = 0;
    root.add(grid);
    const axes = new THREE.AxesHelper(8);
    root.add(axes);

    function rgbHex(rgb) {
      const c = rgb || [220, 220, 220];
      return ((c[0] || 0) << 16) + ((c[1] || 0) << 8) + (c[2] || 0);
    }
    function cssColor(rgb) {
      const c = rgb || [220, 220, 220];
      return `rgb(${c[0] || 0},${c[1] || 0},${c[2] || 0})`;
    }
    function pos(node) {
      const c = node.centroid || [0, 0, 0];
      return new THREE.Vector3(Number(c[0]) || 0, Number(c[2]) || 0, -(Number(c[1]) || 0));
    }
    function radius(node) {
      return Math.max(0.14, Math.min(1.35, 0.11 + Math.log10((node.point_count || 1) + 1) * 0.24));
    }
    function nodeLabel(n) { return `${n.label}:${n.id}`; }
    function edgeText(e) { return `${e.source} ${e.relation} ${e.target}`; }
    function matchesNode(n) {
      if (hiddenLabels.has(n.label)) return false;
      if (!state.nodeSearch) return true;
      const q = state.nodeSearch.toLowerCase();
      return String(n.id).toLowerCase().includes(q) || String(n.label).toLowerCase().includes(q);
    }
    function matchesEdge(e) {
      if (state.relation && e.relation !== state.relation) return false;
      if (!matchesNode(byId.get(e.source) || {}) || !matchesNode(byId.get(e.target) || {})) return false;
      if (!state.edgeSearch) return true;
      const q = state.edgeSearch.toLowerCase();
      return String(e.id || '').toLowerCase().includes(q) || String(e.source).toLowerCase().includes(q) || String(e.target).toLowerCase().includes(q) || String(e.relation).toLowerCase().includes(q);
    }

    function makeTextSprite(text) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      ctx.font = '26px Arial';
      const w = Math.min(760, Math.ceil(ctx.measureText(text).width + 28));
      canvas.width = w;
      canvas.height = 42;
      ctx.font = '26px Arial';
      ctx.fillStyle = 'rgba(255,255,255,.92)';
      ctx.fillRect(0, 0, w, 42);
      ctx.strokeStyle = 'rgba(100,116,139,.6)';
      ctx.strokeRect(0, 0, w, 42);
      ctx.fillStyle = '#111827';
      ctx.fillText(text.length > 52 ? text.slice(0, 51) + '…' : text, 12, 29);
      const tex = new THREE.CanvasTexture(canvas);
      const mat = new THREE.SpriteMaterial({ map: tex, depthTest:false, depthWrite:false });
      const sprite = new THREE.Sprite(mat);
      sprite.scale.set(w / 42 * 1.05, 1.05, 1);
      sprite.visible = false;
      return sprite;
    }

    const sphereGeom = new THREE.SphereGeometry(1, 20, 14);
    for (const node of nodes) {
      const mat = new THREE.MeshStandardMaterial({ color: rgbHex(node.color_rgb), roughness:0.55, metalness:0.04 });
      const mesh = new THREE.Mesh(sphereGeom, mat);
      mesh.scale.setScalar(radius(node));
      mesh.position.copy(pos(node));
      mesh.userData = { type:'node', node, baseColor: mat.color.clone(), baseScale: radius(node) };
      root.add(mesh);
      nodeMeshes.set(node.id, mesh);
      const label = makeTextSprite(nodeLabel(node));
      label.position.copy(mesh.position).add(new THREE.Vector3(radius(node) + 0.35, radius(node) + 0.35, 0));
      // Labels are shown through hover/detail panels to keep the 3D space readable.
      labelSprites.set(node.id, label);
    }

    for (const edge of edges) {
      const s = byId.get(edge.source);
      const t = byId.get(edge.target);
      if (!s || !t) continue;
      const points = [pos(s), pos(t)];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color: relationColors[edge.relation] || '#64748b', transparent:true, opacity:0.42 });
      const line = new THREE.Line(geom, mat);
      line.userData = { type:'edge', edge };
      root.add(line);
      edgeObjects.set(edge.id, line);
    }

    function visibleNodes() { return nodes.filter(matchesNode); }
    function visibleEdges() { return edges.filter(matchesEdge); }

    function updateVisibility() {
      const edgeSet = new Set(visibleEdges().map(e => e.id));
      const nodeSet = new Set(visibleNodes().map(n => n.id));
      for (const [id, mesh] of nodeMeshes) mesh.visible = nodeSet.has(id);
      for (const [id, label] of labelSprites) label.visible = false;
      for (const [id, line] of edgeObjects) line.visible = state.showEdges && edgeSet.has(id);
      updateHighlights();
      renderTables();
      updateCounts();
    }

    function updateHighlights() {
      for (const [id, mesh] of nodeMeshes) {
        const selected = state.selectedNode && state.selectedNode.id === id;
        const edgeSelected = state.selectedEdge && (state.selectedEdge.source === id || state.selectedEdge.target === id);
        const hoverNode = state.hover && state.hover.type === 'node' && state.hover.node.id === id;
        mesh.material.emissive = new THREE.Color(selected || edgeSelected ? 0x2563eb : hoverNode ? 0x111827 : 0x000000);
        mesh.material.emissiveIntensity = selected || edgeSelected || hoverNode ? 0.55 : 0;
        const scale = mesh.userData.baseScale * (selected || edgeSelected ? 1.75 : hoverNode ? 1.35 : 1);
        mesh.scale.setScalar(scale);
        const label = labelSprites.get(id);
        if (label) {
          const major = state.labelMode === 'major' && ((mesh.userData.node.kind || '').includes('region') || (mesh.userData.node.point_count || 0) >= 300);
          const searchHit = state.nodeSearch && matchesNode(mesh.userData.node);
          label.visible = (state.labelMode === 'all' && (selected || edgeSelected || hoverNode || searchHit)) || major || selected || edgeSelected || hoverNode || searchHit;
        }
      }
      for (const [id, line] of edgeObjects) {
        const selected = state.selectedEdge && state.selectedEdge.id === id;
        const hoverEdge = state.hover && state.hover.type === 'edge' && state.hover.edge.id === id;
        line.material.opacity = selected || hoverEdge ? 1.0 : 0.34;
        line.material.linewidth = selected || hoverEdge ? 4 : 1;
      }
    }

    function fitCamera() {
      const box = new THREE.Box3();
      for (const n of visibleNodes()) {
        const mesh = nodeMeshes.get(n.id);
        if (mesh && mesh.visible) box.expandByObject(mesh);
      }
      if (box.isEmpty()) return;
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxDim = Math.max(size.x, size.y, size.z, 1);
      camera.position.copy(center).add(new THREE.Vector3(maxDim * 0.95, maxDim * 0.72, maxDim * 1.05));
      camera.near = Math.max(0.01, maxDim / 1000);
      camera.far = maxDim * 20;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
    }

    function selectNode(node) {
      state.selectedNode = node;
      state.selectedEdge = null;
      updateDetail();
      updateHighlights();
      renderTables();
    }
    function selectEdge(edge) {
      state.selectedEdge = edge;
      state.selectedNode = null;
      updateDetail();
      updateHighlights();
      renderTables();
    }

    function updateDetail() {
      const detail = document.getElementById('detail');
      if (state.selectedNode) {
        const n = state.selectedNode;
        const connected = edges.filter(e => e.source === n.id || e.target === n.id);
        detail.innerHTML = `<div><strong>${n.id}</strong></div><div>类别：${n.label}</div><div>类型：${n.kind}</div><div>点数：${n.point_count}</div><div>坐标：${(n.centroid || []).map(v => Number(v).toFixed(2)).join(', ')}</div><div>关联边：${connected.length}</div>`;
        return;
      }
      if (state.selectedEdge) {
        const e = state.selectedEdge;
        detail.innerHTML = `<div><strong>${e.id}</strong></div><div>源节点：${e.source}</div><div>关系：${e.relation}</div><div>目标节点：${e.target}</div><div>置信度：${e.confidence}</div><div>证据：${JSON.stringify(e.evidence || {})}</div>`;
        return;
      }
      detail.innerHTML = '<span class="muted">未选择节点或边</span>';
    }

    function buildControls() {
      const relationFilter = document.getElementById('relationFilter');
      for (const rel of relations) {
        const opt = document.createElement('option');
        opt.value = rel;
        opt.textContent = rel;
        relationFilter.appendChild(opt);
      }
      const chips = document.getElementById('chips');
      for (const label of labels) {
        const sample = nodes.find(n => n.label === label);
        const chip = document.createElement('button');
        chip.className = 'chip';
        chip.innerHTML = `<span class="dot" style="background:${cssColor(sample.color_rgb)}"></span>${label} ${countsByLabel[label]}`;
        chip.onclick = () => {
          if (hiddenLabels.has(label)) hiddenLabels.delete(label); else hiddenLabels.add(label);
          chip.classList.toggle('off', hiddenLabels.has(label));
          updateVisibility();
        };
        chips.appendChild(chip);
      }
    }

    function renderTables() {
      const nodeTable = document.getElementById('nodeTable');
      nodeTable.innerHTML = visibleNodes().map(n => `<tr data-node="${n.id}" class="${state.selectedNode && state.selectedNode.id === n.id ? 'selected' : ''}"><td>${n.id}</td><td>${n.label}</td><td>${n.kind}</td><td>${n.point_count}</td></tr>`).join('');
      for (const tr of nodeTable.querySelectorAll('tr')) tr.onclick = () => selectNode(byId.get(tr.dataset.node));
      const edgeTable = document.getElementById('edgeTable');
      edgeTable.innerHTML = visibleEdges().map(e => `<tr data-edge="${e.id}" class="${state.selectedEdge && state.selectedEdge.id === e.id ? 'selected' : ''}"><td>${e.source}</td><td>${e.relation}</td><td>${e.target}</td><td>${e.confidence}</td></tr>`).join('');
      for (const tr of edgeTable.querySelectorAll('tr')) tr.onclick = () => selectEdge(edges.find(e => e.id === tr.dataset.edge));
    }

    function updateCounts() {
      document.getElementById('countText').textContent = `${visibleNodes().length}/${nodes.length} nodes, ${visibleEdges().length}/${edges.length} edges`;
      document.getElementById('legendText').textContent = '3D: 左键旋转，滚轮缩放，右键/中键平移；点击节点或边查看详情';
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    function pick(ev, commit=false) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.params.Line.threshold = 0.35;
      raycaster.setFromCamera(pointer, camera);
      const pickables = [...nodeMeshes.values(), ...edgeObjects.values()].filter(o => o.visible);
      const hits = raycaster.intersectObjects(pickables, false);
      const hit = hits[0]?.object;
      if (hit) {
        state.hover = hit.userData;
        tip.style.display = 'block';
        tip.style.left = `${ev.clientX - rect.left + 14}px`;
        tip.style.top = `${ev.clientY - rect.top + 14}px`;
        tip.textContent = hit.userData.type === 'node' ? nodeLabel(hit.userData.node) : edgeText(hit.userData.edge);
        if (commit) hit.userData.type === 'node' ? selectNode(hit.userData.node) : selectEdge(hit.userData.edge);
      } else {
        state.hover = null;
        tip.style.display = 'none';
        if (commit) { state.selectedNode = null; state.selectedEdge = null; updateDetail(); renderTables(); }
      }
      updateHighlights();
    }

    renderer.domElement.addEventListener('mousemove', ev => pick(ev, false));
    renderer.domElement.addEventListener('click', ev => pick(ev, true));
    renderer.domElement.addEventListener('mouseleave', () => { state.hover = null; tip.style.display = 'none'; updateHighlights(); });

    document.getElementById('nodeSearch').oninput = ev => { state.nodeSearch = ev.target.value.trim(); updateVisibility(); fitCamera(); };
    document.getElementById('edgeSearch').oninput = ev => { state.edgeSearch = ev.target.value.trim(); updateVisibility(); };
    document.getElementById('relationFilter').onchange = ev => { state.relation = ev.target.value; updateVisibility(); };
    document.getElementById('labelMode').onchange = ev => { state.labelMode = ev.target.value; updateHighlights(); };
    document.getElementById('edgeToggle').onchange = ev => { state.showEdges = ev.target.checked; updateVisibility(); };
    document.getElementById('fitBtn').onclick = fitCamera;
    document.getElementById('clearBtn').onclick = () => {
      state.nodeSearch = ''; state.edgeSearch = ''; state.relation = ''; state.selectedNode = null; state.selectedEdge = null;
      document.getElementById('nodeSearch').value = '';
      document.getElementById('edgeSearch').value = '';
      document.getElementById('relationFilter').value = '';
      updateDetail(); updateVisibility(); fitCamera();
    };

    function resize() {
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

    buildControls();
    updateDetail();
    resize();
    updateVisibility();
    fitCamera();
    animate();
  </script>
</body>
</html>
"""
    html_text = (
        html_text.replace("__TITLE__", esc(title))
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__RELATION_COLORS__", relation_colors_json)
    )
    path.write_text(html_text, encoding="utf-8")


def write_ply(path: Path, nodes: list[dict], edges: list[dict]) -> None:
    node_index = {node["id"]: idx for idx, node in enumerate(nodes)}
    edge_pairs = []
    edge_colors = []
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in node_index and tgt in node_index:
            edge_pairs.append((node_index[src], node_index[tgt]))
            color = RELATION_COLORS.get(str(edge.get("relation", "")), "#64748b").lstrip("#")
            edge_colors.append(tuple(int(color[i : i + 2], 16) for i in (0, 2, 4)))
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(nodes)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edge_pairs)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for node in nodes:
            x, y, z = [float(v) for v in node.get("centroid", [0, 0, 0])[:3]]
            r, g, b = [int(v) for v in node.get("color_rgb", [220, 220, 220])[:3]]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
        for (a, b), color in zip(edge_pairs, edge_colors):
            f.write(f"{a} {b} {color[0]} {color[1]} {color[2]}\n")


def main() -> int:
    args = parse_args()
    payload, nodes, edges = load_graph(args.scene_graph)
    out_dir = args.out_dir or (args.scene_graph.parent / "visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_three_dependencies(out_dir)
    title = args.title
    if title == "Scene Graph" and payload.get("scene_name"):
        title = f"Scene Graph - {payload['scene_name']}"

    topdown = out_dir / "scene_graph_topdown.svg"
    relation = out_dir / "scene_graph_relation.svg"
    html_path = out_dir / "scene_graph_view.html"
    ply_path = out_dir / "scene_graph_centers_edges.ply"

    topdown_svg = svg_text_topdown(title, nodes, edges, args.max_label_len)
    relation_svg = svg_text_relation(title, nodes, edges, args.max_label_len)
    if args.write_svg:
        topdown.write_text(topdown_svg, encoding="utf-8")
        relation.write_text(relation_svg, encoding="utf-8")
    write_html(html_path, title, payload)
    if args.write_ply:
        write_ply(ply_path, nodes, edges)

    print(f"[INFO] HTML: {html_path}")
    if args.write_svg:
        print(f"[INFO] SVG: {topdown}, {relation}")
    if args.write_ply:
        print(f"[INFO] PLY: {ply_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
