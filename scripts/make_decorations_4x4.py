"""
Visual decoration layer for the 4x4 SUMO grid: grass, buildings with roofs,
trees, parks and ponds, in the style of a small town / campus map.

These are SUMO polygons only (an "additional" file). They are drawn by
sumo-gui but have NO effect on traffic, so training and results are untouched.
They are loaded only by the separate demo config, never by training.

Run from the repo root (locally or in Colab):
    python scripts/make_decorations_4x4.py

Writes:
    configs/decorations_4x4.poly.xml      the shapes
    configs/gui_demo.xml                  sumo-gui view settings
    configs/simulation_4x4_demo.sumocfg   demo config (net + routes + shapes)

View locally:
    sumo-gui -c configs/simulation_4x4_demo.sumocfg
"""

import math
import os
import random
import xml.etree.ElementTree as ET

import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(HERE, "..", "configs")
NET = os.path.join(CONFIGS, "mygrid4x4.net.xml")
OUT_POLY = os.path.join(CONFIGS, "decorations_4x4.poly.xml")
OUT_GUI = os.path.join(CONFIGS, "gui_demo.xml")
OUT_CFG = os.path.join(CONFIGS, "simulation_4x4_demo.sumocfg")

SEED = 7                 # fixed, so the town looks the same every time
ROAD_MARGIN = 16.0       # metres kept clear on each side of a road centreline
PARK_SHARE = 0.25        # roughly 1 block in 4 becomes a park

# Colours (r,g,b) picked to match a warm, illustrated map style
GRASS = (152, 190, 120)
PARK = (126, 178, 98)
PATH = (222, 204, 160)
POND = (110, 170, 215)
SHADOW = (90, 110, 80)
WALLS = [(236, 214, 170), (226, 200, 150), (240, 226, 196)]
ROOFS = [(178, 74, 58), (160, 62, 48), (196, 96, 66), (120, 92, 70)]
TREE_LEAVES = [(70, 130, 70), (88, 150, 80), (60, 115, 65)]
TREE_TRUNK = (110, 80, 50)

rng = random.Random(SEED)
root = ET.Element("additional")
counter = {"n": 0}


def col(c):
    return f"{c[0]},{c[1]},{c[2]}"


def new_id(prefix):
    counter["n"] += 1
    return f"{prefix}_{counter['n']}"


def shape_str(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def add_poly(prefix, kind, colour, points, layer):
    pts = list(points) + [points[0]]  # close the ring
    ET.SubElement(root, "poly", id=new_id(prefix), type=kind, color=col(colour),
                  fill="1", layer=f"{layer}", shape=shape_str(pts))


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def circle(cx, cy, r, n=10):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def add_tree(x, y):
    r = rng.uniform(5.5, 8.5)
    add_poly("trunk", "tree", TREE_TRUNK, circle(x + 1.0, y - 1.0, r * 0.45, 6), 3)
    add_poly("tree", "tree", rng.choice(TREE_LEAVES), circle(x, y, r), 4)


def add_building(x0, y0, x1, y1):
    add_poly("shadow", "building", SHADOW, rect(x0 + 2.5, y0 - 2.5, x1 + 2.5, y1 - 2.5), 1)
    add_poly("wall", "building", rng.choice(WALLS), rect(x0, y0, x1, y1), 2)
    inset = min(x1 - x0, y1 - y0) * 0.12
    add_poly("roof", "building", rng.choice(ROOFS), rect(x0 + inset, y0 + inset, x1 - inset, y1 - inset), 3)


def fill_town_block(x0, y0, x1, y1):
    """2x2 lots, each with a building and a few trees in its garden."""
    w, h = x1 - x0, y1 - y0
    gap = 10.0
    lot_w, lot_h = (w - gap) / 2, (h - gap) / 2
    add_poly("path", "path", PATH, rect(x0 + lot_w, y0, x0 + lot_w + gap, y1), 0.5)
    add_poly("path", "path", PATH, rect(x0, y0 + lot_h, x1, y0 + lot_h + gap), 0.5)
    for i in range(2):
        for j in range(2):
            lx0 = x0 + i * (lot_w + gap)
            ly0 = y0 + j * (lot_h + gap)
            bw = lot_w * rng.uniform(0.6, 0.82)
            bh = lot_h * rng.uniform(0.45, 0.65)
            bx0 = lx0 + rng.uniform(6, max(6.1, lot_w - bw - 6))
            by0 = ly0 + rng.uniform(6, max(6.1, lot_h - bh - 6))
            add_building(bx0, by0, bx0 + bw, by0 + bh)
            for _ in range(rng.randint(1, 3)):
                tx = rng.uniform(lx0 + 5, lx0 + lot_w - 5)
                ty = rng.uniform(ly0 + 5, ly0 + lot_h - 5)
                if not (bx0 - 10 < tx < bx0 + bw + 10 and by0 - 10 < ty < by0 + bh + 10):
                    add_tree(tx, ty)


def fill_park_block(x0, y0, x1, y1):
    """Lawn with a pond, crossing paths and plenty of trees."""
    add_poly("park", "park", PARK, rect(x0, y0, x1, y1), 0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    add_poly("path", "path", PATH, rect(x0, cy - 4, x1, cy + 4), 0.5)
    add_poly("path", "path", PATH, rect(cx - 4, y0, cx + 4, y1), 0.5)
    qx = rng.choice([x0 + (x1 - x0) * 0.27, x0 + (x1 - x0) * 0.73])
    qy = rng.choice([y0 + (y1 - y0) * 0.27, y0 + (y1 - y0) * 0.73])
    pond_r = min(x1 - x0, y1 - y0) * 0.14
    add_poly("pond", "water", POND, circle(qx, qy, pond_r, 16), 1)
    for _ in range(rng.randint(10, 16)):
        tx, ty = rng.uniform(x0 + 5, x1 - 5), rng.uniform(y0 + 5, y1 - 5)
        on_path = abs(tx - cx) < 11 or abs(ty - cy) < 11
        in_pond = math.hypot(tx - qx, ty - qy) < pond_r + 10
        if not on_path and not in_pond:
            add_tree(tx, ty)


def main():
    net = sumolib.net.readNet(NET)
    tls_nodes = [net.getNode(t.getID()) for t in net.getTrafficLights()]
    xs = sorted({round(n.getCoord()[0], 2) for n in tls_nodes})
    ys = sorted({round(n.getCoord()[1], 2) for n in tls_nodes})
    (bx0, by0), (bx1, by1) = net.getBBoxXY()

    # Grass background, slightly larger than the whole network
    pad = 60.0
    add_poly("grass", "grass", GRASS, rect(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad), -10)

    # Block edges: network border, then every grid line
    x_edges = [bx0 - pad] + xs + [bx1 + pad]
    y_edges = [by0 - pad] + ys + [by1 + pad]

    blocks = []
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            x0 = x_edges[i] + (ROAD_MARGIN if i > 0 else 0)
            x1 = x_edges[i + 1] - (ROAD_MARGIN if i + 1 < len(x_edges) - 1 else 0)
            y0 = y_edges[j] + (ROAD_MARGIN if j > 0 else 0)
            y1 = y_edges[j + 1] - (ROAD_MARGIN if j + 1 < len(y_edges) - 1 else 0)
            if x1 - x0 > 40 and y1 - y0 > 40:
                inner = 0 < i < len(x_edges) - 2 and 0 < j < len(y_edges) - 2
                blocks.append((x0, y0, x1, y1, inner))

    for x0, y0, x1, y1, inner in blocks:
        if inner and rng.random() < PARK_SHARE:
            fill_park_block(x0, y0, x1, y1)
        elif inner:
            fill_town_block(x0, y0, x1, y1)
        else:
            corner = (x0 <= bx0 - pad + 1 or x1 >= bx1 + pad - 1) and (y0 <= by0 - pad + 1 or y1 >= by1 + pad - 1)
            if corner:
                # Corners: open countryside, small woods
                for _ in range(rng.randint(10, 16)):
                    add_tree(rng.uniform(x0 + 10, x1 - 10), rng.uniform(y0 + 10, y1 - 10))
            else:
                # Edge blocks along the entry roads: more houses
                fill_town_block(x0, y0, x1, y1)

    # Street trees along every road, kept off the carriageway
    for line in xs:
        for y in range(int(by0), int(by1), 34):
            for side in (-1, 1):
                add_tree(line + side * (ROAD_MARGIN + 1), y + rng.uniform(-4, 4))
    for line in ys:
        for x in range(int(bx0), int(bx1), 34):
            for side in (-1, 1):
                add_tree(x + rng.uniform(-4, 4), line + side * (ROAD_MARGIN + 1))

    ET.indent(root)
    ET.ElementTree(root).write(OUT_POLY, encoding="utf-8", xml_declaration=True)

    gui = ET.Element("viewsettings")
    ET.SubElement(gui, "scheme", name="real world")
    ET.SubElement(gui, "delay", value="50")
    ET.indent(gui)
    ET.ElementTree(gui).write(OUT_GUI, encoding="utf-8", xml_declaration=True)

    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value="mygrid4x4.net.xml")
    ET.SubElement(inp, "route-files", value="routes_4x4_uniform_seed0.rou.xml")
    ET.SubElement(inp, "additional-files", value="decorations_4x4.poly.xml")
    t = ET.SubElement(cfg, "time")
    ET.SubElement(t, "begin", value="0")
    ET.SubElement(t, "end", value="3600")
    g = ET.SubElement(cfg, "gui_only")
    ET.SubElement(g, "gui-settings-file", value="gui_demo.xml")
    ET.indent(cfg)
    ET.ElementTree(cfg).write(OUT_CFG, encoding="utf-8", xml_declaration=True)

    print(f"{len(blocks)} blocks decorated, {counter['n']} shapes written")
    print("wrote:", os.path.relpath(OUT_POLY), os.path.relpath(OUT_GUI), os.path.relpath(OUT_CFG))


if __name__ == "__main__":
    main()