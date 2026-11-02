"""Generate figures/pipeline.svg: overview of the approach (Figure 3)."""
import sys

INK, INK2, GRID = "#0B0B0B", "#52514E", "#C9C8C4"
MID, DARK = "#2A78D6", "#104281"
BAND, LANE2, LANE2S = "#F2F1EE", "#EEF4FB", "#BFD5F0"
COLD, COLDT = "#E4E3DF", "#52514E"
SERIF = "'TeX Gyre Termes', 'Times New Roman', Times, serif"
MONO = "'Liberation Mono', 'Courier New', Courier, monospace"

W, H = 1040, 452
C = [110, 310, 510, 710, 915]          # column centres
Y1, Y2 = 97, 337                       # icon centres of the two lanes
L1, L2 = 162, 402                      # label baselines

out = []
def a(s): out.append(s)

def text(x, y, s, size=15, weight="normal", style="normal", fill=INK,
         anchor="middle", family=SERIF):
    a(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
      f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
      f'text-anchor="{anchor}">{s}</text>')

def label(cx, y, title, sub, mono=False):
    text(cx, y, title, 15.5, "bold", family=MONO if mono else SERIF)
    text(cx, y + 19, sub, 13, fill=INK2)

def doc(x, y, w, h, fold=11, stroke=INK2, fill="white", sw=1.3):
    a(f'<path d="M{x},{y} h{w-fold} l{fold},{fold} v{h-fold} h{-w} Z" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')
    a(f'<path d="M{x+w-fold},{y} v{fold} h{fold}" fill="none" '
      f'stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')

def wasm_tag(x, y, w):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="14" rx="2" fill="{MID}"/>')
    text(x + w / 2, y + 10.5, "wasm", 10.5, "bold", fill="white", family=MONO)

def arrow(x1, y1, x2, y2, color=INK2, sw=1.4, marker="ah"):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
      f'stroke-width="{sw}" marker-end="url(#{marker})"/>')

a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
  f'viewBox="0 0 {W} {H}">')
a('<defs>')
for mid, col in (("ah", INK2), ("ahb", MID), ("ahd", DARK)):
    a(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
      f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
      f'fill="{col}"/></marker>')
a('</defs>')
a(f'<rect width="{W}" height="{H}" fill="white"/>')

# ── lanes ───────────────────────────────────────────────────────────
a(f'<rect x="10" y="10" width="1020" height="200" rx="10" fill="{BAND}"/>')
a(f'<rect x="10" y="248" width="1020" height="194" rx="10" fill="{LANE2}" '
  f'stroke="{LANE2S}" stroke-width="1.2"/>')
text(28, 35, "Offline — once, on a development machine", 14, style="italic",
     fill=INK2, anchor="start")
text(28, 273, "On the device — every time Walrus loads a module", 14,
     style="italic", fill=DARK, anchor="start")

# ── lane 1: 1. training benchmarks ─────────────────────────────────
cx = C[0]
for i, (dx, dy) in enumerate(((-34, -38), (-25, -30), (-16, -22))):
    doc(cx + dx, Y1 + dy, 46, 58)
wasm_tag(cx - 12, Y1 + 12, 38)
label(cx, L1, "Training benchmarks", "232 modules")

# 2. measure per function
cx = C[1]
x0, yb = cx - 52, Y1 + 34
a(f'<line x1="{x0-4}" y1="{yb}" x2="{x0+104}" y2="{yb}" stroke="{INK2}" stroke-width="1.2"/>')
a(f'<line x1="{x0-4}" y1="{yb}" x2="{x0-4}" y2="{Y1-36}" stroke="{INK2}" stroke-width="1.2"/>')
for i, h in enumerate((60, 34, 14, 9, 6, 4, 3)):
    col = MID if i < 2 else "#B9B8B3"
    a(f'<rect x="{x0 + 2 + i*14}" y="{yb - h}" width="10" height="{h}" fill="{col}"/>')
# stopwatch
sx, sy = cx + 40, Y1 - 22
a(f'<circle cx="{sx}" cy="{sy}" r="13" fill="white" stroke="{INK2}" stroke-width="1.5"/>')
a(f'<rect x="{sx-3}" y="{sy-19}" width="6" height="4" fill="{INK2}"/>')
a(f'<line x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy-8}" stroke="{INK}" stroke-width="1.6"/>')
a(f'<line x1="{sx}" y1="{sy}" x2="{sx+6}" y2="{sy+3}" stroke="{INK}" stroke-width="1.6"/>')
label(cx, L1, "Measure per function", "self time and machine code")

# 3. label each function
cx = C[2]
rows = (("f0", "cold"), ("f1", "hot"), ("f2", "cold"), ("f3", "cold"))
for i, (fn, tag) in enumerate(rows):
    y = Y1 - 34 + i * 18
    a(f'<rect x="{cx-56}" y="{y}" width="112" height="15" rx="3" fill="white" '
      f'stroke="{GRID}" stroke-width="1"/>')
    text(cx - 48, y + 11.5, fn, 11.5, family=MONO, anchor="start", fill=INK)
    a(f'<rect x="{cx-30}" y="{y+6}" width="{[38,22,30,14][i]}" height="3" fill="#D8D7D3"/>')
    fill, tf = (MID, "white") if tag == "hot" else (COLD, COLDT)
    a(f'<rect x="{cx+18}" y="{y+2}" width="34" height="11" rx="5.5" fill="{fill}"/>')
    text(cx + 35, y + 11, tag, 9.5, "bold", fill=tf)
label(cx, L1, "Label each function",
      "hot ⇔ <tspan font-style=\"italic\">t</tspan>(<tspan font-style=\"italic\">f</tspan>) "
      "≥ <tspan font-style=\"italic\">θ</tspan>·<tspan font-style=\"italic\">c</tspan>"
      "(<tspan font-style=\"italic\">f</tspan>)")

# 4. one tree per theta
def tree(x, y, s, hot_leaf, path=None, r=5, leaf=9, edge=GRID, sw=1.3):
    """Two-level binary tree with root at (x, y) and spacing s."""
    n1 = [(x - s, y + 1.9 * s), (x + s, y + 1.9 * s)]
    n2 = [(x - 1.5 * s, y + 3.8 * s), (x - 0.5 * s, y + 3.8 * s),
          (x + 0.5 * s, y + 3.8 * s), (x + 1.5 * s, y + 3.8 * s)]
    edges = [((x, y), n1[0], (0, 0)), ((x, y), n1[1], (0, 1)),
             (n1[0], n2[0], (1, 0)), (n1[0], n2[1], (1, 1)),
             (n1[1], n2[2], (1, 2)), (n1[1], n2[3], (1, 3))]
    for p, q, key in edges:
        on = path is not None and key in path
        a(f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" '
          f'stroke="{MID if on else edge}" stroke-width="{sw*1.8 if on else sw}"/>')
    for (nx, ny), key in ((((x, y)), (0,)), (n1[0], (1, 0)), (n1[1], (1, 1))):
        on = path is not None and (key == (0,) or (0, key[1]) in path)
        a(f'<circle cx="{nx}" cy="{ny}" r="{r}" fill="{MID if on else "white"}" '
          f'stroke="{MID if on else INK2}" stroke-width="1.3"/>')
    for i, (nx, ny) in enumerate(n2):
        col = MID if i == hot_leaf else COLD
        a(f'<rect x="{nx-leaf/2}" y="{ny-leaf/2}" width="{leaf}" height="{leaf}" rx="2" '
          f'fill="{col}" stroke="{MID if i == hot_leaf else "#B9B8B3"}" stroke-width="1"/>')

cx = C[3]
for i, (dx, th, hl) in enumerate(((-44, "4", 3), (0, "2", 1), (44, "1", 2))):
    tree(cx + dx, Y1 - 34, 9.5, hl)
    text(cx + dx, Y1 + 30, f'<tspan font-style="italic">θ</tspan> = {th}', 12.5,
         fill=INK2)
label(cx, L1, "Train one tree per <tspan font-style=\"italic\">θ</tspan>",
      "CART, depth ≤ 6")

# 5. generated header
cx = C[4]
doc(cx - 46, Y1 - 38, 92, 74, fold=14, stroke=DARK, sw=2)
a(f'<rect x="{cx-46}" y="{Y1-38}" width="78" height="17" fill="{DARK}"/>')
text(cx - 40, Y1 - 25.5, "model.h", 11.5, "bold", fill="white", family=MONO, anchor="start")
for i, w in enumerate((62, 48, 70, 54)):
    a(f'<rect x="{cx-36}" y="{Y1-12+i*10}" width="{w}" height="4" rx="2" '
      f'fill="{"#9DB9DE" if i % 2 else "#C9D8EC"}"/>')
label(cx, L1, "Generated header", "0.95 KB of integer tables")

for i in range(4):
    arrow(C[i] + 66, Y1, C[i + 1] - 64, Y1)

# ── lane 2: 1. module.wasm ─────────────────────────────────────────
cx = C[0]
doc(cx - 26, Y2 - 34, 52, 66, fold=12)
wasm_tag(cx - 18, Y2 + 2, 36)
for i, w in enumerate((30, 22)):
    a(f'<rect x="{cx-18}" y="{Y2-20+i*9}" width="{w}" height="3.5" rx="1.7" fill="#D8D7D3"/>')
label(cx, L2, "module.wasm", "the portable binary", mono=True)

# 2. extra pass over the binary
cx = C[1]
hexes = ("00 61 73 6D 01".split(), "0A 8B 02 03 7F".split())
for r, row in enumerate(hexes):
    for i, hx in enumerate(row):
        x = cx - 57 + i * 23
        y = Y2 - 32 + r * 24
        hl = r == 1 and i in (1, 2)
        a(f'<rect x="{x}" y="{y}" width="21" height="20" rx="2" '
          f'fill="{"#D6E5F7" if hl else "white"}" stroke="{MID if hl else GRID}" stroke-width="1"/>')
        text(x + 10.5, y + 14, hx, 10.5, family=MONO, fill=INK2)
arrow(cx - 57, Y2 + 24, cx + 56, Y2 + 24, color=MID, sw=1.8, marker="ahb")
label(cx, L2, "Extra pass over the binary", "body and call sites")

# 3. ten integers per function
cx = C[2]
vals = ("4", "12", "0", "1", "2", "3", "1", "2", "1", "96")
for i, v in enumerate(vals):
    r, k = divmod(i, 5)
    x = cx - 58 + k * 24
    y = Y2 - 30 + r * 26
    a(f'<rect x="{x}" y="{y}" width="21" height="21" rx="3" fill="white" '
      f'stroke="{INK2}" stroke-width="1.1"/>')
    text(x + 10.5, y + 15, v, 11.5, family=MONO, fill=INK)
label(cx, L2, "10 integers per function", "body and call-graph features")

# 4. walk the trees
cx = C[3]
tree(cx, Y2 - 34, 17, 2, path={(0, 1), (1, 2)}, r=7, leaf=13, edge="#B9B8B3", sw=1.4)
text(cx + 32, Y2 + 36, "hot", 11.5, "bold", fill=MID, anchor="start")
label(cx, L2, "Walk the trees", "≤ 6 integer comparisons")

# 5. compile or interpret
cx = C[4]
bx, bw, bh = cx - 52, 128, 30
yj, yi = Y2 - 38, Y2 + 6
a(f'<rect x="{bx}" y="{yj}" width="{bw}" height="{bh}" rx="5" fill="white" '
  f'stroke="{MID}" stroke-width="1.8"/>')
# chip icon
chx, chy = bx + 10, yj + 7
a(f'<rect x="{chx}" y="{chy}" width="16" height="16" rx="2" fill="{MID}"/>')
for k in range(3):
    for (x1, y1, x2, y2) in ((chx + 4 + k*4, chy - 3, chx + 4 + k*4, chy),
                             (chx + 4 + k*4, chy + 16, chx + 4 + k*4, chy + 19),
                             (chx - 3, chy + 4 + k*4, chx, chy + 4 + k*4),
                             (chx + 16, chy + 4 + k*4, chx + 19, chy + 4 + k*4)):
        a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MID}" stroke-width="1.2"/>')
text(bx + 38, yj + 20, "JIT compile", 14, "bold", fill=DARK, anchor="start")
a(f'<rect x="{bx}" y="{yi}" width="{bw}" height="{bh}" rx="5" fill="white" '
  f'stroke="{INK2}" stroke-width="1.2"/>')
# loop icon (interpreter dispatch)
lx, ly = bx + 18, yi + 15
a(f'<path d="M{lx+7},{ly-3} A8,8 0 1 0 {lx+6},{ly+5}" fill="none" stroke="{INK2}" '
  f'stroke-width="1.6" marker-end="url(#ah)"/>')
text(bx + 38, yi + 20, "Interpret", 14, "bold", fill=INK2, anchor="start")
# branches from the tree
sx, sy = C[3] + 50, Y2 - 2
a(f'<path d="M{sx},{sy} H{bx-22} V{yj+bh/2} H{bx-2}" fill="none" stroke="{MID}" '
  f'stroke-width="1.6" marker-end="url(#ahb)"/>')
a(f'<path d="M{bx-22},{sy} V{yi+bh/2} H{bx-2}" fill="none" stroke="{INK2}" '
  f'stroke-width="1.4" marker-end="url(#ah)"/>')
text(bx - 27, yj + 12, "hot", 12, "bold", fill=MID, anchor="end")
text(bx - 27, yi + 27, "cold", 12, "bold", fill=INK2, anchor="end")
label(cx, L2, "Compile or interpret", "decided before the first call")

for i in range(3):
    arrow(C[i] + 66, Y2, C[i + 1] - 66, Y2)

# ── the only link between the phases ─────────────────────────────
hx = C[4]
a(f'<path d="M{hx},{L1+25} V230 H{C[3]} V{Y2-46}" fill="none" stroke="{DARK}" '
  f'stroke-width="2" marker-end="url(#ahd)"/>')
text((hx + C[3]) / 2 + 8, 225, "compiled into Walrus", 13, style="italic", fill=DARK)

a('</svg>')
open(sys.argv[1], "w").write("\n".join(out))
