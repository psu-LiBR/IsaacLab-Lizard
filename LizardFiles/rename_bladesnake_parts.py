# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rename the randomly-named parts of Onshape-published BladeSnake assemblies.

The Onshape publisher names every link prim, joint prim and mesh file after the
part's occurrence ID (e.g. ``_MY6_8coFRhVRStLUa``). This script walks the joint
graph in ``robot.usda`` to recover the kinematic chain, which alternates
link-servo-link-...-link (L-S-L-S-L-S-L-S-L-S-L), and renames everything:

* links:            ``link1`` ... ``link6`` and ``servo1`` ... ``servo5`` (front to tail)
* revolute joints:  ``joint1`` ... ``joint5`` (front to tail)
* fixed joints:     ``fixed1`` ... ``fixed5`` (front to tail)
* mesh files:       ``meshes/link1.gltf``, ``meshes/servo1.gltf``, ...

The renaming is applied consistently to ``robot.usda`` (prim names, display names,
relationships, mesh references), ``robot.json`` (occurrence/feature IDs and names)
and the ``meshes/*.gltf`` filenames. A ``rename_map.json`` recording the old and
new names is written next to each ``robot.usda``; ``--undo`` uses it to restore
the original Onshape names exactly.

Onshape also publishes the servo parts without mass properties (negative mass, zero
inertia). Any servo link whose mass is not set gets ``--servo-mass`` (default 57.2 g,
the XL430-W250-T), with the center of mass and inertia computed from its mesh assuming
uniform density. ``--undo`` only restores names; the mass fix is kept.

Check in Isaacsim or another sim platform for joint orientation. Some joints may come in
with opposite directions because of the order the parts were assigned in Onshape.

Usage::

    python LizardFiles/rename_bladesnake_parts.py              # all assemblies in LizardFiles/BladeSnake
    python LizardFiles/rename_bladesnake_parts.py --dry-run    # only print the mapping
    python LizardFiles/rename_bladesnake_parts.py --undo       # restore the original Onshape names
    python LizardFiles/rename_bladesnake_parts.py "LizardFiles/BladeSnake/20mm Assembly"
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "BladeSnake"
MAP_FILE = "rename_map.json"
DEFAULT_SERVO_MASS_G = 57.2  # Dynamixel XL430-W250-T
UNSET_MASS_KG = 1e-6  # masses at or below this are treated as "not set"

# Display-name hints used to identify the head end of the chain and the servo bodies.
FRONT_HINT = re.compile(r"front|head", re.IGNORECASE)
SERVO_HINT = re.compile(r"servo|xl430|xm430|dynamixel", re.IGNORECASE)
# Matches the names this script generates when they stand alone as a token.
NEW_NAME_RE = re.compile(r"(?<![A-Za-z0-9])(?:link|servo|joint|fixed)\d+(?![0-9])")

JOINT_RE = re.compile(
    r'def (?P<type>Physics\w*Joint) "(?P<name>[^"]+)"\s*\((?P<meta>[^)]*)\)\s*\{(?P<body>[^{}]*)\}',
    re.DOTALL,
)
REL_RE = r"rel physics:{} = </robot/links/(?P<link>[^>]+)>"
ROBOT_LINKS_RE = re.compile(r"rel isaac:physics:robotLinks = \[(?P<links>[^\]]*)\]")
DISPLAY_NAME_RE = re.compile(r'displayName = "(?P<display>[^"]*)"')


# ----------------------------------------------------------------------------- file helpers


def read_text(path: Path) -> str:
    """Read a text file without newline translation so line endings are preserved on write."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def json_literal(value: str) -> list[str]:
    """Possible spellings of a JSON string literal (plain, and Go-style escaped as Onshape writes it)."""
    plain = json.dumps(value)
    go_style = plain.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return list(dict.fromkeys([go_style, plain]))


def sanitize_onshape_id(raw_id: str) -> str:
    """Convert a raw Onshape occurrence/feature ID to the prim name the publisher uses."""
    return "_" + re.sub(r"[^A-Za-z0-9_]", "_", raw_id)


# ----------------------------------------------------------------------------- parsing


def link_display_name(usda: str, prim_name: str) -> str:
    match = re.search(rf'def Xform "{re.escape(prim_name)}"\s*\((?P<meta>[^)]*)\)', usda)
    if match is None:
        raise ValueError(f"Link prim '{prim_name}' not found in robot.usda")
    display = DISPLAY_NAME_RE.search(match.group("meta"))
    return display.group("display") if display else prim_name


def parse_usda(usda: str) -> tuple[dict[str, str], list[dict]]:
    """Return ``({link prim: display name}, [joint dicts])`` parsed from the published USD."""
    links_match = ROBOT_LINKS_RE.search(usda)
    if links_match is None:
        raise ValueError("Could not find 'isaac:physics:robotLinks' in robot.usda")
    link_names = re.findall(r"</robot/links/([^>]+)>", links_match.group("links"))
    links = {name: link_display_name(usda, name) for name in link_names}

    joints = []
    for match in JOINT_RE.finditer(usda):
        body = match.group("body")
        body0 = re.search(REL_RE.format("body0"), body)
        body1 = re.search(REL_RE.format("body1"), body)
        if body0 is None or body1 is None:
            raise ValueError(f"Joint '{match.group('name')}' does not connect two links")
        display = DISPLAY_NAME_RE.search(match.group("meta"))
        joints.append({
            "prim": match.group("name"),
            "type": match.group("type"),
            "display": display.group("display") if display else match.group("name"),
            "bodies": (body0.group("link"), body1.group("link")),
        })
    return links, joints


def collect_json_ids(data, found: set[str]) -> set[str]:
    """Collect every ID-like string in the published JSON (``Id``/``FeatureId`` values and dict keys)."""
    if isinstance(data, dict):
        for key, value in data.items():
            found.add(key)
            if key in ("Id", "FeatureId") and isinstance(value, str):
                found.add(value)
            collect_json_ids(value, found)
    elif isinstance(data, list):
        for value in data:
            collect_json_ids(value, found)
    return found


# ----------------------------------------------------------------------------- chain + mapping


def find_chain(links: dict[str, str], joints: list[dict], fixed_instances: set[str]) -> list[str]:
    """Order the links from front to tail by walking the joint graph."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for joint in joints:
        a, b = joint["bodies"]
        adjacency[a].append(b)
        adjacency[b].append(a)

    ends = [name for name in links if len(adjacency[name]) == 1]
    if len(ends) != 2 or any(len(adjacency[name]) > 2 for name in links):
        degrees = {links[n]: len(adjacency[n]) for n in links}
        raise ValueError(f"Joint graph is not a simple chain (connections per link: {degrees})")

    # Prefer the end whose name says "front"/"head"; otherwise the part fixed in the Onshape assembly.
    front_ends = [n for n in ends if FRONT_HINT.search(links[n])] or [n for n in ends if n in fixed_instances]
    if len(front_ends) != 1:
        raise ValueError(f"Cannot tell which end is the front: {[links[n] for n in ends]}")

    chain, previous = [front_ends[0]], None
    while len(chain) < len(links):
        current = chain[-1]
        nxt = [n for n in adjacency[current] if n != previous]
        previous = current
        chain.append(nxt[0])
    if len(set(chain)) != len(links):
        raise ValueError("Joint graph is disconnected; not every link was reached from the front")
    return chain


def build_mapping(links: dict[str, str], joints: list[dict], chain: list[str]) -> dict[str, dict]:
    """Return ``{old prim name: {"new": ..., "old_display": ...}}`` for links and joints, front to tail."""
    if len(chain) % 2 == 0:
        raise ValueError(f"Expected an odd number of bodies (L-S-...-L), found {len(chain)}")

    mapping: dict[str, dict] = {}
    for index, prim in enumerate(chain):
        is_servo = index % 2 == 1
        new = f"servo{index // 2 + 1}" if is_servo else f"link{index // 2 + 1}"
        if bool(SERVO_HINT.search(links[prim])) != is_servo:
            raise ValueError(
                f"Chain does not alternate link/servo: '{links[prim]}' sits in the {new} position. "
                f"Chain: {' - '.join(links[p] for p in chain)}"
            )
        mapping[prim] = {"new": new, "old_display": links[prim]}

    counters: dict[str, int] = defaultdict(int)
    for joint in joints:  # already sorted front to tail
        prefix = "fixed" if joint["type"] == "PhysicsFixedJoint" else "joint"
        counters[prefix] += 1
        mapping[joint["prim"]] = {"new": f"{prefix}{counters[prefix]}", "old_display": joint["display"]}
    return mapping


# ----------------------------------------------------------------------------- text rewriting


def set_display_names(usda: str, names: dict[str, str]) -> str:
    """Set ``displayName`` of the prims named in ``names`` (prim name -> display name)."""
    for prim, display in names.items():
        usda, count = re.subn(
            rf'(def \w+ "{re.escape(prim)}"\s*\([^)]*?displayName = )"[^"]*"',
            lambda m, d=display: f'{m.group(1)}"{d}"',
            usda,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"Could not find displayName of prim '{prim}'")
    return usda


def replace_tokens(text: str, replacements: dict[str, str], pattern: re.Pattern | None = None) -> str:
    if pattern is None:
        pattern = re.compile("|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda m: replacements.get(m.group(0), m.group(0)), text)


def replace_json_literals(text: str, replacements: dict[str, str], name_replacements: dict[str, str]) -> str:
    """Replace exact JSON string literals (keys or values). ``name_replacements`` only apply to ``"Name"`` values."""
    for old, new in name_replacements.items():
        for literal in json_literal(old):
            text = re.sub(rf'("Name":\s*){re.escape(literal)}', lambda m, n=new: m.group(1) + json_literal(n)[0], text)
    literals = {lit: json_literal(new)[0] for old, new in replacements.items() for lit in json_literal(old)}
    return replace_tokens(text, literals) if literals else text


# ----------------------------------------------------------------------------- rename / undo


def rename_assembly(folder: Path, dry_run: bool) -> None:
    usda_path, json_path, mesh_dir = folder / "robot.usda", folder / "robot.json", folder / "meshes"
    usda = read_text(usda_path)
    json_text = read_text(json_path) if json_path.exists() else None
    data = json.loads(json_text) if json_text is not None else {}

    links, joints = parse_usda(usda)
    if (folder / MAP_FILE).exists() or all(NEW_NAME_RE.fullmatch(name) for name in links):
        print(f"[skip] {folder.name}: already renamed (use --undo to restore the Onshape names)")
        return

    fixed_instances = {sanitize_onshape_id(f["Id"]) for f in data.get("FixedInstances", [])}
    chain = find_chain(links, joints, fixed_instances)
    position = {prim: i for i, prim in enumerate(chain)}
    joints.sort(key=lambda j: min(position[b] for b in j["bodies"]))
    mapping = build_mapping(links, joints, chain)

    print(f"\n== {folder.name}")
    for prim in chain:
        print(f"  {mapping[prim]['new']:<8} <- {mapping[prim]['old_display']:<32} ({prim})")
    for joint in joints:
        a, b = (mapping[x]["new"] for x in joint["bodies"])
        print(f"  {mapping[joint['prim']]['new']:<8} <- {joint['display']:<32} ({a} <-> {b})")

    meshes = {prim: mesh_dir / f"{prim}.gltf" for prim in links}
    missing = [str(p) for p in meshes.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Mesh files referenced by robot.usda are missing: {missing}")
    # The new names must not already occur, otherwise the rename could not be undone exactly.
    for name, text in (("robot.usda", usda), ("robot.json", json_text or "")):
        clashes = sorted(set(NEW_NAME_RE.findall(text)))
        if clashes:
            raise RuntimeError(f"{name} already contains names like {clashes}; refusing to rename")

    # Map raw JSON IDs (e.g. "MY6+8coFRhVRStLUa") onto the prims they were sanitized into.
    json_ids: dict[str, str] = {}
    for raw in collect_json_ids(data, set()):
        prim = sanitize_onshape_id(raw)
        if prim in mapping:
            if prim in json_ids:
                raise RuntimeError(f"Two JSON IDs sanitize to the same prim '{prim}'")
            json_ids[prim] = raw
    unmatched = [mapping[p]["old_display"] for p in mapping if p not in json_ids]
    if json_text is not None and unmatched:
        print(f"  [warn] robot.json has no IDs for: {unmatched}")
    if dry_run:
        return

    new_usda = replace_tokens(usda, {old: m["new"] for old, m in mapping.items()})
    new_usda = set_display_names(new_usda, {m["new"]: m["new"] for m in mapping.values()})
    leftovers = [old for old in mapping if old in new_usda]
    if leftovers:
        raise RuntimeError(f"Old names still present in robot.usda after renaming: {leftovers}")

    if json_text is not None:
        new_json = replace_json_literals(
            json_text,
            {raw: mapping[prim]["new"] for prim, raw in json_ids.items()},
            {m["old_display"]: m["new"] for m in mapping.values()},
        )
        json.loads(new_json)  # make sure it is still valid JSON
        write_text(json_path, new_json)
    write_text(usda_path, new_usda)
    for prim, mesh in meshes.items():
        mesh.rename(mesh_dir / f"{mapping[prim]['new']}.gltf")

    def entry(prim: str) -> dict:
        return {"old_prim": prim, "old_name": mapping[prim]["old_display"], "json_id": json_ids.get(prim)}

    rename_map = {
        "links": {mapping[p]["new"]: entry(p) for p in chain},
        "joints": {
            mapping[j["prim"]]["new"]: {
                **entry(j["prim"]),
                "type": j["type"],
                "connects": [mapping[b]["new"] for b in j["bodies"]],
            }
            for j in joints
        },
    }
    write_text(folder / MAP_FILE, json.dumps(rename_map, indent=2) + "\n")
    print(f"  [done] renamed {len(links)} links, {len(joints)} joints, {len(meshes)} mesh files")


def undo_assembly(folder: Path, dry_run: bool) -> None:
    usda_path, json_path, mesh_dir = folder / "robot.usda", folder / "robot.json", folder / "meshes"
    map_path = folder / MAP_FILE
    if not map_path.exists():
        print(f"[skip] {folder.name}: no {MAP_FILE}, nothing to undo")
        return
    rename_map = json.loads(read_text(map_path))
    entries = {**rename_map["links"], **rename_map["joints"]}
    print(f"[undo] {folder.name}: restoring {len(entries)} Onshape names")
    if dry_run:
        return

    usda = set_display_names(read_text(usda_path), {new: e["old_name"] for new, e in entries.items()})
    usda = replace_tokens(usda, {new: e["old_prim"] for new, e in entries.items()}, NEW_NAME_RE)
    write_text(usda_path, usda)

    if json_path.exists():
        if any(e.get("json_id") is None for e in entries.values()):
            print(f"  [warn] {MAP_FILE} has no JSON IDs; restore robot.json from version control instead")
        else:
            json_text = replace_json_literals(
                read_text(json_path),
                {new: e["json_id"] for new, e in entries.items()},
                {new: e["old_name"] for new, e in entries.items()},
            )
            json.loads(json_text)
            write_text(json_path, json_text)

    for new, e in rename_map["links"].items():
        (mesh_dir / f"{new}.gltf").rename(mesh_dir / f"{e['old_prim']}.gltf")
    map_path.unlink()


# ----------------------------------------------------------------------------- servo mass properties

Vec3 = tuple[float, float, float]
Mat3 = list[list[float]]


def load_gltf_triangles(path: Path) -> list[tuple[Vec3, Vec3, Vec3]]:
    """Read the triangles of an Onshape-exported glTF (embedded buffer, no node transforms)."""
    gltf = json.loads(read_text(path))
    if any(key in node for node in gltf["nodes"] for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError(f"{path.name}: node transforms are not supported")
    buffer = base64.b64decode(gltf["buffers"][0]["uri"].split(",", 1)[1])

    def accessor(index: int) -> tuple:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        count = {"SCALAR": 1, "VEC3": 3}[acc["type"]] * acc["count"]
        fmt = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}[acc["componentType"]]
        return struct.unpack_from(f"<{count}{fmt}", buffer, view.get("byteOffset", 0) + acc.get("byteOffset", 0))

    triangles = []
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                raise ValueError(f"{path.name}: only triangle meshes are supported")
            flat = accessor(prim["attributes"]["POSITION"])
            points = [flat[i : i + 3] for i in range(0, len(flat), 3)]
            indices = accessor(prim["indices"]) if "indices" in prim else range(len(points))
            triangles += [tuple(points[indices[i + k]] for k in range(3)) for i in range(0, len(indices), 3)]
    return triangles


def is_closed_mesh(triangles: list[tuple[Vec3, Vec3, Vec3]]) -> bool:
    """True if every edge (welded by position) is shared by exactly two triangles."""
    edges: Counter = Counter()
    for tri in triangles:
        keys = [tuple(round(c, 7) for c in v) for v in tri]
        for i in range(3):
            edges[frozenset((keys[i], keys[(i + 1) % 3]))] += 1
    return all(count == 2 for count in edges.values())


def solid_mass_properties(triangles: list[tuple[Vec3, Vec3, Vec3]], mass: float) -> tuple[Vec3, Mat3]:
    """Center of mass and inertia tensor (about the COM) of a closed mesh with uniform density.

    Uses D. Eberly, "Polyhedral Mass Properties (Revisited)".
    """
    s = [0.0] * 10
    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in triangles:
        a1, b1, c1, a2, b2, c2 = x1 - x0, y1 - y0, z1 - z0, x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1

        def sub(w0, w1, w2):
            t0, t1 = w0 + w1, w0 * w0
            f1 = t0 + w2
            t2 = t1 + w1 * t0
            f2 = t2 + w2 * f1
            f3 = w0 * t1 + w1 * t2 + w2 * f2
            return f1, f2, f3, f2 + w0 * (f1 + w0), f2 + w1 * (f1 + w1), f2 + w2 * (f1 + w2)

        f1x, f2x, f3x, g0x, g1x, g2x = sub(x0, x1, x2)
        _, f2y, f3y, g0y, g1y, g2y = sub(y0, y1, y2)
        _, f2z, f3z, g0z, g1z, g2z = sub(z0, z1, z2)
        s[0] += d0 * f1x
        s[1] += d0 * f2x
        s[2] += d1 * f2y
        s[3] += d2 * f2z
        s[4] += d0 * f3x
        s[5] += d1 * f3y
        s[6] += d2 * f3z
        s[7] += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
        s[8] += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
        s[9] += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)
    s = [v * k for v, k in zip(s, [1 / 6] + [1 / 24] * 3 + [1 / 60] * 3 + [1 / 120] * 3)]
    volume = s[0]
    if volume <= 0:
        raise ValueError("mesh has non-positive volume (inverted or open)")
    cx, cy, cz = s[1] / volume, s[2] / volume, s[3] / volume
    rho = mass / volume
    ixx = rho * (s[5] + s[6]) - mass * (cy * cy + cz * cz)
    iyy = rho * (s[4] + s[6]) - mass * (cz * cz + cx * cx)
    izz = rho * (s[4] + s[5]) - mass * (cx * cx + cy * cy)
    ixy = -(rho * s[7] - mass * cx * cy)
    iyz = -(rho * s[8] - mass * cy * cz)
    ixz = -(rho * s[9] - mass * cz * cx)
    return (cx, cy, cz), [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]


def box_mass_properties(triangles: list[tuple[Vec3, Vec3, Vec3]], mass: float) -> tuple[Vec3, Mat3]:
    """Fallback for open meshes: a solid box filling the mesh's bounding box."""
    lo = [min(v[i] for t in triangles for v in t) for i in range(3)]
    hi = [max(v[i] for t in triangles for v in t) for i in range(3)]
    dx, dy, dz = (hi[i] - lo[i] for i in range(3))
    com = tuple((hi[i] + lo[i]) / 2 for i in range(3))
    k = mass / 12
    return com, [[k * (dy * dy + dz * dz), 0, 0], [0, k * (dx * dx + dz * dz), 0], [0, 0, k * (dx * dx + dy * dy)]]


def principal_axes(inertia: Mat3) -> tuple[Vec3, tuple[float, float, float, float]]:
    """Diagonalize a symmetric 3x3 tensor; returns (diagonal inertia, quaternion (w, x, y, z)) as USD expects."""
    a = [row[:] for row in inertia]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(50):  # cyclic Jacobi rotations
        off = max(abs(a[0][1]), abs(a[0][2]), abs(a[1][2]))
        if off <= 1e-30 + 1e-15 * max(abs(a[i][i]) for i in range(3)):
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if a[p][q] == 0:
                continue
            theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
            c = 1 / math.sqrt(t * t + 1)
            s = t * c
            for k in range(3):  # A <- A J, V <- V J
                akp, akq = a[k][p], a[k][q]
                a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                vkp, vkq = v[k][p], v[k][q]
                v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
            for k in range(3):  # A <- J^T A
                apk, aqk = a[p][k], a[q][k]
                a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
    det = (
        v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1])
        - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
        + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0])
    )
    if det < 0:  # make it a proper rotation
        for k in range(3):
            v[k][2] = -v[k][2]
    # Rotation matrix (columns = principal axes) to quaternion.
    m = v
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        quat = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2 * math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
        quat = ((m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = 2 * math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
        quat = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
        quat = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4)
    if quat[0] < 0:
        quat = tuple(-q for q in quat)
    return (a[0][0], a[1][1], a[2][2]), quat


def usd_tuple(values) -> str:
    return "(" + ", ".join(f"{v:.9g}" for v in values) + ")"


def set_link_attribute(header: str, decl: str, value: str) -> str:
    """Replace (or add) ``<decl> = <value>`` in the attribute section of a link prim."""
    pattern = re.compile(rf"^(?P<indent>[ \t]*){re.escape(decl)} = [^\r\n]*", re.MULTILINE)
    if pattern.search(header):
        return pattern.sub(lambda m: f"{m.group('indent')}{decl} = {value}", header, count=1)
    match = re.search(r"\{(?P<nl>\r?\n)(?P<indent>[ \t]*)", header)
    pos = match.end("nl")
    return header[:pos] + f"{match.group('indent')}{decl} = {value}{match.group('nl')}" + header[pos:]


def fix_servo_masses(folder: Path, servo_mass_kg: float, dry_run: bool) -> None:
    """Give servo links with unset mass a real mass, center of mass and inertia."""
    usda_path = folder / "robot.usda"
    usda = read_text(usda_path)
    links, _ = parse_usda(usda)
    changed = False
    for prim, display in sorted(links.items()):
        if not (SERVO_HINT.search(display) or SERVO_HINT.search(prim)):
            continue
        start = re.search(rf'def Xform "{re.escape(prim)}"\s*\(', usda).start()
        end = usda.find('def Xform "visuals"', start)
        header = usda[start:end]
        mass_match = re.search(r"float physics:mass = (?P<mass>[^\s]+)", header)
        if mass_match and float(mass_match.group("mass")) > UNSET_MASS_KG:
            continue

        mesh_ref = re.search(r"references = \[@(?P<path>[^@]+\.gltf)@\]", usda[end:])
        mesh_path = folder / mesh_ref.group("path")
        triangles = load_gltf_triangles(mesh_path)
        if is_closed_mesh(triangles):
            com, inertia = solid_mass_properties(triangles, servo_mass_kg)
            method = "mesh, uniform density"
        else:
            com, inertia = box_mass_properties(triangles, servo_mass_kg)
            method = "bounding box (mesh is not closed)"
        diagonal, quat = principal_axes(inertia)
        print(
            f"  [mass] {prim}: {servo_mass_kg * 1000:g} g ({method}), "
            f"com={usd_tuple(com)}, inertia={usd_tuple(diagonal)}"
        )
        if dry_run:
            continue
        header = set_link_attribute(header, "quatf physics:principalAxes", usd_tuple(quat))
        header = set_link_attribute(header, "float3 physics:diagonalInertia", usd_tuple(diagonal))
        header = set_link_attribute(header, "float physics:mass", f"{servo_mass_kg:.9g}")
        header = set_link_attribute(header, "point3f physics:centerOfMass", usd_tuple(com))
        usda = usda[:start] + header + usda[end:]
        changed = True
    if changed:
        write_text(usda_path, usda)


# ----------------------------------------------------------------------------- CLI


def find_assemblies(paths: list[Path]) -> list[Path]:
    folders = []
    for path in paths:
        if (path / "robot.usda").exists():
            folders.append(path)
        else:
            folders.extend(sorted(p.parent for p in path.glob("*/robot.usda")))
    return folders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_ROOT],
        help="Assembly folders, or folders containing assembly folders (default: LizardFiles/BladeSnake).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without touching files.")
    parser.add_argument("--undo", action="store_true", help=f"Restore the original Onshape names using {MAP_FILE}.")
    parser.add_argument(
        "--servo-mass",
        type=float,
        default=DEFAULT_SERVO_MASS_G,
        help=f"Mass in grams given to servos whose mass is not set (default: {DEFAULT_SERVO_MASS_G}).",
    )
    parser.add_argument("--skip-mass-fix", action="store_true", help="Do not touch servo mass properties.")
    args = parser.parse_args()

    folders = find_assemblies(args.paths)
    if not folders:
        print(f"No folders containing robot.usda found in: {[str(p) for p in args.paths]}")
        return 1

    failures = 0
    for folder in folders:
        try:
            if args.undo:
                undo_assembly(folder, args.dry_run)
                continue
            rename_assembly(folder, args.dry_run)
            if not args.skip_mass_fix:
                fix_servo_masses(folder, args.servo_mass / 1000, args.dry_run)
        except Exception as err:  # keep going so one bad export does not block the others
            failures += 1
            print(f"[error] {folder.name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
