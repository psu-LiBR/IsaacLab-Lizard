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

Usage::

    python LizardFiles/rename_bladesnake_parts.py              # all assemblies in LizardFiles/BladeSnake
    python LizardFiles/rename_bladesnake_parts.py --dry-run    # only print the mapping
    python LizardFiles/rename_bladesnake_parts.py --undo       # restore the original Onshape names
    python LizardFiles/rename_bladesnake_parts.py "LizardFiles/BladeSnake/20mm Assembly"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "BladeSnake"
MAP_FILE = "rename_map.json"

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
    args = parser.parse_args()

    folders = find_assemblies(args.paths)
    if not folders:
        print(f"No folders containing robot.usda found in: {[str(p) for p in args.paths]}")
        return 1

    failures = 0
    for folder in folders:
        try:
            (undo_assembly if args.undo else rename_assembly)(folder, args.dry_run)
        except Exception as err:  # keep going so one bad export does not block the others
            failures += 1
            print(f"[error] {folder.name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
