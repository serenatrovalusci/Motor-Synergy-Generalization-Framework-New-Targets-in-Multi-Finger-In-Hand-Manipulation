"""Run the Shadow Hand block environments with a different cube size."""
import importlib
import os
import re
import shutil
import sys

import gymnasium_robotics

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "assets")


def cube_model_path(env_id, size_cm):
    """Copy the block XML with the cube scaled; return its path.
    README convention: 2.5 = stock block = MuJoCo half-size 0.025."""
    assert "Block" in env_id, "cube size only works with the Block environments"
    half = size_cm / 100.0
    pkg = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets")
    os.makedirs(ASSETS, exist_ok=True)
    if not os.path.isdir(os.path.join(ASSETS, "hand")):
        shutil.copytree(os.path.join(pkg, "hand"), os.path.join(ASSETS, "hand"))
    for sub in ("stls", "textures"):
        src, link = os.path.join(pkg, sub), os.path.join(ASSETS, sub)
        assert os.path.isdir(src), f"{src} not found"
        if not os.path.lexists(link):
            os.symlink(src, link)
    base = ("manipulate_block_touch_sensors.xml" if "TouchSensors" in env_id
            else "manipulate_block.xml")
    text = open(os.path.join(pkg, "hand", base)).read()
    for name, h in (("object", half), ("object_hidden", half - 0.001), ("target", half)):
        pat = re.compile(r'(<geom name="%s" type="box" size=")[^"]*(")' % name)
        text, n = pat.subn(
            lambda m: m.group(1) + " ".join([f"{h:.4f}"] * 3) + m.group(2), text)
        assert n == 1, f"could not find geom {name} in {base}"
    path = os.path.join(ASSETS, "hand", base.replace(".xml", f"_cube{size_cm:g}.xml"))
    open(path, "w").write(text)
    return path


def patch_block_xml(xml_path, env_id):
    """Make the block env load our XML by replacing its module-level *_XML constant."""
    for mod in ("manipulate_block", "manipulate_block_touch_sensors"):
        importlib.import_module(f"gymnasium_robotics.envs.shadow_dexterous_hand.{mod}")
    base = ("manipulate_block_touch_sensors.xml" if "TouchSensors" in env_id
            else "manipulate_block.xml")
    patched, seen = 0, []
    for name, m in list(sys.modules.items()):
        if not name.startswith("gymnasium_robotics.envs.shadow_dexterous_hand"):
            continue
        for attr in dir(m):
            val = getattr(m, attr, None)
            if attr.endswith("XML") and isinstance(val, str):
                seen.append(f"{name}.{attr} = {val}")
                if os.path.basename(val) == base:
                    setattr(m, attr, xml_path)
                    patched += 1
    assert patched, "no XML constant found to patch. Candidates:\n" + "\n".join(seen)


def apply_cube_size(env_id, size_cm):
    path = cube_model_path(env_id, size_cm)
    patch_block_xml(path, env_id)
    print(f"cube size {size_cm} cm -> {path}")
    return path
