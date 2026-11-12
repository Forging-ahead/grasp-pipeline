"""Save reset-time fixture transforms, including edits outside joint qpos.

The compiled binary embeds assets and preserves the model for the recorded
MuJoCo version. XML remains readable; its decimals can be rounded on export.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def save_reset_model(sim, directory):
    import mujoco
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = getattr(sim.model, '_model', sim.model)
    xml, binary, parameters = [directory / name for name in
        ('merged_model.xml', 'reset_model.mjb', 'MODEL_PARAMETERS.npz')]
    mujoco.mj_saveLastXML(str(xml), model)
    mujoco.mj_saveModel(model, str(binary))
    fields = ('body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'geom_size',
              'site_pos', 'site_quat', 'jnt_pos', 'jnt_axis')
    np.savez_compressed(parameters, **{name: np.array(getattr(model, name), copy=True) for name in fields})
    record = dict(phase='after_official_reset_and_settle', mujoco_version=mujoco.__version__,
        binary_replay_requires_same_mujoco_version=True, binary_assets_embedded=True,
        xml_decimal_rounding=True, live_physics_modified=False,
        files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in (xml, binary, parameters)})
    (directory/'MODEL_SNAPSHOT.json').write_text(json.dumps(record, indent=2)+'\n')
    return record
