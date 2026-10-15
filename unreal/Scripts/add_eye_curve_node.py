"""Per-eye gaze: each of her eyes aims on its own at the visitor's eye centreline, so they converge on the visitor,
which makes her gaze more convincing.

ABP_AvatarFace's gaze variables (GazeLookLeft/Right/Up/Down) move both eyes together. This adds a Modify Curve node in
Add mode holding the eight per-eye CTRL_expressions_eyeLook{Left,Right,Up,Down}{L,R} curves, in front of the resting
smile node; UEyeGazeComponent (KioskAvatar C++) writes them every frame. Editor Python cannot add AnimGraph nodes, so
this calls the editor-only KioskAvatarEditor module (UKioskAnimGraphTools::SetEyeCurveNode). Safe to run twice.

  UnrealEditor.exe unreal/KioskAvatar.uproject -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound
  set KIOSK_EYE_NODE_REMOVE=1 first to take the node out again (both eyes then move together).

Log: unreal/Saved/Logs/eye_curve_node.log. Stop the stack and close the editor first; the face blueprint is saved.
"""
import os
import traceback

import unreal

ABP = "/Game/ABP_AvatarFace"            # your face Animation Blueprint: set this first
CURVES = [f"CTRL_expressions_eyeLook{d}{e}" for e in "LR" for d in ("Left", "Right", "Up", "Down")]
LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "eye_curve_node.log")


def log(msg):
    unreal.log("EYENODE| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


open(LOG, "w").close()
try:
    install = os.environ.get("KIOSK_EYE_NODE_REMOVE") != "1"
    abp = unreal.EditorAssetLibrary.load_asset(ABP)
    skeleton = abp.get_editor_property("target_skeleton")
    names = {str(n).lower(): str(n) for n in skeleton.get_curve_meta_data_names()}
    log(f"{skeleton.get_path_name()}: {len(names)} curves; eye look curves: "
        f"{sorted(v for k, v in names.items() if 'eyelook' in k)}")
    missing = [c for c in CURVES if c.lower() not in names]
    if missing:
        raise RuntimeError(f"curves missing on the face skeleton: {missing}")
    report = unreal.KioskAnimGraphTools.set_eye_curve_node(abp, install)
    log(f"{'install' if install else 'remove'}: {report}")
    if report.startswith("ERROR"):
        raise RuntimeError(report)
    if report not in ("nothing to remove",) and not report.startswith("already there"):
        log(f"saved: {unreal.EditorAssetLibrary.save_loaded_asset(abp, only_if_is_dirty=False)}")
except Exception:
    log("ERROR " + traceback.format_exc())
log("EYENODE_DONE")
