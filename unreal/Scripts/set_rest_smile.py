"""Her resting expression: a faint smile added on top of everything else in ABP_AvatarFace. Her lip sync rewrites
the mouth curves even when she is silent, so the smile has to be added AFTER it, in Add mode, so her speech and mood
still move the mouth corners.
Editor Python cannot add AnimGraph nodes, so this calls the editor-only KioskAvatarEditor module
(UKioskAnimGraphTools::SetFinalCurveOffsets).

  set KIOSK_REST_SMILE=0.12        (0 removes the node again; default 0.12)
  UnrealEditor.exe unreal/KioskAvatar.uproject -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound

Log: unreal/Saved/Logs/rest_smile.log. Stop the stack and close the editor first; the face blueprint is saved.
"""
import os
import traceback

import unreal

ABP = "/Game/ABP_AvatarFace"            # your face Animation Blueprint: set this first
CURVES = ("CTRL_expressions_mouthCornerPullL", "CTRL_expressions_mouthCornerPullR")
LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "rest_smile.log")


def log(msg):
    unreal.log("RESTSMILE| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


open(LOG, "w").close()
try:
    amount = float(os.environ.get("KIOSK_REST_SMILE", "0.12"))
    abp = unreal.EditorAssetLibrary.load_asset(ABP)
    skeleton = abp.get_editor_property("target_skeleton")
    names = {str(n).lower(): str(n) for n in skeleton.get_curve_meta_data_names()}
    log(f"{skeleton.get_path_name()}: {len(names)} curves; mouth corner curves: "
        f"{sorted(v for k, v in names.items() if 'mouthcorner' in k)}")
    missing = [c for c in CURVES if c.lower() not in names]
    if missing:
        raise RuntimeError(f"curves missing on the face skeleton: {missing}")
    curves = unreal.Map(unreal.Name, float)
    if amount > 0:
        for c in CURVES:
            curves[unreal.Name(c)] = amount
    report = unreal.KioskAnimGraphTools.set_final_curve_offsets(abp, curves)
    log(f"rest smile {amount:.2f}: {report}")
    if report.startswith("ERROR"):
        raise RuntimeError(report)
    if report != "nothing to remove":
        log(f"saved: {unreal.EditorAssetLibrary.save_loaded_asset(abp, only_if_is_dirty=False)}")
except Exception:
    log("ERROR " + traceback.format_exc())
log("RESTSMILE_DONE")
