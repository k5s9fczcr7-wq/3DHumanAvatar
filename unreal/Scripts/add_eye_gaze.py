"""Give BP_Avatar the eye gaze controller (UEyeGazeComponent, KioskAvatar C++).

The controller aims her eyes at the camera every frame (turned toward the visitor) and corrects for wherever her head
points, so she looks directly at the visitor and her gaze follows them whatever idle or gesture plays. While it is on
her actor, UEmotionMoodLibrary::ApplyGazeToFace does nothing. Safe to run twice.

  UnrealEditor.exe unreal/KioskAvatar.uproject -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound
  set KIOSK_GAZE_REMOVE=1 first to take the controller off again (the Blueprint's per-message gaze path then applies).

Close the editor first. Log: unreal/Saved/Logs/eye_gaze.log
"""
import os
import traceback

import unreal

LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "eye_gaze.log")
BP_PATH = "/Game/BP_Avatar"            # your MetaHuman blueprint: set this first
sdl = unreal.SubobjectDataBlueprintFunctionLibrary


def log(msg):
    unreal.log("EYEGAZE| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ours(handles):
    return [h for h in handles if isinstance(sdl.get_object(sdl.get_data(h)), unreal.EyeGazeComponent)]


open(LOG, "w").close()
try:
    bp = unreal.load_asset(BP_PATH)
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    handles = sds.k2_gather_subobject_data_for_blueprint(bp)
    log(f"{BP_PATH}: {len(handles)} subobjects: "
        + ", ".join(f"{sdl.get_object(sdl.get_data(h)).get_name()}" for h in handles if sdl.get_object(sdl.get_data(h))))
    have = ours(handles)
    if os.environ.get("KIOSK_GAZE_REMOVE") == "1":
        for h in have:
            sds.delete_subobject(handles[0], h, bp)
        log(f"removed {len(have)} EyeGazeComponent(s)")
    elif not have:
        params = unreal.AddNewSubobjectParams(parent_handle=handles[0], new_class=unreal.EyeGazeComponent,
                                              blueprint_context=bp)
        result = sds.add_new_subobject(params)
        log(f"add_new_subobject -> {result}")
    else:
        log("already has an EyeGazeComponent")
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    now = ours(sds.k2_gather_subobject_data_for_blueprint(bp))
    log(f"EyeGazeComponent on the avatar Blueprint: {len(now)}")
    log(f"saved: {unreal.EditorAssetLibrary.save_loaded_asset(bp, only_if_is_dirty=False)}")
except Exception:
    log("ERROR " + traceback.format_exc())
log("EYEGAZE_DONE")
