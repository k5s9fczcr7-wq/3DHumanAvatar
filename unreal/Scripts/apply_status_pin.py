"""Put the realistic status pin on the kiosk, after build_status_pin.py (silver bezel by default).
- The six indicator colours (/Game/Indicators/MI_Indicator_*) move from M_SimpleGlow (translucent, unlit: flat and
  jagged) to M_PinLens; it has the same "Color" parameter, so BP_WebSocketManager's recolouring keeps working.
- LapelPin_Indicator in MainLevel gets SM_StatusPin at scale 1 (slot 0 lens = the colour, slot 1 = the bezel).
  Its FabricPinComponent reads the pin's thickness from the mesh bounds.

  set KIOSK_PIN_BEZEL=Silver        (or Gold, Gunmetal)
  set KIOSK_PIN_RESTORE=1           to switch back to the plain sphere pin and M_SimpleGlow
  MSYS_NO_PATHCONV=1 UnrealEditor.exe unreal/KioskAvatar.uproject /Game/MainLevel -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound

Close the editor first. Log: unreal/Saved/Logs/apply_status_pin.log
"""
import os
import traceback

import unreal

LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "apply_status_pin.log")
DIR = "/Game/Indicators"
STATES = ("Grey", "White", "Green", "Amber", "Red", "Purple")
eal, mel = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary


def log(msg):
    unreal.log("APPLYPIN| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


open(LOG, "w").close()
try:
    restore = os.environ.get("KIOSK_PIN_RESTORE") == "1"
    parent = eal.load_asset("/Game/LevelPrototyping/Interactable/JumpPad/Assets/Materials/M_SimpleGlow" if restore
                            else f"{DIR}/M_PinLens")
    for state in STATES:
        mi = eal.load_asset(f"{DIR}/MI_Indicator_{state}")
        mel.set_material_instance_parent(mi, parent)
        eal.save_loaded_asset(mi, only_if_is_dirty=False)
    log(f"MI_Indicator_* now on {parent.get_name()}")

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    pins = [a for a in actors if a.get_actor_label() == "LapelPin_Indicator"]
    if len(pins) != 1:
        raise RuntimeError(f"expected one LapelPin_Indicator in the open level (pass /Game/MainLevel), found {len(pins)}")
    pin = pins[0]
    smc = pin.static_mesh_component
    # Setting a mesh or scale from Python doesn't mark the level as changed, so it would never be saved.
    pin.modify()
    smc.modify()
    if restore:
        smc.set_static_mesh(eal.load_asset("/Engine/BasicShapes/Sphere"))
        pin.set_actor_scale3d(unreal.Vector(0.04, 0.04, 0.02))
        smc.set_material(0, eal.load_asset(f"{DIR}/MI_Indicator_Grey"))
    else:
        bezel = os.environ.get("KIOSK_PIN_BEZEL", "Silver")
        smc.set_static_mesh(eal.load_asset(f"{DIR}/SM_StatusPin"))
        pin.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
        smc.set_material(0, eal.load_asset(f"{DIR}/MI_Indicator_Grey"))
        smc.set_material(1, eal.load_asset(f"{DIR}/MI_PinBezel_{bezel}"))
    log(f"LapelPin_Indicator: mesh {smc.static_mesh.get_name()}, scale {pin.get_actor_scale3d()}, "
        f"materials {[m.get_name() if m else None for m in smc.get_materials()]}")
    log(f"level saved: {unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()}, "
        f"others: {unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)}")
except Exception:
    log("ERROR " + traceback.format_exc())
log("APPLYPIN_DONE")
