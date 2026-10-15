"""Put the lapel-pin indicator on her top garment instead of a spine bone. Adds a FabricPinComponent (KioskAvatar C++)
to the LapelPin_Indicator actor in MainLevel. At play time it seats the pin on the garment (BP_Avatar's Clothing_Top
component) and moves it with the fabric every frame. Safe to run twice.

  UnrealEditor.exe unreal/KioskAvatar.uproject /Game/MainLevel -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound
  set KIOSK_PIN_ON_BONE=1 first to remove the component again (the pin then hangs on the spine_05 bone).

Log: unreal/Saved/Logs/fabric_pin.log. Close the editor first; the level is saved.
"""
import os
import traceback

import unreal

LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "fabric_pin.log")
sdl = unreal.SubobjectDataBlueprintFunctionLibrary


def log(msg):
    unreal.log("FABRICPIN| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


open(LOG, "w").close()
try:
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    pins = [a for a in eas.get_all_level_actors() if a.get_actor_label() == "LapelPin_Indicator"]
    if len(pins) != 1:
        raise RuntimeError(f"expected one LapelPin_Indicator in the open level, found {len(pins)}")
    pin = pins[0]
    root = pin.root_component
    log(f"pin attached to {root.get_attach_parent().get_name() if root.get_attach_parent() else None} "
        f"socket {root.get_attach_socket_name()} mobility {root.get_editor_property('mobility')}")
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    handles = sds.k2_gather_subobject_data_for_instance(pin)
    ours = [h for h in handles if isinstance(sdl.get_object(sdl.get_data(h)), unreal.FabricPinComponent)]
    if os.environ.get("KIOSK_PIN_ON_BONE") == "1":
        for h in ours:
            sds.delete_subobject(handles[0], h, None)
        log(f"removed {len(ours)} FabricPinComponent(s)")
    elif not ours:
        params = unreal.AddNewSubobjectParams(parent_handle=handles[0], new_class=unreal.FabricPinComponent,
                                              blueprint_context=None)
        result = sds.add_new_subobject(params)
        log(f"add_new_subobject -> {result}")
    else:
        log("already has a FabricPinComponent")
    have = pin.get_components_by_class(unreal.FabricPinComponent)
    log(f"FabricPinComponent on the pin: {len(have)}"
        + (f"; rides on {have[0].get_editor_property('fabric_component_name')}" if have else ""))
    log(f"saved: {unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)}")
except Exception:
    log("ERROR " + traceback.format_exc())
log("FABRICPIN_DONE")
