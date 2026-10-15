"""Build the realistic status pin's assets (a metal frame around the lens, so it looks like an actual object).
New assets only, so it can run while the kiosk is up:
  /Game/Indicators/SM_StatusPin       lathe-turned pin from make_status_pin_obj.py: slot 0 Lens, slot 1 Bezel
  /Game/Indicators/M_PinLens          opaque, lit, glossy; "Color" (same name the indicator MICs set) + "Glow"
  /Game/Indicators/M_PinBezel         metal; "MetalColor", "Roughness"
  /Game/Indicators/MI_PinBezel_<Silver|Gold|Gunmetal>
The plain sphere pin uses M_SimpleGlow instead (translucent + unlit: flat colour and jagged edges).

  python make_status_pin_obj.py <out.obj>; set KIOSK_PIN_OBJ=<out.obj>
  UnrealEditor.exe unreal/KioskAvatar.uproject -ExecutePythonScript=<this> -nullrhi -unattended -nosplash -nosound
Log: unreal/Saved/Logs/status_pin.log
"""
import os
import traceback

import unreal

LOG = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "Logs", "status_pin.log")
DIR = "/Game/Indicators"
eal, mel = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
BEZELS = {"Silver": ((0.91, 0.92, 0.92), 0.22), "Gold": ((1.00, 0.77, 0.34), 0.28), "Gunmetal": ((0.22, 0.23, 0.25), 0.32)}


def log(msg):
    unreal.log("STATUSPIN| " + msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def new_material(name):
    path = f"{DIR}/{name}"
    if eal.does_asset_exist(path):
        eal.delete_asset(path)
    m = tools.create_asset(name, DIR, unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_DEFAULT_LIT)
    return m


def expr(m, cls, x, y, **props):
    e = mel.create_material_expression(m, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


open(LOG, "w").close()
try:
    # Mesh (an OBJ with "g" groups imports as one mesh per group: remove any such leftovers)
    for stray in ("Lens", "Bezel"):
        if eal.does_asset_exist(f"{DIR}/{stray}"):
            eal.delete_asset(f"{DIR}/{stray}")
            log(f"deleted stray {DIR}/{stray}")
    obj = os.environ.get("KIOSK_PIN_OBJ", "")
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", obj)
    task.set_editor_property("destination_path", DIR)
    task.set_editor_property("destination_name", "SM_StatusPin")
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    if os.environ.get("KIOSK_PIN_SKIP_IMPORT") != "1":
        tools.import_asset_tasks([task])
    log(f"imported: {[str(p) for p in task.get_editor_property('imported_object_paths')]}")
    mesh = eal.load_asset(f"{DIR}/SM_StatusPin")
    # A 4 cm prop: no Nanite (its simplified stand-in cuts 8,544 triangles to 498, a visible polygon). Normals are
    # recomputed by Unreal (the OBJ's are not imported); the OBJ's single smoothing group keeps them smooth.
    ns = mesh.get_editor_property("nanite_settings")
    ns.set_editor_property("enabled", False)
    mesh.set_editor_property("nanite_settings", ns)
    log(f"nanite {mesh.get_editor_property('nanite_settings').get_editor_property('enabled')}, "
        f"lod0 {mesh.get_num_triangles(0)} tris / {mesh.get_num_vertices(0)} verts")
    b = mesh.get_bounds()
    log(f"bounds extent {b.box_extent}  slots {[str(s.get_editor_property('material_slot_name')) for s in mesh.get_editor_property('static_materials')]}")

    # Lens: the indicator colour, lit and glossy, glowing from inside
    lens = new_material("M_PinLens")
    color = expr(lens, unreal.MaterialExpressionVectorParameter, -700, 0, parameter_name="Color",
                 default_value=unreal.LinearColor(1, 1, 1, 1))
    glow = expr(lens, unreal.MaterialExpressionScalarParameter, -700, 250, parameter_name="Glow", default_value=1.0)
    emis = expr(lens, unreal.MaterialExpressionMultiply, -350, 150)
    mel.connect_material_expressions(color, "", emis, "A")
    mel.connect_material_expressions(glow, "", emis, "B")
    base = expr(lens, unreal.MaterialExpressionMultiply, -350, -100, const_b=0.25)
    mel.connect_material_expressions(color, "", base, "A")
    mel.connect_material_property(base, "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(emis, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.connect_material_property(expr(lens, unreal.MaterialExpressionConstant, -350, 350, r=0.10), "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(expr(lens, unreal.MaterialExpressionConstant, -350, 450, r=0.8), "", unreal.MaterialProperty.MP_SPECULAR)
    mel.recompile_material(lens)

    # Bezel: plain metal
    bez = new_material("M_PinBezel")
    mcol = expr(bez, unreal.MaterialExpressionVectorParameter, -500, 0, parameter_name="MetalColor",
                default_value=unreal.LinearColor(0.91, 0.92, 0.92, 1))
    rough = expr(bez, unreal.MaterialExpressionScalarParameter, -500, 250, parameter_name="Roughness", default_value=0.25)
    mel.connect_material_property(mcol, "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(expr(bez, unreal.MaterialExpressionConstant, -500, 400, r=1.0), "", unreal.MaterialProperty.MP_METALLIC)
    mel.recompile_material(bez)
    for name, (rgb, r) in BEZELS.items():
        path = f"{DIR}/MI_PinBezel_{name}"
        if eal.does_asset_exist(path):
            eal.delete_asset(path)
        mi = tools.create_asset(f"MI_PinBezel_{name}", DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(mi, bez)
        mel.set_material_instance_vector_parameter_value(mi, "MetalColor", unreal.LinearColor(*rgb, 1))
        mel.set_material_instance_scalar_parameter_value(mi, "Roughness", r)
        eal.save_loaded_asset(mi)

    # Default materials on the mesh: lens + silver (by slot name; the indicator Blueprint recolours slot 0)
    for i, s in enumerate(mesh.get_editor_property("static_materials")):
        is_lens = "lens" in str(s.get_editor_property("material_slot_name")).lower()
        mesh.set_material(i, lens if is_lens else eal.load_asset(f"{DIR}/MI_PinBezel_Silver"))
    for a in (lens, bez, mesh):
        eal.save_loaded_asset(a, only_if_is_dirty=False)
    log("materials: " + ", ".join(f"{i}:{str(s.get_editor_property('material_slot_name'))}="
                                  f"{s.get_editor_property('material_interface').get_name()}" for i, s in enumerate(mesh.get_editor_property('static_materials'))))
    left = [str(a.asset_name) for a in unreal.AssetRegistryHelpers.get_asset_registry().get_assets_by_path(DIR)]
    log("in /Game/Indicators: " + ", ".join(sorted(left)))
except Exception:
    log("ERROR " + traceback.format_exc())
log("STATUSPIN_DONE")
