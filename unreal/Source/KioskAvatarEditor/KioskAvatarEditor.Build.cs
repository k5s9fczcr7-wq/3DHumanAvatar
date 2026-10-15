using UnrealBuildTool;

// Editor-only tools (never packaged): AnimGraph edits that editor Python cannot make, callable from Python.
public class KioskAvatarEditor : ModuleRules
{
    public KioskAvatarEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "UnrealEd",          // FKismetEditorUtilities, FBlueprintEditorUtils
            "AnimGraph",         // UAnimGraphNode_ModifyCurve, UAnimGraphNode_Root
            "AnimGraphRuntime",  // FAnimNode_ModifyCurve
            "BlueprintGraph",
        });
    }
}
