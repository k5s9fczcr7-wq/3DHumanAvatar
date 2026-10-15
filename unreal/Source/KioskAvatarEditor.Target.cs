using UnrealBuildTool;

public class KioskAvatarEditorTarget : TargetRules
{
    public KioskAvatarEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V6;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("KioskAvatar");
        ExtraModuleNames.Add("KioskAvatarEditor");   // editor-only tools, never packaged
    }
}
