using UnrealBuildTool;

public class KioskAvatarTarget : TargetRules
{
    public KioskAvatarTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V6;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("KioskAvatar");
    }
}
