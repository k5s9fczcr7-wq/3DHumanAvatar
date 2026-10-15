using UnrealBuildTool;

public class KioskAvatar : ModuleRules
{
    public KioskAvatar(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "WebSockets",              // AWebSocketManager
            "Json",                    // FJsonSerializer
            "JsonUtilities",           // convenience helpers
            "Sockets",                 // FSocket, UDP
            "Networking",              // FIPv4Endpoint, etc.
            "RuntimeMetaHumanLipSync", // Runtime MetaHuman Lip Sync plugin — mood enum + SetMood
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",
            "SlateCore",
            "ApplicationCore",         // FDisplayMetrics: where her eyes are on the monitor (eye gaze)
            "AnimGraphRuntime",        // FAnimNode_ModifyCurve (her resting smile)
            "AudioMixer",              // master submix effect: her voice delay (lip sync)
            "Synthesis",               // USubmixEffectDelayPreset
        });
    }
}
