#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "KioskAnimGraphTools.generated.h"

class UAnimBlueprint;

/**
 * AnimGraph edits that editor Python cannot make (it has no way to create graph nodes), so they can run headless:
 *   UnrealEditor.exe <uproject> -ExecutePythonScript=<script> -nullrhi -unattended -nosplash -nosound
 * From Python: unreal.KioskAnimGraphTools.set_final_curve_offsets(abp, {"CTRL_expressions_mouthCornerPullL": 0.1})
 */
UCLASS()
class KIOSKAVATAREDITOR_API UKioskAnimGraphTools : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Keeps one Modify Curve node in Add mode directly in front of the AnimGraph's Output Pose, holding exactly these
     * curve offsets, so they are added on top of everything upstream (her lip sync included). The first call inserts
     * the node, later calls replace its values, and an empty map removes it and reconnects the graph as it was.
     * Compiles the blueprint and marks it dirty; the caller saves it. The report starts with "ERROR" on failure.
     */
    UFUNCTION(BlueprintCallable, Category = "Kiosk|Editor")
    static FString SetFinalCurveOffsets(UAnimBlueprint* AnimBlueprint, const TMap<FName, float>& Curves);

    /**
     * Per-eye gaze: keeps one Modify Curve node in Add mode holding the eight
     * CTRL_expressions_eyeLook{Left,Right,Up,Down}{L,R} curves at 0, just upstream of the final offsets node (the
     * resting smile) or of Output Pose if there is none. UEyeGazeComponent finds it at run time and writes each
     * eye's own values every frame, so both eyes converge on the visitor. bInstall = false removes it. Compiles the
     * blueprint and marks it dirty; the caller saves it. The report starts with "ERROR" on failure.
     */
    UFUNCTION(BlueprintCallable, Category = "Kiosk|Editor")
    static FString SetEyeCurveNode(UAnimBlueprint* AnimBlueprint, bool bInstall);
};
