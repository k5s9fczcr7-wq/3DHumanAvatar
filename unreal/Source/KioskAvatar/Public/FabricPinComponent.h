#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "FabricPinComponent.generated.h"

class USceneComponent;
class USkinnedMeshComponent;

/**
 * Keeps its owner (the lapel-pin indicator) seated on a garment instead of a single bone. Every frame the pin rides
 * on one triangle of the garment as the garment is actually skinned, so it moves with the fabric (breathing,
 * shoulders, gestures). At start of play it lands on the fabric the camera sees behind it. If the garment's mesh
 * can't be read on the CPU, the pin stays on the bone it was attached to.
 */
UCLASS(ClassGroup = (Kiosk), meta = (BlueprintSpawnableComponent))
class KIOSKAVATAR_API UFabricPinComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UFabricPinComponent();

    /** The garment to ride on: a skinned mesh component of the actor the pin hangs on (BP_Avatar's top garment). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Fabric Pin")
    FName FabricComponentName = TEXT("Clothing_Top");

    /** How much of the pin's half-thickness sinks into the fabric (0 = resting on top of it). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Fabric Pin", meta = (ClampMin = "0", ClampMax = "1"))
    float SinkFraction = 0.25f;

    /** Fabric seen further than this (cm) behind or in front of the pin is not trusted; the nearest front-facing spot is used. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Fabric Pin")
    float MaxSeatDistance = 30.0f;

protected:
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
    bool Bind();
    bool SurfaceFrame(FVector& OutPoint, FVector& OutNormal, FVector& OutTangent);
    void Release(const TCHAR* Why);

    TWeakObjectPtr<USkinnedMeshComponent> Fabric;
    TWeakObjectPtr<USceneComponent> OriginalParent;
    FName OriginalSocket;
    FTransform OriginalRelative;
    uint32 Tri[3] = {0, 0, 0};
    FVector3f Bary = FVector3f(1.0f / 3.0f);
    float NormalSign = 1.0f;
    float Offset = 0.0f;                 // pin centre above the fabric, cm
    TArray<FMatrix44f> RefToLocals;      // scratch, reused every frame
    bool bBound = false;
    bool bGaveUp = false;
    int32 BindAttempts = 0;
    double NextBindAt = 0.0;

    // Diagnostics for the first few minutes: how far the original bone attachment would be from the fabric spot.
    double DriftWindowStart = 0.0;
    float DriftMin = 0.0f, DriftMax = 0.0f, DriftSum = 0.0f;
    int32 DriftCount = 0, DriftWindows = 0;
};
