#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "GestureLibrary.generated.h"

class UAnimSequenceBase;

/**
 * One gesture snippet and what it denotes.
 *
 * Snippets are cut from conversational mocap at moments close to her rest pose, with the hips and legs
 * held still, and stored as ADDITIVE animations (local space, relative to their own first frame), so
 * each one starts and ends at whatever pose her idle is in. Labels (intent, energy, tone) are assigned
 * when the library is curated.
 */
USTRUCT(BlueprintType)
struct KIOSKAVATAR_API FGestureClip
{
    GENERATED_BODY()

    /** Snippet id, e.g. "c11_03" (source clip + index). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    FName Id;

    /** The additive snippet on her body skeleton. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    TObjectPtr<UAnimSequenceBase> Anim = nullptr;

    /** What it denotes: agree, disagree, emphasize, explain, list, contrast, unsure, empathize,
     *  welcome, surprise, listen, talk, idle. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    FName Intent;

    /** Other intents it also fits (scored lower than the main one). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    TArray<FName> AlsoIntents;

    /** speaking, listening or idle (idle = only while nobody is at the kiosk). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    FName Role;

    /** 0 = low, 1 = medium, 2 = high. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture", meta = (ClampMin = "0", ClampMax = "2"))
    int32 Energy = 1;

    /** warm, neutral or serious. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    FName Tone;

    /** What moves: head, shoulders, lean, arm_L, arm_R, both_arms (the kiosk shot shows head + shoulders). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    FName Body;

    /** Strength (0-1) when the request carries no emotion intensity. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture", meta = (ClampMin = "0", ClampMax = "1"))
    float Strength = 0.7f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    bool bEnabled = true;
};

/** The curated set of gesture snippets her body anim instance chooses from. */
UCLASS(BlueprintType)
class KIOSKAVATAR_API UGestureLibrary : public UDataAsset
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Gesture")
    TArray<FGestureClip> Clips;
};
