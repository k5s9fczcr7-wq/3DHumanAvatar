#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Components/SkeletalMeshComponent.h"
#include "WebSocketManager.h"
#include "RealisticMetaHumanLipSyncGenerator.h"
#include "EmotionMoodLibrary.generated.h"

/**
 * UEmotionMoodLibrary
 *
 * Blueprint Function Library that maps EAvatarEmotion (from WebSocket messages)
 * to ERealisticMetaHumanLipSyncMood (Runtime MetaHuman Lip Sync plugin) and calls SetMood/SetMoodIntensity.
 *
 * Usage in Blueprint:
 *   OnEmotionChanged(Emotion, Intensity) → SetMoodFromEmotion(Generator, Emotion, Intensity)
 */
UCLASS()
class KIOSKAVATAR_API UEmotionMoodLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Convert EAvatarEmotion to the lip sync plugin's mood enum.
     *
     * @param Emotion The avatar emotion from the WebSocket message
     * @return The corresponding lip sync mood
     */
    UFUNCTION(BlueprintPure, Category = "Avatar|Emotion",
              meta = (DisplayName = "Emotion to Lip Sync Mood"))
    static ERealisticMetaHumanLipSyncMood EmotionToLipSyncMood(EAvatarEmotion Emotion);

    /**
     * One-call convenience: set mood AND intensity on a lip sync generator from an emotion event.
     *
     * @param Generator The lip sync generator (must be mood-enabled)
     * @param Emotion   The avatar emotion from the WebSocket message
     * @param Intensity Emotion intensity (0.0-1.0, default 0.5)
     */
    UFUNCTION(BlueprintCallable, Category = "Avatar|Emotion",
              meta = (DisplayName = "Set Mood From Emotion"))
    static void SetMoodFromEmotion(
        URealisticMetaHumanLipSyncGenerator* Generator,
        EAvatarEmotion Emotion,
        float Intensity = 0.5f);

    /**
     * Feed a short buffer of silence to the generator so it produces mood-tinted
     * resting expressions (brows, cheeks, eyes) while keeping the mouth closed.
     * Call this on anim tick when no speech audio is playing.
     *
     * @param Generator The lip sync generator
     * @param SampleRate Sample rate to match TTS output (default 24000)
     */
    UFUNCTION(BlueprintCallable, Category = "Avatar|Emotion",
              meta = (DisplayName = "Feed Silence To Generator"))
    static void FeedSilenceToGenerator(
        URealisticMetaHumanLipSyncGenerator* Generator,
        int32 SampleRate = 24000);

    /**
     * Apply gaze yaw/pitch to a MetaHuman face mesh via ARKit eye morph targets.
     * Converts signed degree angles into 0-1 blendshape weights for all 8 eye
     * look directions (eyeLookUpLeft, eyeLookDownRight, etc.).
     *
     * @param FaceMesh  The Face skeletal mesh component from BP_Avatar
     * @param Yaw       Horizontal gaze in degrees (positive = the avatar looks left)
     * @param Pitch     Vertical gaze in degrees (positive = the avatar looks up)
     * @param MaxYaw    Maximum yaw range for normalization (default 35)
     * @param MaxPitch  Maximum pitch range for normalization (default 25)
     */
    UFUNCTION(BlueprintCallable, Category = "Avatar|Gaze",
              meta = (DisplayName = "Apply Gaze To Face"))
    static void ApplyGazeToFace(
        USkeletalMeshComponent* FaceMesh,
        float Yaw,
        float Pitch,
        float MaxYaw = 35.0f,
        float MaxPitch = 25.0f);
};
