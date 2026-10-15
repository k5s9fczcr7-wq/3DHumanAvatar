#include "EmotionMoodLibrary.h"
#include "EyeGazeComponent.h"
#include "GameFramework/Actor.h"
#include "UObject/UnrealType.h"

ERealisticMetaHumanLipSyncMood UEmotionMoodLibrary::EmotionToLipSyncMood(EAvatarEmotion Emotion)
{
    switch (Emotion)
    {
        case EAvatarEmotion::Happy:     return ERealisticMetaHumanLipSyncMood::Happiness;
        case EAvatarEmotion::Sad:       return ERealisticMetaHumanLipSyncMood::Sadness;
        case EAvatarEmotion::Disgust:   return ERealisticMetaHumanLipSyncMood::Disgust;
        case EAvatarEmotion::Angry:     return ERealisticMetaHumanLipSyncMood::Anger;
        case EAvatarEmotion::Surprised: return ERealisticMetaHumanLipSyncMood::Surprise;
        case EAvatarEmotion::Fearful:   return ERealisticMetaHumanLipSyncMood::Fear;
        case EAvatarEmotion::Confident: return ERealisticMetaHumanLipSyncMood::Confidence;
        case EAvatarEmotion::Excited:   return ERealisticMetaHumanLipSyncMood::Excitement;
        case EAvatarEmotion::Bored:     return ERealisticMetaHumanLipSyncMood::Boredom;
        case EAvatarEmotion::Playful:   return ERealisticMetaHumanLipSyncMood::Playfulness;
        case EAvatarEmotion::Confused:  return ERealisticMetaHumanLipSyncMood::Confusion;
        default:                        return ERealisticMetaHumanLipSyncMood::Neutral;
    }
}

void UEmotionMoodLibrary::SetMoodFromEmotion(
    URealisticMetaHumanLipSyncGenerator* Generator,
    EAvatarEmotion Emotion,
    float Intensity)
{
    if (!Generator)
    {
        UE_LOG(LogTemp, Warning, TEXT("[EmotionMood] Generator is null — cannot set mood"));
        return;
    }

    const ERealisticMetaHumanLipSyncMood Mood = EmotionToLipSyncMood(Emotion);
    Generator->SetMood(Mood);
    Generator->SetMoodIntensity(FMath::Clamp(Intensity, 0.0f, 1.0f));

    UE_LOG(LogTemp, Log, TEXT("[EmotionMood] Set mood=%d intensity=%.2f"),
           static_cast<int32>(Mood), Intensity);
}

void UEmotionMoodLibrary::FeedSilenceToGenerator(
    URealisticMetaHumanLipSyncGenerator* Generator,
    int32 SampleRate)
{
    if (!Generator)
    {
        return;
    }

    // 20ms of silence at the given sample rate (e.g., 480 samples at 24kHz)
    const int32 NumSamples = SampleRate / 50;
    TArray<float> Silence;
    Silence.SetNumZeroed(NumSamples);

    Generator->ProcessAudioData(Silence, SampleRate, 1);
}

void UEmotionMoodLibrary::ApplyGazeToFace(
    USkeletalMeshComponent* FaceMesh,
    float Yaw,
    float Pitch,
    float MaxYaw,
    float MaxPitch)
{
    if (!FaceMesh)
    {
        return;
    }
    // UEyeGazeComponent aims the eyes every frame. This per-message path (from the Blueprint's OnGazeReceived)
    // would fight it, so it does nothing while the component is on her actor.
    if (FaceMesh->GetOwner() && FaceMesh->GetOwner()->FindComponentByClass<UEyeGazeComponent>())
    {
        return;
    }

    // Convert gaze yaw/pitch to 4 directional weights (0-1) for Modify Curve node.
    // These drive CTRL_expressions_eyeLook* curves, which the lip sync plugin
    // intentionally leaves alone.

    // Subtract neutral offset — at center position, Pitch≈21° and Yaw≈3° because the
    // camera is above her eyes. Without this, the eyes look far up.
    // The neutral values are calibrated so the center position produces the right gaze.
    const float NeutralPitch = -1.5f;
    const float NeutralYaw = -20.0f;
    const float AdjPitch = Pitch - NeutralPitch;
    const float AdjYaw = Yaw - NeutralYaw;

    // A large normalization range reduces sensitivity.
    // At 22" distance, a user moving 12" laterally ≈ 5° yaw change from the gaze calculator.
    // With a range of 170, that is 5/170 ≈ 0.03 curve weight = subtle eye movement.
    const float YawRange = 170.0f;
    const float PitchRange = 125.0f;
    const float NormYaw   = FMath::Clamp(FMath::Abs(AdjYaw)   / YawRange,   0.0f, 1.0f);
    const float NormPitch = FMath::Clamp(FMath::Abs(AdjPitch)  / PitchRange,  0.0f, 1.0f);

    // Positive adjusted yaw = user is to her left → she looks left
    // Curve names use screen convention: eyeLookLeft = eyes move to screen-left = her right
    // So swap: positive yaw (user to her left) → LookRight curve
    const float LookLeft  = AdjYaw < 0.0f ? NormYaw : 0.0f;
    const float LookRight = AdjYaw > 0.0f ? NormYaw : 0.0f;

    // Positive adjusted pitch = user is above neutral → she looks up
    const float LookUp   = AdjPitch > 0.0f ? NormPitch : 0.0f;
    const float LookDown = AdjPitch < 0.0f ? NormPitch : 0.0f;

    // Set 4 direction weights on the anim instance via reflection.
    // Read by Modify Curve node in ABP_AvatarFace Anim Graph.
    UAnimInstance* AnimInst = FaceMesh->GetAnimInstance();
    if (!AnimInst) return;

    UClass* AnimClass = AnimInst->GetClass();
    auto SetDouble = [&](FName PropName, double Value)
    {
        FProperty* Prop = AnimClass->FindPropertyByName(PropName);
        if (!Prop) return;
        void* ValPtr = Prop->ContainerPtrToValuePtr<void>(AnimInst);
        if (CastField<FDoubleProperty>(Prop))
            *static_cast<double*>(ValPtr) = Value;
        else if (CastField<FFloatProperty>(Prop))
            *static_cast<float*>(ValPtr) = static_cast<float>(Value);
    };

    // Fake convergence: add a small inward bias to both horizontal directions.
    // This makes both eyes point slightly inward, simulating a ~30" focal distance.
    const float Convergence = 0.06f;

    SetDouble(FName("GazeLookLeft"),  static_cast<double>(LookLeft + Convergence));
    SetDouble(FName("GazeLookRight"), static_cast<double>(LookRight + Convergence));
    SetDouble(FName("GazeLookUp"),    static_cast<double>(LookUp));
    SetDouble(FName("GazeLookDown"),  static_cast<double>(LookDown));
}
