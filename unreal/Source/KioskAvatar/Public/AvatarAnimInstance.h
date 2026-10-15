#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimInstanceProxy.h"
#include "WebSocketManager.h"
#include "GestureLibrary.h"
#include "AvatarAnimInstance.generated.h"

struct FAnimNode_ModifyCurve;

class URealisticMetaHumanLipSyncGenerator;

/**
 * The body's animation proxy: after her graph is evaluated it turns her neck and head by the head-attention
 * correction that UEyeGazeComponent computes (a rotation in the body's component space), shared across
 * UAvatarAnimInstance::HeadCorrectionBones, so idles and gestures keep playing underneath it.
 */
struct FAvatarAnimInstanceProxy : public FAnimInstanceProxy
{
    FAvatarAnimInstanceProxy() = default;
    explicit FAvatarAnimInstanceProxy(UAnimInstance* InAnimInstance) : FAnimInstanceProxy(InAnimInstance) {}

    virtual void PreUpdate(UAnimInstance* InAnimInstance, float DeltaSeconds) override;
    virtual bool Evaluate_WithRoot(FPoseContext& Output, FAnimNode_Base* InRootNode) override;

private:
    void TurnHead(FCompactPose& Pose) const;

    FQuat HeadCorrection = FQuat::Identity;
    TArray<FName> Bones;
    TArray<float> Shares;
};

/**
 * UAvatarAnimInstance
 *
 * Animation Instance for the MetaHuman skeletal mesh.
 * Exposes state + emotion floats that drive the Animation Blueprint
 * state machine and additive emotion pose blends.
 *
 * Usage:
 *   1. Set this as the Anim Class on your MetaHuman's Body skeletal mesh.
 *   2. In your level Blueprint (or AWebSocketManager's event graph),
 *      call SetAvatarState() and SetAvatarEmotion() when delegates fire.
 *   3. In the Animation Blueprint graph, read the float properties below
 *      to drive state machine transitions and emotion pose blends.
 */
UCLASS(BlueprintType, Blueprintable)
class KIOSKAVATAR_API UAvatarAnimInstance : public UAnimInstance
{
    GENERATED_BODY()

public:
    UAvatarAnimInstance();

    // ── State machine variables (read in Anim BP transition rules) ────────────

    /** 0=Idle  1=Listening  2=Thinking  3=Talking */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar|State")
    EAvatarState AvatarState = EAvatarState::Idle;

    // Convenience booleans — easier to wire into transition rules than enum compare
    UPROPERTY(BlueprintReadOnly, Category = "Avatar|State")
    bool bIsIdle = true;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|State")
    bool bIsListening = false;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|State")
    bool bIsThinking = false;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|State")
    bool bIsTalking = false;

    // ── Emotion blend weights (0.0–1.0, used as additive pose blend alphas) ──

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|Emotion")
    float EmotionHappy = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|Emotion")
    float EmotionCurious = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|Emotion")
    float EmotionConcerned = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|Emotion")
    float EmotionAmused = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar|Emotion")
    float EmotionSad = 0.0f;

    /** Blend speed for emotion transitions (higher = snappier) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Emotion")
    float EmotionBlendSpeed = 3.0f;

    // ── Lip sync generator (set from ABP_AvatarFace BeginPlay) ───────────────

    /** Mood-enabled lip sync generator. Set this in ABP_AvatarFace BeginPlay
     *  right after "Create Realistic MetaHuman Lip Sync With Mood Generator". */
    UPROPERTY(BlueprintReadWrite, Category = "Avatar|LipSync")
    URealisticMetaHumanLipSyncGenerator* LipSyncGenerator = nullptr;

    // ── Called by AWebSocketManager delegates ────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "Avatar")
    void SetAvatarState(EAvatarState NewState);

    UFUNCTION(BlueprintCallable, Category = "Avatar")
    void SetAvatarEmotion(EAvatarEmotion NewEmotion);

    /** Apply an emotion to the lip sync generator's mood.
     *  Call this from BP_Avatar when OnEmotionChanged fires. */
    UFUNCTION(BlueprintCallable, Category = "Avatar|Emotion")
    void ApplyEmotionMood(EAvatarEmotion Emotion, float Intensity = 0.5f);

    // ── UAnimInstance interface ───────────────────────────────────────────────

    virtual void NativeUpdateAnimation(float DeltaSeconds) override;
    virtual void NativeInitializeAnimation() override;
    virtual void NativeUninitializeAnimation() override;

    // ── Gestures: additive snippets played into GestureSlot over her idle ───

    /** The gesture snippets to choose from. Loaded from /Game/Gestures/DA_GestureLibrary when left empty. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    TObjectPtr<UGestureLibrary> GestureLibrary = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    bool bGesturesEnabled = true;

    /** Slot node in ABP_AvatarController (just before Output Pose) that the snippets play into. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    FName GestureSlot = TEXT("DefaultSlot");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    float GestureBlendIn = 0.3f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    float GestureBlendOut = 0.4f;

    /** Multiplies every gesture's strength (1 = unchanged). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    float GestureStrengthScale = 1.0f;

    /** Random speed variation per play, +/- this fraction. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    float GestureSpeedJitter = 0.1f;

    /** A new gesture waits while the current one has more than this many seconds left (unless it interrupts). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Gesture")
    float GestureOverlapAllowance = 0.4f;

    /**
     * Resting smile: ABP_AvatarFace ends with a Modify Curve (Add) node that adds a faint smile
     * (unreal/Scripts/set_rest_smile.py). Every frame its Alpha follows 1 - EmotionSad - EmotionConcerned, so the
     * smile fades out while she is sad or concerned and never shows through bad news. False = leave it alone.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Face")
    bool bRestSmileFollowsEmotion = true;

    /**
     * Share of the resting smile kept while she talks (0 = none). Her lip sync carries her mood while she speaks;
     * a smile added on top pulls her mouth corners wide and makes her mouth look too open.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Face", meta = (ClampMin = "0", ClampMax = "1"))
    float RestSmileWhileTalking = 0.0f;

    /** Seconds the resting smile takes to fade when she starts talking, and to come back after. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Face", meta = (ClampMin = "0.01"))
    float RestSmileFadeTime = 0.25f;

    /** Neck and head bones that share the head-attention correction (root to tip), and their shares (sum 1). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Head")
    TArray<FName> HeadCorrectionBones = {TEXT("neck_01"), TEXT("neck_02"), TEXT("head")};

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avatar|Head")
    TArray<float> HeadCorrectionShares = {0.25f, 0.30f, 0.45f};

    /** Head attention (UEyeGazeComponent): a rotation in this mesh's component space that turns her neck and head
     *  after everything else this frame. Identity = none. */
    void SetHeadCorrection(const FQuat& Correction) { HeadCorrection = Correction; }
    const FQuat& GetHeadCorrection() const { return HeadCorrection; }

    /**
     * Play the snippet that best fits Intent, the emotion's energy and tone, and the role
     * (speaking / listening / idle), avoiding the last few used. Intensity (0-1, the emotion
     * intensity; < 0 = none) sets the strength, roughly 45-85%. Returns the snippet id, or None.
     */
    UFUNCTION(BlueprintCallable, Category = "Avatar|Gesture")
    FName PlayGesture(FName Intent, EAvatarEmotion Emotion, float Intensity = -1.0f, FName Role = NAME_None, bool bInterrupt = false);

    /** The body anim instance running in the game world (what the WebSocket manager drives). */
    static UAvatarAnimInstance* GetActive();

protected:
    virtual FAnimInstanceProxy* CreateAnimInstanceProxy() override;
    virtual void DestroyAnimInstanceProxy(FAnimInstanceProxy* InProxy) override;

private:
    FQuat HeadCorrection = FQuat::Identity;

    // While she talks the resting smile fades (TalkBlend 0..1); "talking" comes from the WebSocket manager's state.
    float TalkBlend = 0.0f;
    TWeakObjectPtr<AWebSocketManager> WSManager;
    double WSRetryAt = 0.0;

    // Target emotion weights — NativeUpdateAnimation lerps toward these
    float TargetHappy     = 0.0f;
    float TargetCurious   = 0.0f;
    float TargetConcerned = 0.0f;
    float TargetAmused    = 0.0f;
    float TargetSad       = 0.0f;

    // Gestures
    TArray<FName> RecentGestures;     // oldest first; the newest is penalised most
    FRandomStream GestureRandom;
    static TWeakObjectPtr<UAvatarAnimInstance> ActiveInstance;
    static void EmotionFit(EAvatarEmotion Emotion, float& OutEnergy, FName& OutTone);

    // Resting smile: the face ABP's Add-mode Modify Curve node, found once (again if the face re-initialises)
    TWeakObjectPtr<UAnimInstance> FaceAnimInstance;
    FAnimNode_ModifyCurve* RestSmileNode = nullptr;
    double RestSmileRetryAt = 0.0;
    void UpdateRestSmile();
};
