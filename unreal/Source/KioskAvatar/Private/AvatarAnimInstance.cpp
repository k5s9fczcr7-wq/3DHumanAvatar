#include "AvatarAnimInstance.h"
#include "EmotionMoodLibrary.h"
#include "Math/UnrealMathUtility.h"
#include "Animation/AnimMontage.h"
#include "Components/SkeletalMeshComponent.h"
#include "AnimNodes/AnimNode_ModifyCurve.h"
#include "EngineUtils.h"
#include "HAL/PlatformTime.h"

TWeakObjectPtr<UAvatarAnimInstance> UAvatarAnimInstance::ActiveInstance;

UAvatarAnimInstance::UAvatarAnimInstance()
{
}

// ── Head attention: the body's proxy turns her neck and head after her graph ──────────────────────────────────

FAnimInstanceProxy* UAvatarAnimInstance::CreateAnimInstanceProxy()
{
    return new FAvatarAnimInstanceProxy(this);
}

void UAvatarAnimInstance::DestroyAnimInstanceProxy(FAnimInstanceProxy* InProxy)
{
    delete static_cast<FAvatarAnimInstanceProxy*>(InProxy);
}

void FAvatarAnimInstanceProxy::PreUpdate(UAnimInstance* InAnimInstance, float DeltaSeconds)
{
    FAnimInstanceProxy::PreUpdate(InAnimInstance, DeltaSeconds);
    if (const UAvatarAnimInstance* Avatar = Cast<UAvatarAnimInstance>(InAnimInstance))
    {
        HeadCorrection = Avatar->GetHeadCorrection();
        if (Bones != Avatar->HeadCorrectionBones)
        {
            Bones = Avatar->HeadCorrectionBones;
        }
        if (Shares != Avatar->HeadCorrectionShares)
        {
            Shares = Avatar->HeadCorrectionShares;
        }
    }
}

bool FAvatarAnimInstanceProxy::Evaluate_WithRoot(FPoseContext& Output, FAnimNode_Base* InRootNode)
{
    EvaluateAnimationNode_WithRoot(Output, InRootNode);      // her whole graph, as the base proxy does
    if (InRootNode == GetRootNode() && !HeadCorrection.Equals(FQuat::Identity, 1.e-6f))
    {
        TurnHead(Output.Pose);
    }
    return true;
}

void FAvatarAnimInstanceProxy::TurnHead(FCompactPose& Pose) const
{
    const FBoneContainer& Required = Pose.GetBoneContainer();
    // A bone's rotation in component space, from the local pose (parents first: parent * local).
    auto ComponentRotation = [&Pose](FCompactPoseBoneIndex Index)
    {
        FQuat Q = FQuat::Identity;
        while (Index.IsValid())
        {
            Q = Pose[Index].GetRotation() * Q;
            Index = Pose.GetParentBoneIndex(Index);
        }
        return Q;
    };
    // Each bone turns by its share of the correction about the same component-space axis, at its own joint, so the
    // shares add up to the whole correction at her head. Root to tip: a child sees its parents already turned.
    for (int32 I = 0; I < Bones.Num(); ++I)
    {
        const float Share = Shares.IsValidIndex(I) ? Shares[I] : 0.0f;
        const int32 MeshIndex = Required.GetPoseBoneIndexForBoneName(Bones[I]);
        if (Share <= 0.0f || MeshIndex == INDEX_NONE)
        {
            continue;
        }
        const FCompactPoseBoneIndex Bone = Required.MakeCompactPoseIndex(FMeshPoseBoneIndex(MeshIndex));
        if (!Bone.IsValid())
        {
            continue;
        }
        const FQuat Delta = FQuat::Slerp(FQuat::Identity, HeadCorrection, Share);
        const FCompactPoseBoneIndex Parent = Pose.GetParentBoneIndex(Bone);
        const FQuat ParentCS = Parent.IsValid() ? ComponentRotation(Parent) : FQuat::Identity;
        FTransform& Local = Pose[Bone];
        Local.SetRotation((ParentCS.Inverse() * Delta * ParentCS * Local.GetRotation()).GetNormalized());
    }
}

// ── Registration: the WebSocket manager drives the body instance in the game world ──

void UAvatarAnimInstance::NativeInitializeAnimation()
{
    Super::NativeInitializeAnimation();
    GestureRandom.GenerateNewSeed();
    const USkeletalMeshComponent* Comp = GetOwningComponent();
    const UWorld* World = GetWorld();
    // Editor previews also run anim instances; only her Body in a game (or PIE) world counts.
    if (Comp && World && World->IsGameWorld() && Comp->GetName() == TEXT("Body"))
    {
        ActiveInstance = this;
        UE_LOG(LogTemp, Log, TEXT("[Gesture] body anim instance registered (%s)"), *GetPathName());
    }
}

void UAvatarAnimInstance::NativeUninitializeAnimation()
{
    if (ActiveInstance.Get() == this)
    {
        ActiveInstance.Reset();
    }
    Super::NativeUninitializeAnimation();
}

UAvatarAnimInstance* UAvatarAnimInstance::GetActive()
{
    return ActiveInstance.Get();
}

// ── Gestures ──────────────────────────────────────────────────────────────────

void UAvatarAnimInstance::EmotionFit(EAvatarEmotion Emotion, float& OutEnergy, FName& OutTone)
{
    // Energy 0 = low .. 2 = high; tone warm / neutral / serious (the gesture library's labels).
    switch (Emotion)
    {
        case EAvatarEmotion::Happy:     OutEnergy = 1.0f; OutTone = TEXT("warm");    break;
        case EAvatarEmotion::Excited:   OutEnergy = 2.0f; OutTone = TEXT("warm");    break;
        case EAvatarEmotion::Playful:   OutEnergy = 1.5f; OutTone = TEXT("warm");    break;
        case EAvatarEmotion::Confident: OutEnergy = 1.0f; OutTone = TEXT("neutral"); break;
        case EAvatarEmotion::Surprised: OutEnergy = 2.0f; OutTone = TEXT("neutral"); break;
        case EAvatarEmotion::Confused:  OutEnergy = 1.0f; OutTone = TEXT("neutral"); break;
        case EAvatarEmotion::Sad:       OutEnergy = 0.0f; OutTone = TEXT("warm");    break;
        case EAvatarEmotion::Fearful:   OutEnergy = 0.0f; OutTone = TEXT("serious"); break;
        case EAvatarEmotion::Bored:     OutEnergy = 0.0f; OutTone = TEXT("neutral"); break;
        case EAvatarEmotion::Angry:     OutEnergy = 1.0f; OutTone = TEXT("serious"); break;
        case EAvatarEmotion::Disgust:   OutEnergy = 1.0f; OutTone = TEXT("serious"); break;
        default:                        OutEnergy = 0.5f; OutTone = TEXT("neutral"); break;   // Neutral
    }
}

// Intents with no snippet of their own (or whose snippets are all disabled) borrow the nearest one:
// a greeting gets a warm nod, a list or plain talk an explaining hand. No cycles; at most two hops.
static FName FallbackIntent(FName Intent)
{
    static const TCHAR* const Fallbacks[][2] = {
        { TEXT("welcome"),     TEXT("agree") },
        { TEXT("surprise"),    TEXT("emphasize") },
        { TEXT("list"),        TEXT("explain") },
        { TEXT("talk"),        TEXT("explain") },
        { TEXT("contrast"),    TEXT("explain") },
        { TEXT("emphasize"),   TEXT("explain") },
        { TEXT("thinking"),    TEXT("unsure") },
        { TEXT("disagree"),    TEXT("unsure") },
        { TEXT("agree"),       TEXT("listen") },
        { TEXT("empathize"),   TEXT("listen") },
        { TEXT("look_around"), TEXT("idle") },
    };
    for (const auto& Pair : Fallbacks)
    {
        if (Intent == FName(Pair[0]))
        {
            return FName(Pair[1]);
        }
    }
    return NAME_None;
}

FName UAvatarAnimInstance::PlayGesture(FName Intent, EAvatarEmotion Emotion, float Intensity, FName Role, bool bInterrupt)
{
    if (!bGesturesEnabled || Intent.IsNone())
    {
        return NAME_None;
    }
    if (!GestureLibrary)
    {
        GestureLibrary = LoadObject<UGestureLibrary>(nullptr, TEXT("/Game/Gestures/DA_GestureLibrary.DA_GestureLibrary"));
        if (!GestureLibrary)
        {
            UE_LOG(LogTemp, Warning, TEXT("[Gesture] no gesture library (/Game/Gestures/DA_GestureLibrary) - gesture %s skipped"), *Intent.ToString());
            bGesturesEnabled = false;   // don't retry the load on every message
            return NAME_None;
        }
    }

    // Let the current gesture finish unless this one interrupts it.
    if (!bInterrupt)
    {
        if (const FAnimMontageInstance* Current = GetActiveMontageInstance())
        {
            if (Current->IsPlaying() && Current->Montage)
            {
                const float Rate = FMath::Max(FMath::Abs(Current->GetPlayRate()), 0.01f);
                const float Left = (Current->Montage->GetPlayLength() - Current->GetPosition()) / Rate;
                if (Left > GestureOverlapAllowance)
                {
                    UE_LOG(LogTemp, Verbose, TEXT("[Gesture] %s skipped: %.2f s of the current gesture left"), *Intent.ToString(), Left);
                    return NAME_None;
                }
            }
        }
    }

    float WantEnergy = 0.5f;
    FName WantTone;
    EmotionFit(Emotion, WantEnergy, WantTone);
    const FName IdleName(TEXT("idle"));
    const bool bIdleWanted = (Role == IdleName) || (Intent == IdleName);

    // The best snippet for one intent, or nullptr if no enabled snippet has it.
    auto Pick = [&](FName Want) -> const FGestureClip*
    {
        const FGestureClip* Found = nullptr;
        float FoundScore = -1.0e9f;
        for (const FGestureClip& Clip : GestureLibrary->Clips)
        {
            if (!Clip.bEnabled || !Clip.Anim || ((Clip.Role == IdleName) != bIdleWanted))
            {
                continue;
            }
            float Score;
            if (Clip.Intent == Want)                { Score = 3.0f; }
            else if (Clip.AlsoIntents.Contains(Want)) { Score = 1.5f; }
            else                                     { continue; }
            if (!Role.IsNone() && Clip.Role == Role) { Score += 1.0f; }
            Score -= 0.75f * FMath::Abs(static_cast<float>(Clip.Energy) - WantEnergy);
            if (Clip.Tone != WantTone)               { Score -= 0.5f; }
            const int32 Recent = RecentGestures.Find(Clip.Id);
            if (Recent != INDEX_NONE)
            {
                Score -= 2.0f * static_cast<float>(Recent + 1) / static_cast<float>(RecentGestures.Num());
            }
            Score += GestureRandom.FRandRange(0.0f, 0.6f);
            if (Score > FoundScore)
            {
                FoundScore = Score;
                Found = &Clip;
            }
        }
        return Found;
    };

    FName Used = Intent;
    const FGestureClip* Best = Pick(Used);
    for (int32 Hop = 0; !Best && Hop < 2; ++Hop)
    {
        Used = FallbackIntent(Used);
        if (Used.IsNone())
        {
            break;
        }
        Best = Pick(Used);
    }
    if (!Best)
    {
        UE_LOG(LogTemp, Verbose, TEXT("[Gesture] no snippet for %s"), *Intent.ToString());
        return NAME_None;
    }

    const float Base = (Intensity >= 0.0f) ? FMath::Clamp(0.4f + 0.6f * Intensity, 0.45f, 0.85f) : Best->Strength;
    const float Strength = FMath::Clamp(Base * GestureStrengthScale, 0.0f, 1.0f);
    const float PlayRate = 1.0f + GestureRandom.FRandRange(-GestureSpeedJitter, GestureSpeedJitter);
    UAnimMontage* Montage = PlaySlotAnimationAsDynamicMontage(Best->Anim, GestureSlot, GestureBlendIn, GestureBlendOut, PlayRate, 1);
    if (!Montage)
    {
        return NAME_None;
    }
    if (FAnimMontageInstance* Instance = GetActiveInstanceForMontage(Montage))
    {
        Instance->SetDesiredWeight(Strength);   // additive weight = gesture strength
    }
    RecentGestures.Remove(Best->Id);
    RecentGestures.Add(Best->Id);
    while (RecentGestures.Num() > 4)
    {
        RecentGestures.RemoveAt(0);
    }
    const FString Via = (Used == Intent) ? FString() : FString::Printf(TEXT(" via %s"), *Used.ToString());
    UE_LOG(LogTemp, Log, TEXT("[Gesture] %s (%s) -> %s%s, strength %.2f, rate %.2f"),
           *Intent.ToString(), *Role.ToString(), *Best->Id.ToString(), *Via, Strength, PlayRate);
    return Best->Id;
}

// ── State ─────────────────────────────────────────────────────────────────────

void UAvatarAnimInstance::SetAvatarState(EAvatarState NewState)
{
    AvatarState  = NewState;
    bIsIdle      = (NewState == EAvatarState::Idle);
    bIsListening = (NewState == EAvatarState::Listening);
    bIsThinking  = (NewState == EAvatarState::Thinking);
    bIsTalking   = (NewState == EAvatarState::Talking);
}

// ── Emotion ───────────────────────────────────────────────────────────────────

void UAvatarAnimInstance::SetAvatarEmotion(EAvatarEmotion NewEmotion)
{
    // Clear all targets first, then set the active one to 1.0.
    // Only 5 blend weight channels exist — the other emotions map to the closest channel.
    TargetHappy     = 0.0f;
    TargetCurious   = 0.0f;
    TargetConcerned = 0.0f;
    TargetAmused    = 0.0f;
    TargetSad       = 0.0f;

    switch (NewEmotion)
    {
        case EAvatarEmotion::Happy:     TargetHappy     = 1.0f; break;
        case EAvatarEmotion::Sad:       TargetSad       = 1.0f; break;
        case EAvatarEmotion::Disgust:   TargetConcerned = 1.0f; break;
        case EAvatarEmotion::Angry:     TargetConcerned = 1.0f; break;
        case EAvatarEmotion::Surprised: TargetCurious   = 1.0f; break;
        case EAvatarEmotion::Fearful:   TargetConcerned = 1.0f; break;
        case EAvatarEmotion::Confident: TargetHappy     = 0.6f; break;
        case EAvatarEmotion::Excited:   TargetHappy     = 0.8f; TargetCurious = 0.5f; break;
        case EAvatarEmotion::Bored:     TargetSad       = 0.4f; break;
        case EAvatarEmotion::Playful:   TargetAmused    = 1.0f; break;
        case EAvatarEmotion::Confused:  TargetCurious   = 0.8f; break;
        default: break; // Neutral — all targets already 0
    }
}

// ── Lip sync mood ────────────────────────────────────────────────────────────

void UAvatarAnimInstance::ApplyEmotionMood(EAvatarEmotion Emotion, float Intensity)
{
    UEmotionMoodLibrary::SetMoodFromEmotion(LipSyncGenerator, Emotion, Intensity);
}

// ── Per-frame update ──────────────────────────────────────────────────────────

void UAvatarAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
    Super::NativeUpdateAnimation(DeltaSeconds);

    // Smooth lerp toward target emotion weights each frame.
    // EmotionBlendSpeed=3 → ~95% of the way in ~1 second.
    const float Alpha = FMath::Clamp(EmotionBlendSpeed * DeltaSeconds, 0.0f, 1.0f);

    EmotionHappy     = FMath::Lerp(EmotionHappy,     TargetHappy,     Alpha);
    EmotionCurious   = FMath::Lerp(EmotionCurious,   TargetCurious,   Alpha);
    EmotionConcerned = FMath::Lerp(EmotionConcerned, TargetConcerned, Alpha);
    EmotionAmused    = FMath::Lerp(EmotionAmused,    TargetAmused,    Alpha);
    EmotionSad       = FMath::Lerp(EmotionSad,       TargetSad,       Alpha);

    // Is she talking? (The WebSocket manager's state; the Blueprint may not call SetAvatarState.)
    if (ActiveInstance.Get() == this && !WSManager.IsValid() && GetWorld() && FPlatformTime::Seconds() >= WSRetryAt)
    {
        WSRetryAt = FPlatformTime::Seconds() + 1.0;
        for (TActorIterator<AWebSocketManager> It(GetWorld()); It; ++It)
        {
            WSManager = *It;
            break;
        }
    }
    const bool bTalkingNow = bIsTalking || (WSManager.IsValid() && WSManager->CurrentState == EAvatarState::Talking);
    TalkBlend = FMath::FInterpConstantTo(TalkBlend, bTalkingNow ? 1.0f : 0.0f, DeltaSeconds,
                                         1.0f / FMath::Max(RestSmileFadeTime, 0.01f));

    if (bRestSmileFollowsEmotion && ActiveInstance.Get() == this)
    {
        UpdateRestSmile();
    }
}

// ── Resting smile ─────────────────────────────────────────────────────────────

void UAvatarAnimInstance::UpdateRestSmile()
{
    if (!FaceAnimInstance.IsValid())
    {
        RestSmileNode = nullptr;
        const double Now = FPlatformTime::Seconds();
        if (Now < RestSmileRetryAt)
        {
            return;
        }
        RestSmileRetryAt = Now + 1.0;   // the face may not be initialised yet
        UAnimInstance* Face = nullptr;
        TInlineComponentArray<USkeletalMeshComponent*> Meshes;
        if (AActor* Owner = GetOwningActor())
        {
            Owner->GetComponents(Meshes);
        }
        for (USkeletalMeshComponent* Mesh : Meshes)
        {
            if (Mesh && Mesh->GetName() == TEXT("Face"))
            {
                Face = Mesh->GetAnimInstance();
                break;
            }
        }
        if (!Face)
        {
            return;
        }
        FaceAnimInstance = Face;
        static const FName SmileCurve(TEXT("CTRL_expressions_mouthCornerPullL"));
        for (TFieldIterator<FStructProperty> It(Face->GetClass()); It; ++It)
        {
            if (It->Struct == FAnimNode_ModifyCurve::StaticStruct())
            {
                FAnimNode_ModifyCurve* Node = It->ContainerPtrToValuePtr<FAnimNode_ModifyCurve>(Face);
                if (Node->ApplyMode == EModifyCurveApplyMode::Add && Node->CurveMap.Contains(SmileCurve))
                {
                    RestSmileNode = Node;
                    break;
                }
            }
        }
        UE_LOG(LogTemp, Log, TEXT("[Face] resting smile node %s"), RestSmileNode ? TEXT("found") : TEXT("not found (no smile)"));
    }
    if (RestSmileNode)
    {
        // Written on the game thread, read by the face's animation worker: one float, so at worst a frame late.
        // ...and it gives way to her lip sync while she talks (RestSmileWhileTalking of it stays).
        RestSmileNode->Alpha = FMath::Clamp(1.0f - EmotionSad - EmotionConcerned, 0.0f, 1.0f) *
                               FMath::Lerp(1.0f, RestSmileWhileTalking, TalkBlend);
    }
}
