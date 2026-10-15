#include "EyeGazeComponent.h"

#include "WebSocketManager.h"
#include "AvatarAnimInstance.h"
#include "Animation/AnimInstance.h"
#include "AnimNodes/AnimNode_ModifyCurve.h"
#include "Camera/PlayerCameraManager.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/GameViewportClient.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Framework/Application/SlateApplication.h"
#include "Widgets/SViewport.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformTime.h"
#include "Kismet/GameplayStatics.h"
#include "UObject/UnrealType.h"

namespace
{
    TAutoConsoleVariable<int32> CVarGazeTest(
        TEXT("Kiosk.Gaze.Test"), 0,
        TEXT("1 = act as if a visitor is present at Kiosk.Gaze.Yaw / Kiosk.Gaze.Pitch (for testing her eyes)."));
    TAutoConsoleVariable<float> CVarGazeYaw(
        TEXT("Kiosk.Gaze.Yaw"), 0.0f, TEXT("Test visitor offset from the camera direction, degrees, + = her left."));
    TAutoConsoleVariable<float> CVarGazePitch(
        TEXT("Kiosk.Gaze.Pitch"), 0.0f, TEXT("Test visitor offset from the camera direction, degrees, + = up."));
    TAutoConsoleVariable<float> CVarGazeFocus(
        TEXT("Kiosk.Gaze.FocusCm"), 0.0f,
        TEXT("Test: where her eyes converge, cm in front of them (0 = the visitor's distance, else the camera's)."));
    TAutoConsoleVariable<int32> CVarHeadAttention(
        TEXT("Kiosk.Head.Attention"), 1, TEXT("1 = her head keeps turned toward the visitor (gesture turns softened); 0 = off."));
    TAutoConsoleVariable<int32> CVarHeadLog(
        TEXT("Kiosk.Head.Log"), 0, TEXT("1 = log her head's offset from the visitor and the correction once a second."));
    TAutoConsoleVariable<int32> CVarGazeLog(
        TEXT("Kiosk.Gaze.Log"), 0, TEXT("1 = log her eyes' aim error once a second while she looks at someone."));

    // Per-eye controls, [eye][left, right, up, down] (eye 0 = her left eye).
    const FName EyeCurve[2][4] = {
        {TEXT("CTRL_expressions_eyeLookLeftL"), TEXT("CTRL_expressions_eyeLookRightL"), TEXT("CTRL_expressions_eyeLookUpL"), TEXT("CTRL_expressions_eyeLookDownL")},
        {TEXT("CTRL_expressions_eyeLookLeftR"), TEXT("CTRL_expressions_eyeLookRightR"), TEXT("CTRL_expressions_eyeLookUpR"), TEXT("CTRL_expressions_eyeLookDownR")}};

    // MetaHumans in component space: forward +Y, her left +X, up +Z.
    const FVector CompLeft(1.0, 0.0, 0.0);
    const FVector CompFwd(0.0, 1.0, 0.0);
    const FVector CompUp(0.0, 0.0, 1.0);

    // Dir turned toward Axis by Deg (negative = away from it), keeping it a unit vector.
    FVector TurnToward(const FVector& Dir, const FVector& Axis, float Deg)
    {
        FVector Perp = Axis - Dir * FVector::DotProduct(Axis, Dir);
        if (!Perp.Normalize())
        {
            return Dir;
        }
        const float R = FMath::DegreesToRadians(Deg);
        return (Dir * FMath::Cos(R) + Perp * FMath::Sin(R)).GetSafeNormal();
    }

    // The size of a rotation, degrees (0..180).
    float QuatAngleDeg(const FQuat& Q)
    {
        return FMath::RadiansToDegrees(Q.GetShortestArcWith(FQuat::Identity).GetAngle());
    }

    float AngleBetweenDeg(const FVector& A, const FVector& B)
    {
        return FMath::RadiansToDegrees(FMath::Acos(FMath::Clamp(FVector::DotProduct(A, B), -1.0, 1.0)));
    }
}

UEyeGazeComponent::UEyeGazeComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UEyeGazeComponent::BeginPlay()
{
    Super::BeginPlay();
    NextFindAt = 0.0;
    FindFace();
    UE_LOG(LogTemp, Log, TEXT("[Gaze] eye controller on %s (face %s)"), *GetNameSafe(GetOwner()),
           FaceMesh.IsValid() ? TEXT("found") : TEXT("not found yet"));
}

USkeletalMeshComponent* UEyeGazeComponent::FindFace()
{
    if (FaceMesh.IsValid())
    {
        return FaceMesh.Get();
    }
    const double Now = FPlatformTime::Seconds();
    if (Now < NextFindAt || !GetOwner())
    {
        return nullptr;
    }
    NextFindAt = Now + 2.0;
    TArray<USkeletalMeshComponent*> Meshes;
    GetOwner()->GetComponents<USkeletalMeshComponent>(Meshes);
    for (USkeletalMeshComponent* M : Meshes)
    {
        if (M && M->GetFName() == FaceComponentName)
        {
            LeftIdx = M->GetBoneIndex(LeftEyeBone);
            RightIdx = M->GetBoneIndex(RightEyeBone);
            HeadIdx = M->GetBoneIndex(HeadBone);
            if (HeadIdx == INDEX_NONE && LeftIdx != INDEX_NONE)
            {
                HeadIdx = M->GetBoneIndex(M->GetParentBone(LeftEyeBone));
            }
            if (LeftIdx == INDEX_NONE || RightIdx == INDEX_NONE || HeadIdx == INDEX_NONE)
            {
                UE_LOG(LogTemp, Warning, TEXT("[Gaze] %s lacks the eye or head bones - eye controller idle"), *M->GetName());
                return nullptr;
            }
            FaceMesh = M;
            // Run before the face animates, so the controls set here are used in the same frame.
            M->PrimaryComponentTick.AddPrerequisite(this, PrimaryComponentTick);
            // ...and before the body, whose anim proxy turns her neck and head by the head-attention correction.
            for (USkeletalMeshComponent* B : Meshes)
            {
                if (B && B->GetFName() == TEXT("Body"))
                {
                    BodyMesh = B;
                    B->PrimaryComponentTick.AddPrerequisite(this, PrimaryComponentTick);
                }
            }
            UE_LOG(LogTemp, Log, TEXT("[Gaze] head attention %s"),
                   BodyMesh.IsValid() ? TEXT("ready (body found)") : TEXT("off: no Body mesh on this actor"));
            return M;
        }
    }
    return nullptr;
}

AWebSocketManager* UEyeGazeComponent::FindManager()
{
    if (Manager.IsValid())
    {
        return Manager.Get();
    }
    if (UWorld* World = GetWorld())
    {
        for (TActorIterator<AWebSocketManager> It(World); It; ++It)
        {
            Manager = *It;
            return *It;
        }
    }
    return nullptr;
}

bool UEyeGazeComponent::InitAxes(USkeletalMeshComponent* Face)
{
    const TArray<FTransform>& CS = Face->GetComponentSpaceTransforms();
    if (CS.Num() <= FMath::Max3(LeftIdx, RightIdx, HeadIdx))
    {
        return false;
    }
    const FQuat HeadQ = CS[HeadIdx].GetRotation();
    HeadLeftLocal = HeadQ.UnrotateVector(CompLeft);
    HeadUpLocal = HeadQ.UnrotateVector(CompUp);
    HeadFwdLocal = HeadQ.UnrotateVector(CompFwd);
    // The eye bone axis that looks out of the pupil = the one pointing most along her forward right now.
    const FQuat EyeQ = CS[LeftIdx].GetRotation();
    const FVector Axes[6] = {FVector(1, 0, 0), FVector(-1, 0, 0), FVector(0, 1, 0), FVector(0, -1, 0),
                             FVector(0, 0, 1), FVector(0, 0, -1)};
    double Best = -2.0;
    for (const FVector& A : Axes)
    {
        const double D = FVector::DotProduct(EyeQ.RotateVector(A), CompFwd);
        if (D > Best)
        {
            Best = D;
            EyeFwdLocal = A;
        }
    }
    bAxesReady = true;
    UE_LOG(LogTemp, Log, TEXT("[Gaze] axes ready: eye looks along local %s (%.2f along her forward)"),
           *EyeFwdLocal.ToString(), Best);
    return true;
}

void UEyeGazeComponent::SetControls(UAnimInstance* Anim, float Left, float Right, float Up, float Down)
{
    UClass* Cls = Anim->GetClass();
    auto Set = [&](const TCHAR* Name, float Value)
    {
        FProperty* Prop = Cls->FindPropertyByName(FName(Name));
        if (!Prop)
        {
            return;
        }
        void* Ptr = Prop->ContainerPtrToValuePtr<void>(Anim);
        const float V = FMath::Clamp(Value, 0.0f, 1.0f);
        if (CastField<FDoubleProperty>(Prop))
        {
            *static_cast<double*>(Ptr) = V;
        }
        else if (CastField<FFloatProperty>(Prop))
        {
            *static_cast<float*>(Ptr) = V;
        }
    };
    Set(TEXT("GazeLookLeft"), Left);       // measured: GazeLookLeft turns her eyes to HER left
    Set(TEXT("GazeLookRight"), Right);
    Set(TEXT("GazeLookUp"), Up);
    Set(TEXT("GazeLookDown"), Down);
}

FAnimNode_ModifyCurve* UEyeGazeComponent::FindEyeNode(UAnimInstance* Anim)
{
    if (EyeNodeOwner.Get() != Anim)
    {
        EyeNodeOwner = Anim;
        EyeNode = nullptr;
        bEyeNodeLooked = false;
    }
    if (!bEyeNodeLooked)
    {
        bEyeNodeLooked = true;
        for (TFieldIterator<FStructProperty> It(Anim->GetClass()); It; ++It)
        {
            if (It->Struct == FAnimNode_ModifyCurve::StaticStruct())
            {
                FAnimNode_ModifyCurve* Node = It->ContainerPtrToValuePtr<FAnimNode_ModifyCurve>(Anim);
                if (Node->ApplyMode == EModifyCurveApplyMode::Add && Node->CurveMap.Contains(EyeCurve[0][0]) &&
                    Node->CurveMap.Contains(EyeCurve[1][0]))
                {
                    EyeNode = Node;
                    break;
                }
            }
        }
        UE_LOG(LogTemp, Log, TEXT("[Gaze] per-eye curve node %s"),
               EyeNode ? TEXT("found: each eye aims at the visitor (convergence)") : TEXT("not found: both eyes move together"));
    }
    return EyeNode;
}

void UEyeGazeComponent::ReportEyesOnScreen(const FVector& EyeMid)
{
    AWebSocketManager* WS = FindManager();
    APlayerController* PC = UGameplayStatics::GetPlayerController(this, 0);
    UGameViewportClient* VC = GetWorld() ? GetWorld()->GetGameViewport() : nullptr;
    if (!WS || !WS->IsConnected() || !PC || !VC)
    {
        return;
    }
    FVector2D Screen;
    if (!PC->ProjectWorldLocationToScreen(EyeMid, Screen, false))
    {
        return;
    }
    FVector2D Size;
    VC->GetViewportSize(Size);
    if (Size.X < 1.0 || Size.Y < 1.0)
    {
        return;
    }
    // Where that is on the physical monitor: the viewport may be a window (e.g. play-in-editor), not the whole
    // screen, so add the viewport's position on the desktop and measure against the monitor it is on.
    FVector2D Desk = Screen;
    FVector2D MonMin(0.0, 0.0), MonSize = Size;
    TSharedPtr<SViewport> Widget = VC->GetGameViewportWidget();
    if (Widget.IsValid() && FSlateApplication::IsInitialized())
    {
        const FGeometry& G = Widget->GetCachedGeometry();
        const FVector2D AbsSize = G.GetAbsoluteSize();
        if (AbsSize.X > 1.0 && AbsSize.Y > 1.0)
        {
            Desk = G.GetAbsolutePosition() + Screen / Size * AbsSize;
            FDisplayMetrics Metrics;
            FDisplayMetrics::RebuildDisplayMetrics(Metrics);     // the editor's cached copy can be empty
            bool bFound = false;
            for (const FMonitorInfo& M : Metrics.MonitorInfo)
            {
                const FPlatformRect& R = M.DisplayRect;
                if (Desk.X >= R.Left && Desk.X < R.Right && Desk.Y >= R.Top && Desk.Y < R.Bottom)
                {
                    MonMin = FVector2D(R.Left, R.Top);
                    MonSize = FVector2D(R.Right - R.Left, R.Bottom - R.Top);
                    bFound = true;
                    break;
                }
            }
            if (!bFound && Metrics.PrimaryDisplayWidth > 0 && Metrics.PrimaryDisplayHeight > 0)
            {
                MonSize = FVector2D(Metrics.PrimaryDisplayWidth, Metrics.PrimaryDisplayHeight);
            }
        }
    }
    const FVector2D OnMonitor = Desk - MonMin;
    if (FVector2D::Distance(OnMonitor, LastReported) < 6.0)
    {
        return;
    }
    LastReported = OnMonitor;
    WS->SendJson(FString::Printf(TEXT("{\"type\":\"eyes_on_screen\",\"x\":%.1f,\"y\":%.1f,\"w\":%.0f,\"h\":%.0f}"),
                                 OnMonitor.X, OnMonitor.Y, MonSize.X, MonSize.Y));
}

void UEyeGazeComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    USkeletalMeshComponent* Face = FindFace();
    UAnimInstance* Anim = Face ? Face->GetAnimInstance() : nullptr;
    if (!Anim || (!bAxesReady && !InitAxes(Face)))
    {
        return;
    }
    const double Now = FPlatformTime::Seconds();

    // 1. Is someone there, and where (degrees from the camera direction, + = her left / up)?
    bool bWant = false;
    float VisYaw = 0.0f, VisPitch = 0.0f;
    if (CVarGazeTest.GetValueOnGameThread() != 0)
    {
        bWant = true;
        VisYaw = CVarGazeYaw.GetValueOnGameThread();
        VisPitch = CVarGazePitch.GetValueOnGameThread();
    }
    else if (AWebSocketManager* WS = FindManager())
    {
        bWant = WS->bGazePresent && (Now - WS->GazeReceivedAt) < StaleAfter;
        VisYaw = WS->GazeYaw;
        VisPitch = WS->GazePitch;
    }
    if (bWant != bLocked)
    {
        if (bLocked && ErrSamples > 0)
        {
            UE_LOG(LogTemp, Log, TEXT("[Gaze] looked at the visitor for %.0f s, mean aim error %.2f deg"),
                   Now - LockedSince, SumAbsErr / ErrSamples);
        }
        bLocked = bWant;
        LockedSince = Now;
        SumAbsErr = 0.0f;
        ErrSamples = 0;
        UE_LOG(LogTemp, Log, TEXT("[Gaze] %s"), bLocked ? TEXT("looking at the visitor") : TEXT("nobody there: eyes rest"));
    }

    // 2. Head and eyes as posed last frame (world space), and the head as it will be this frame: its last-frame motion
    //    once more. The eyes are set for the predicted head, so they counter-rotate without lag (vestibulo-ocular reflex).
    const FQuat HeadQ = Face->GetBoneTransform(HeadIdx).GetRotation();
    FQuat HeadPredQ = HeadQ;
    if (bHavePrevHead && bPredictHead)
    {
        const FQuat Step = (HeadQ * PrevHeadQ.Inverse()).GetNormalized();
        if (QuatAngleDeg(Step) < 10.0f)
        {
            HeadPredQ = (Step * HeadQ).GetNormalized();
        }
    }
    PrevHeadQ = HeadQ;
    bHavePrevHead = true;
    const FVector HLeft = HeadQ.RotateVector(HeadLeftLocal);
    const FVector HUp = HeadQ.RotateVector(HeadUpLocal);
    const FVector HFwd = HeadQ.RotateVector(HeadFwdLocal);
    const FVector PLeft = HeadPredQ.RotateVector(HeadLeftLocal);
    const FVector PUp = HeadPredQ.RotateVector(HeadUpLocal);
    const FVector PFwd = HeadPredQ.RotateVector(HeadFwdLocal);
    const FTransform EyeT[2] = {Face->GetBoneTransform(LeftIdx), Face->GetBoneTransform(RightIdx)};
    const FVector Mid = (EyeT[0].GetLocation() + EyeT[1].GetLocation()) * 0.5;
    FVector EyeDir[2];
    for (int32 E = 0; E < 2; ++E)
    {
        EyeDir[E] = EyeT[E].GetRotation().RotateVector(EyeFwdLocal).GetSafeNormal();
    }
    auto YawIn = [](const FVector& V, const FVector& L, const FVector& F)
    {
        return static_cast<float>(FMath::RadiansToDegrees(FMath::Atan2(FVector::DotProduct(V, L), FVector::DotProduct(V, F))));
    };
    auto PitchIn = [](const FVector& V, const FVector& U)
    {
        return static_cast<float>(FMath::RadiansToDegrees(FMath::Asin(FMath::Clamp(FVector::DotProduct(V, U), -1.0, 1.0))));
    };
    FAnimNode_ModifyCurve* Node = FindEyeNode(Anim);
    const int32 NumEyes = Node ? 2 : 1;           // shared controls: steer the average of both eyes

    // What each eye did last frame vs what it was asked: the face rig's own offset (rest pose, idle eye curves).
    float ActY[2] = {0.0f, 0.0f}, ActP[2] = {0.0f, 0.0f};
    FVector Have[2] = {EyeDir[0], EyeDir[1]};
    const float KOff = 1.0f - FMath::Exp(-DeltaTime / FMath::Max(RigOffsetTime, 0.02f));
    for (int32 E = 0; E < NumEyes; ++E)
    {
        Have[E] = Node ? EyeDir[E] : (EyeDir[0] + EyeDir[1]).GetSafeNormal();
        ActY[E] = YawIn(Have[E], HLeft, HFwd);
        ActP[E] = PitchIn(Have[E], HUp);
        RigOffYaw[E] += (ActY[E] - CmdYaw[E] - RigOffYaw[E]) * KOff;
        RigOffPitch[E] += (ActP[E] - CmdPitch[E] - RigOffPitch[E]) * KOff;
    }

    if (bReportEyesOnScreen && Now >= NextReportAt)
    {
        NextReportAt = Now + 5.0;
        ReportEyesOnScreen(Mid);
    }

    // 3. Where the eyes should point: the visitor's eye centreline, seen through the camera that films her (turned
    //    toward where the visitor stands, and by any glance-away), at the visitor's distance - each eye at that one
    //    point, so they converge. The offset is smoothed a little; the head's motion is not.
    APlayerCameraManager* CamMgr = UGameplayStatics::GetPlayerCameraManager(this, 0);
    const FTransform CompT = Face->GetComponentTransform();
    const float KLook = 1.0f - FMath::Exp(-DeltaTime / FMath::Max(EyeTimeConstant, 0.005f));
    SmoothLook += (FVector2D(VisYaw, VisPitch) - SmoothLook) * KLook;
    if (bLocked && CamMgr)
    {
        if (MicroSaccadeDeg > 0.0f && Now >= NextMicroAt)
        {
            NextMicroAt = Now + FMath::FRandRange(0.7, 2.0);
            Micro = FVector2D(FMath::FRandRange(-1.0, 1.0), FMath::FRandRange(-1.0, 1.0)).GetSafeNormal() *
                    FMath::FRandRange(0.0, MicroSaccadeDeg);
        }
        const FVector ToCam = CamMgr->GetCameraLocation() - Mid;
        FVector Dir = TurnToward(ToCam.GetSafeNormal(), CompT.TransformVectorNoScale(CompLeft), SmoothLook.X + Micro.X);
        Dir = TurnToward(Dir, CompT.TransformVectorNoScale(CompUp), SmoothLook.Y + Micro.Y);
        float FocusCm = static_cast<float>(ToCam.Size());
        if (DefaultFocusCm > 0.0f)
        {
            FocusCm = DefaultFocusCm;
        }
        if (AWebSocketManager* WS = Manager.Get())
        {
            if (WS->GazeFocusIn > 0.0f && CVarGazeTest.GetValueOnGameThread() == 0)
            {
                FocusCm = FMath::Clamp(WS->GazeFocusIn * 2.54f, 25.0f, 400.0f);
            }
        }
        if (CVarGazeFocus.GetValueOnGameThread() > 0.0f)
        {
            FocusCm = FMath::Clamp(CVarGazeFocus.GetValueOnGameThread(), 10.0f, 10000.0f);
        }
        const FVector Target = Mid + Dir * FocusCm;

        // 4. Each eye: the angle it needs in the head as the head will be this frame, less the rig's own offset.
        float Err[2] = {0.0f, 0.0f};
        float TgtY[2] = {0.0f, 0.0f}, TgtP[2] = {0.0f, 0.0f};
        for (int32 E = 0; E < NumEyes; ++E)
        {
            const FVector From = Node ? EyeT[E].GetLocation() : Mid;
            const FVector Want = (Target - From).GetSafeNormal();
            TgtY[E] = FMath::Clamp(YawIn(Want, PLeft, PFwd), -MaxYawDeg, MaxYawDeg);
            TgtP[E] = FMath::Clamp(PitchIn(Want, PUp), -MaxDownDeg, MaxUpDeg);
            CmdYaw[E] = FMath::Clamp(TgtY[E] - RigOffYaw[E], -MaxYawDeg - 10.0f, MaxYawDeg + 10.0f);
            CmdPitch[E] = FMath::Clamp(TgtP[E] - RigOffPitch[E], -MaxDownDeg - 10.0f, MaxUpDeg + 10.0f);
            Err[E] = AngleBetweenDeg(Have[E], Want);
        }
        const float WorstErr = FMath::Max(Err[0], Err[1]);
        SumAbsErr += WorstErr;
        ++ErrSamples;
        if ((bLogAim || CVarGazeLog.GetValueOnGameThread() != 0) && Now >= NextLogAt)
        {
            NextLogAt = Now + 1.0;
            const float Vergence = AngleBetweenDeg(EyeDir[0], EyeDir[1]);
            UE_LOG(LogTemp, Log,
                   TEXT("[Gaze] aim error L %.2f R %.2f deg | focus %.0f cm, eyes converge %.1f deg | left eye in head %+.1f %+.1f "
                        "(target %+.1f %+.1f, rig offset %+.1f %+.1f) | head vs camera yaw %+.1f pitch %+.1f | offset %+.1f %+.1f"),
                   Err[0], NumEyes > 1 ? Err[1] : Err[0], FocusCm, Vergence, ActY[0], ActP[0], TgtY[0], TgtP[0],
                   RigOffYaw[0], RigOffPitch[0], YawIn(ToCam.GetSafeNormal(), HLeft, HFwd), PitchIn(ToCam.GetSafeNormal(), HUp),
                   VisYaw, VisPitch);
        }
        else if (Now - LockedSince > 60.0 && ErrSamples > 0 && Now >= NextLogAt)
        {
            NextLogAt = Now + 60.0;
            UE_LOG(LogTemp, Log, TEXT("[Gaze] still on the visitor, mean aim error %.2f deg"), SumAbsErr / ErrSamples);
        }
    }
    else
    {
        // Nobody there: the eyes relax into the head and follow its look-arounds.
        const float K = 1.0f - FMath::Exp(-DeltaTime / RelaxTime);
        for (int32 E = 0; E < 2; ++E)
        {
            CmdYaw[E] -= K * CmdYaw[E];
            CmdPitch[E] -= K * CmdPitch[E];
        }
    }

    auto Units = [&](int32 E, float Out[4])
    {
        Out[0] = CmdYaw[E] > 0.0f ? CmdYaw[E] / DegPerUnitLeft : 0.0f;       // GazeLookLeft / eyeLookLeft* = HER left
        Out[1] = CmdYaw[E] < 0.0f ? -CmdYaw[E] / DegPerUnitRight : 0.0f;
        Out[2] = CmdPitch[E] > 0.0f ? CmdPitch[E] / DegPerUnitUp : 0.0f;
        Out[3] = CmdPitch[E] < 0.0f ? -CmdPitch[E] / DegPerUnitDown : 0.0f;
    };
    if (Node)
    {
        // Written on the game thread before the face animates this frame (this component ticks first).
        for (int32 E = 0; E < 2; ++E)
        {
            float U[4];
            Units(E, U);
            for (int32 C = 0; C < 4; ++C)
            {
                if (float* V = Node->CurveMap.Find(EyeCurve[E][C]))
                {
                    *V = FMath::Clamp(U[C], 0.0f, 1.0f);
                }
            }
        }
        SetControls(Anim, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        float U[4];
        Units(0, U);
        SetControls(Anim, U[0], U[1], U[2], U[3]);
    }

    // 5. Her head attends to the visitor (the body animates after this component and turns her neck and head by it).
    UpdateHead(DeltaTime, Now, Face, HFwd, Mid, CamMgr, SmoothLook.X, SmoothLook.Y);
}

void UEyeGazeComponent::UpdateHead(float DeltaTime, double Now, USkeletalMeshComponent* Face, const FVector& HeadFwd,
                                   const FVector& EyeMid, APlayerCameraManager* CamMgr, float LookYaw, float LookPitch)
{
    USkeletalMeshComponent* Body = BodyMesh.Get();
    UAvatarAnimInstance* BodyAnim = Body ? Cast<UAvatarAnimInstance>(Body->GetAnimInstance()) : nullptr;
    if (!BodyAnim || !CamMgr)
    {
        return;
    }
    // Her forward / left / up, the camera direction and her head's forward, all in the body's component space.
    const FTransform BodyT = Body->GetComponentTransform();
    const FTransform FaceT = Face->GetComponentTransform();
    const FVector Fwd = BodyT.InverseTransformVectorNoScale(FaceT.TransformVectorNoScale(CompFwd)).GetSafeNormal();
    const FVector Left = BodyT.InverseTransformVectorNoScale(FaceT.TransformVectorNoScale(CompLeft)).GetSafeNormal();
    const FVector Up = BodyT.InverseTransformVectorNoScale(FaceT.TransformVectorNoScale(CompUp)).GetSafeNormal();
    auto Yaw = [&](const FVector& V)
    {
        return static_cast<float>(FMath::RadiansToDegrees(FMath::Atan2(FVector::DotProduct(V, Left), FVector::DotProduct(V, Fwd))));
    };
    auto Pitch = [&](const FVector& V)
    {
        return static_cast<float>(FMath::RadiansToDegrees(FMath::Asin(FMath::Clamp(FVector::DotProduct(V, Up), -1.0, 1.0))));
    };
    auto Direction = [&](float YawDeg, float PitchDeg)
    {
        const float Y = FMath::DegreesToRadians(YawDeg), P = FMath::DegreesToRadians(PitchDeg);
        return (Fwd * (FMath::Cos(P) * FMath::Cos(Y)) + Left * (FMath::Cos(P) * FMath::Sin(Y)) + Up * FMath::Sin(P)).GetSafeNormal();
    };
    // Where her head points without last frame's correction (the body used HeadApplied for the pose measured here).
    const FVector Measured = BodyT.InverseTransformVectorNoScale(HeadFwd).GetSafeNormal();
    const FVector Raw = HeadApplied.Inverse().RotateVector(Measured);
    const FVector CamDir = BodyT.InverseTransformVectorNoScale((CamMgr->GetCameraLocation() - EyeMid).GetSafeNormal());

    // Where it should point: the camera (= at the visitor, on a flat screen), turned by a share of where her eyes look.
    const FVector2D Share = bLocked ? FVector2D(LookYaw, LookPitch) * HeadShare : FVector2D::ZeroVector;
    HeadLook += (Share - HeadLook) * (1.0f - FMath::Exp(-DeltaTime / FMath::Max(HeadFollowTime, 0.02f)));
    const float TgtYaw = Yaw(CamDir) + HeadLook.X;
    const float TgtPitch = Pitch(CamDir) + HeadLook.Y;
    const float DevYaw = FMath::UnwindDegrees(Yaw(Raw) - TgtYaw);
    const float DevPitch = Pitch(Raw) - TgtPitch;

    // Her head's resting offset from there (her idle's stance; measured ~4.5 deg above the camera) is learned slowly,
    // only from moments her head is near where it should point, and removed, so on average her head faces the
    // visitor. Gesture turns, look-arounds and the first frames of play are ignored (before the camera is in place a
    // sample can read ~79 deg and would stick). Starts at zero; never more than HeadBiasMaxDeg.
    if (FMath::Abs(DevYaw) < HeadBiasGateDeg && FMath::Abs(DevPitch) < HeadBiasGateDeg)
    {
        HeadBias += (FVector2D(DevYaw, DevPitch) - HeadBias) * (1.0f - FMath::Exp(-DeltaTime / FMath::Max(HeadCenterTime, 1.0f)));
        HeadBias.X = FMath::Clamp(HeadBias.X, -HeadBiasMaxDeg, HeadBiasMaxDeg);
        HeadBias.Y = FMath::Clamp(HeadBias.Y, -HeadBiasMaxDeg, HeadBiasMaxDeg);
    }
    // Gesture turns: free up to HeadFreeDeg from the visitor, beyond it only HeadCompress of the turn shows.
    auto Soften = [this](float X)
    {
        const float A = FMath::Abs(X);
        return A <= HeadFreeDeg ? X : FMath::Sign(X) * (HeadFreeDeg + (A - HeadFreeDeg) * HeadCompress);
    };
    // Only SUSTAINED turns are softened: the deviation's slow part (over ~HeadTurnTime) is compressed, its quick part
    // passes through, so nods and head shakes keep their full size while a long look away is held near the visitor.
    const FVector2D Centred(DevYaw - HeadBias.X, DevPitch - HeadBias.Y);
    HeadSlow += (Centred - HeadSlow) * (1.0f - FMath::Exp(-DeltaTime / FMath::Max(HeadTurnTime, 0.05f)));
    const float ShowYaw = Soften(HeadSlow.X) + (Centred.X - HeadSlow.X);
    const float ShowPitch = Soften(HeadSlow.Y) + (Centred.Y - HeadSlow.Y);

    // Only while someone is there; it comes in over HeadEngageTime (she turns to face them) and lets go the same way.
    const bool bOn = bHeadAttention && bLocked && CVarHeadAttention.GetValueOnGameThread() != 0;
    HeadEngage = FMath::FInterpConstantTo(HeadEngage, bOn ? 1.0f : 0.0f, DeltaTime, 1.0f / FMath::Max(HeadEngageTime, 0.05f));
    FQuat Want = FQuat::FindBetweenNormals(Raw, Direction(TgtYaw + ShowYaw, TgtPitch + ShowPitch));
    const float WantDeg = QuatAngleDeg(Want);
    if (WantDeg > HeadMaxDeg && WantDeg > KINDA_SMALL_NUMBER)
    {
        Want = FQuat::Slerp(FQuat::Identity, Want, HeadMaxDeg / WantDeg);
    }
    Want = FQuat::Slerp(FQuat::Identity, Want, HeadEngage);
    HeadApplied = FQuat::Slerp(HeadApplied, Want, 1.0f - FMath::Exp(-DeltaTime / 0.05f)).GetNormalized();
    BodyAnim->SetHeadCorrection(HeadApplied);

    if (CVarHeadLog.GetValueOnGameThread() != 0 && Now >= NextHeadLogAt)
    {
        NextHeadLogAt = Now + 1.0;
        UE_LOG(LogTemp, Log,
               TEXT("[Head] off the visitor yaw %+.1f pitch %+.1f (resting offset %+.1f %+.1f) -> shown %+.1f %+.1f, "
                    "measured %+.1f %+.1f | correction %.1f deg, engaged %.2f"),
               DevYaw, DevPitch, HeadBias.X, HeadBias.Y, ShowYaw, ShowPitch,
               FMath::UnwindDegrees(Yaw(Measured) - TgtYaw), Pitch(Measured) - TgtPitch, QuatAngleDeg(HeadApplied), HeadEngage);
    }
}
