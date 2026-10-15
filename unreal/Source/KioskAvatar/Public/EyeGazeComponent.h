#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "EyeGazeComponent.generated.h"

class AWebSocketManager;
class APlayerCameraManager;
struct FAnimNode_ModifyCurve;
class UAnimInstance;
class USkeletalMeshComponent;

/**
 * Her eyes look at the visitor.
 *
 * Every frame the eyes are aimed at the camera that films her - on a flat screen a gaze straight into that camera
 * reads as "she is looking at me" from anywhere in front of the screen - turned toward where the visitor physically
 * stands by the orchestrator's offset (degrees, + = her left / up). The aim is measured on the eye bones themselves
 * and corrected every frame, so wherever her head points (idle, nods, gestures) the eyes counter-rotate and stay on
 * the visitor. Fixed eye offsets relative to the head would only suit one idle: a different idle turns the head
 * (measured: ~7 deg left and ~4 deg up) and sends her gaze over the visitor's shoulder.
 * When nobody is there the eyes rest in the head and follow its look-arounds.
 *
 * Each eye is aimed on its own at one point - the visitor's eye centreline, at their measured distance - so her eyes
 * converge on them, which makes the gaze more convincing. The eight per-eye CTRL_expressions_eyeLook* curves go
 * through a Modify Curve node (Add mode) that UKioskAnimGraphTools::SetEyeCurveNode puts in ABP_AvatarFace; the
 * Blueprint's shared GazeLook* variables are held at 0. Without that node both eyes move together through GazeLook*
 * (no convergence). While this component is on her actor, UEmotionMoodLibrary::ApplyGazeToFace does nothing, so the
 * Blueprint's per-message path cannot fight it.
 *
 * Eyes and head (so she keeps looking at the visitor when a gesture turns her head away):
 * the eye command is set from where the head WILL be this frame (its motion extrapolated one frame), so the eyes
 * counter-rotate without lag like the vestibulo-ocular reflex; the face rig's own offset is learned continuously.
 * While someone is there her head keeps attending to them: its resting offset from the camera (her idle's stance) is
 * learned and removed, and gesture head turns beyond HeadFreeDeg are softened, through a neck/head correction the
 * body's anim proxy applies (UAvatarAnimInstance). Glance-aways come from the orchestrator as offsets; the head turns
 * HeadShare of them. Console: Kiosk.Head.Attention 0/1, Kiosk.Head.Log 1.
 *
 * Testing without a visitor (console, e.g. through the editor):  Kiosk.Gaze.Test 1  Kiosk.Gaze.Yaw 10
 * Kiosk.Gaze.Pitch -5  Kiosk.Gaze.Log 1   (the log line reports the eyes' aim error in degrees).
 */
UCLASS(ClassGroup = (Kiosk), meta = (BlueprintSpawnableComponent))
class KIOSKAVATAR_API UEyeGazeComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UEyeGazeComponent();

    /** Her face skeletal mesh component on this actor. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze")
    FName FaceComponentName = TEXT("Face");

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze")
    FName LeftEyeBone = TEXT("FACIAL_L_Eye");

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze")
    FName RightEyeBone = TEXT("FACIAL_R_Eye");

    /** The bone whose axes define "eye in head" (left/right, up/down) for the eye controls. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze")
    FName HeadBone = TEXT("head");

    /** Degrees the eyes turn per unit of each control (measured on her face in PIE). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Calibration")
    float DegPerUnitLeft = 39.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Calibration")
    float DegPerUnitRight = 37.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Calibration")
    float DegPerUnitUp = 28.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Calibration")
    float DegPerUnitDown = 41.0f;

    /** Furthest the eyes turn in the head, degrees (beyond this people turn their head). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Limits")
    float MaxYawDeg = 32.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Limits")
    float MaxUpDeg = 20.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Limits")
    float MaxDownDeg = 25.0f;

    /** Seconds the eyes take to close most of the gap to a new target (short = a quick glance). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion", meta = (ClampMin = "0.01"))
    float EyeTimeConstant = 0.045f;

    /** Seconds to relax back into the head after the visitor leaves. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion", meta = (ClampMin = "0.05"))
    float RelaxTime = 0.4f;

    /** The visitor counts as gone when no gaze message has arrived for this long, seconds. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion")
    float StaleAfter = 1.5f;

    /** Tiny fixational eye movements while she looks at someone, degrees (0 = off). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion")
    float MicroSaccadeDeg = 0.3f;

    /** Where her eyes converge when the visitor's distance is unknown: 0 = the camera's distance, else cm. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion")
    float DefaultFocusCm = 0.0f;

    /** Seconds over which the face rig's own eye offset (rest pose, idle eye curves) is learned. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion", meta = (ClampMin = "0.02"))
    float RigOffsetTime = 0.15f;

    /** Extrapolate the head's motion one frame so the eyes counter-rotate without lag. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Motion")
    bool bPredictHead = true;

    /** Head attention: while someone is there her head stays turned toward them. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head")
    bool bHeadAttention = true;

    /** How far gestures may turn her head away from the visitor freely, degrees; beyond it only HeadCompress shows. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head")
    float HeadFreeDeg = 6.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "0", ClampMax = "1"))
    float HeadCompress = 0.3f;

    /** Turns held longer than about this (seconds) count as looking away and are softened; quicker motion (nods,
     *  head shakes) keeps its full size. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "0.05"))
    float HeadTurnTime = 0.5f;

    /** Share of where her eyes look (the follow and glance offsets) that her head turns with. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "0", ClampMax = "1"))
    float HeadShare = 0.35f;

    /** Seconds over which her head's resting offset from the camera is learned (and then removed). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "1"))
    float HeadCenterTime = 15.0f;

    /** Her resting offset is learned only while her head is within this many degrees of where it should point. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head")
    float HeadBiasGateDeg = 12.0f;

    /** The largest resting offset that is removed, degrees. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head")
    float HeadBiasMaxDeg = 10.0f;

    /** Seconds her head takes to follow a new look direction (her eyes move first). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "0.02"))
    float HeadFollowTime = 0.3f;

    /** Seconds the head correction takes to come in when someone arrives, and to let go when they leave. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head", meta = (ClampMin = "0.05"))
    float HeadEngageTime = 0.6f;

    /** The largest head correction, degrees. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Head")
    float HeadMaxDeg = 30.0f;

    /** Report where her eyes are on the screen to the orchestrator (it measures the visitor's angle from there). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze")
    bool bReportEyesOnScreen = true;

    /** Log the aim error once a second while she looks at someone (also console: Kiosk.Gaze.Log 1). */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Eye Gaze|Debug")
    bool bLogAim = false;

protected:
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
    USkeletalMeshComponent* FindFace();
    AWebSocketManager* FindManager();
    bool InitAxes(USkeletalMeshComponent* Face);
    void SetControls(UAnimInstance* Anim, float Left, float Right, float Up, float Down);
    FAnimNode_ModifyCurve* FindEyeNode(UAnimInstance* Anim);
    void ReportEyesOnScreen(const FVector& EyeMid);
    void UpdateHead(float DeltaTime, double Now, USkeletalMeshComponent* Face, const FVector& HeadFwd, const FVector& EyeMid,
                    APlayerCameraManager* CamMgr, float LookYaw, float LookPitch);

    TWeakObjectPtr<USkeletalMeshComponent> FaceMesh;
    TWeakObjectPtr<USkeletalMeshComponent> BodyMesh;
    TWeakObjectPtr<AWebSocketManager> Manager;
    double NextFindAt = 0.0;

    int32 LeftIdx = INDEX_NONE;
    int32 RightIdx = INDEX_NONE;
    int32 HeadIdx = INDEX_NONE;
    bool bAxesReady = false;
    FVector HeadLeftLocal = FVector::ZeroVector;     // her left / up / forward, in the head bone's frame
    FVector HeadUpLocal = FVector::ZeroVector;
    FVector HeadFwdLocal = FVector::ZeroVector;
    FVector EyeFwdLocal = FVector::ZeroVector;       // the eye bone axis that looks out of the pupil

    float CmdYaw[2] = {0.0f, 0.0f};                  // per eye (0 = left, 1 = right): eye-in-head angle asked of the
    float CmdPitch[2] = {0.0f, 0.0f};                // face rig, degrees (+ = her left / up)
    FAnimNode_ModifyCurve* EyeNode = nullptr;        // the per-eye curve node in her face ABP (null = shared controls)
    TWeakObjectPtr<UAnimInstance> EyeNodeOwner;
    bool bEyeNodeLooked = false;
    bool bLocked = false;
    FVector2D Micro = FVector2D::ZeroVector;
    double NextMicroAt = 0.0;
    double NextLogAt = 0.0;
    double LockedSince = 0.0;
    float SumAbsErr = 0.0f;
    int32 ErrSamples = 0;
    double NextReportAt = 0.0;
    FVector2D LastReported = FVector2D(-1.0, -1.0);

    // eyes: the face rig's offset from what it was asked, per eye; the head last frame (for its motion)
    float RigOffYaw[2] = {0.0f, 0.0f};
    float RigOffPitch[2] = {0.0f, 0.0f};
    FQuat PrevHeadQ = FQuat::Identity;
    bool bHavePrevHead = false;
    FVector2D SmoothLook = FVector2D::ZeroVector;   // the visitor offset, lightly smoothed (head motion is not)

    // head attention
    FQuat HeadApplied = FQuat::Identity;            // the correction the body used last frame (body component space)
    FVector2D HeadLook = FVector2D::ZeroVector;     // the share of the eye offset her head follows, smoothed
    FVector2D HeadBias = FVector2D::ZeroVector;     // her head's resting offset from where it should point (yaw, pitch)
    FVector2D HeadSlow = FVector2D::ZeroVector;     // the slow part of her head's turn away (what gets softened)
    float HeadEngage = 0.0f;
    double NextHeadLogAt = 0.0;
};
