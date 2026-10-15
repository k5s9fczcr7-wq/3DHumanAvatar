#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "IWebSocket.h"
#include "WebSocketManager.generated.h"

class USubmixEffectDelayPreset;

// Avatar states driven by Python backend
UENUM(BlueprintType)
enum class EAvatarState : uint8
{
    Idle        UMETA(DisplayName = "Idle"),
    Listening   UMETA(DisplayName = "Listening"),
    Thinking    UMETA(DisplayName = "Thinking"),
    Talking     UMETA(DisplayName = "Talking"),
};

// Emotion hints driven by LLM reply tags — uint8 values aligned to ERealisticMetaHumanLipSyncMood
UENUM(BlueprintType)
enum class EAvatarEmotion : uint8
{
    Neutral    = 0  UMETA(DisplayName = "Neutral"),
    Happy      = 1  UMETA(DisplayName = "Happy"),
    Sad        = 2  UMETA(DisplayName = "Sad"),
    Disgust    = 3  UMETA(DisplayName = "Disgust"),
    Angry      = 4  UMETA(DisplayName = "Angry"),
    Surprised  = 5  UMETA(DisplayName = "Surprised"),
    Fearful    = 6  UMETA(DisplayName = "Fearful"),
    Confident  = 10 UMETA(DisplayName = "Confident"),
    Excited    = 14 UMETA(DisplayName = "Excited"),
    Bored      = 15 UMETA(DisplayName = "Bored"),
    Playful    = 17 UMETA(DisplayName = "Playful"),
    Confused   = 19 UMETA(DisplayName = "Confused"),
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnSpeakReceived,  const FString&, WavFilePath);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnStateChanged,   EAvatarState,   NewState);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEmotionChanged, EAvatarEmotion, NewEmotion, float, Intensity);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNameReceived,   const FString&, UserName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnStopReceived);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnGazeReceived, double, Yaw, double, Pitch);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnIndicatorChanged, const FString&, IndicatorValue);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnGestureReceived, const FString&, Intent, const FString&, PlayedSnippet);

/**
 * AWebSocketManager
 *
 * Connects to the Python backend WebSocket server on ws://127.0.0.1:8765.
 * Parses incoming JSON messages and fires Blueprint-assignable delegates so
 * the Animation Blueprint and the avatar Blueprint (audio, lip sync) can react.
 *
 * Drop this Actor into your persistent level. It auto-connects on BeginPlay
 * and reconnects automatically if the connection drops.
 *
 * Protocol (Python → UE5):
 *   {"type":"speak",   "path":"/tmp/kiosk_speech_1234.wav"}
 *   {"type":"stop"}                                         — barge-in: stop audio + lip sync
 *   {"type":"state",   "value":"idle"|"listening"|"thinking"|"talking"}
 *   {"type":"emotion", "value":"neutral"|"happy"|"sad"|"disgust"|"angry"|"surprised"|"fearful"|"confident"|"excited"|"bored"|"playful"|"confused", "intensity":0.0-1.0}
 *   {"type":"name",    "value":"Jane Doe"}
 *   {"type":"gaze",    "yaw":3.2, "pitch":18.6}               — eye gaze target (degrees, ~10fps)
 *
 * Protocol (UE5 → Python):
 *   {"type":"ready"}   — sent once on successful connection
 */
UCLASS(BlueprintType, Blueprintable)
class KIOSKAVATAR_API AWebSocketManager : public AActor
{
    GENERATED_BODY()

public:
    AWebSocketManager();

    // ── Configuration ────────────────────────────────────────────────────────

    /** WebSocket server URL — must match the orchestrator's WebSocket port (8765) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    FString ServerURL = TEXT("ws://127.0.0.1:8765");

    /** Seconds between reconnect attempts when connection is lost */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    float ReconnectDelay = 3.0f;

    // ── Current state (readable from Blueprint / Animation Blueprint) ────────

    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    EAvatarState CurrentState = EAvatarState::Idle;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    EAvatarEmotion CurrentEmotion = EAvatarEmotion::Neutral;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float CurrentEmotionIntensity = 0.5f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    FString CurrentUserName;

    /** Current gaze target — updated at ~10fps from face tracking */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float GazeYaw = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float GazePitch = 0.0f;

    /** True while the orchestrator sees a visitor (the gaze message's "present" field; true when it is missing). */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    bool bGazePresent = false;

    /** The visitor's distance from the camera, inches (0 = unknown). */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float GazeDistanceIn = 0.0f;

    /** The visitor's eye centreline distance from HER eyes on the screen, inches (0 = unknown): where her eyes converge. */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float GazeFocusIn = 0.0f;

    /** Her voice is held back this long on its way to the speakers so it lines up with her lips, ms. The lip sync
     *  hears each sound as Unreal generates it, then needs its lookahead and smoothing before her mouth moves
     *  (without the delay her mouth trails her voice by about 0.1 s). Set by the orchestrator:
     *  {"type":"av_sync","voice_delay_ms":N} from kiosk_config.ini [avatar] voice_delay_ms. */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    float VoiceDelayMs = 0.0f;

    /** FPlatformTime::Seconds() when the last gaze message arrived (UEyeGazeComponent drops a stale visitor). */
    double GazeReceivedAt = 0.0;

    /** Current UX mode indicator state. One of:
     *  "grey"   — no face + no authenticated session
     *  "white"  — face present, pre-authentication
     *  "green"  — listening (includes VAD active + ASR running)
     *  "amber"  — processing (LLM streaming + TTS synth)
     *  "red"    — speaking (audio playing)
     *  "purple" — committing facts to storage (post user_left)
     */
    UPROPERTY(BlueprintReadOnly, Category = "Avatar")
    FString CurrentIndicator = TEXT("grey");

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket")
    bool bIsConnected = false;

    // ── Delegates (bind in Blueprint or C++) ─────────────────────────────────

    /** Fired when a "speak" message arrives with the WAV file path */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnSpeakReceived OnSpeakReceived;

    /** Fired when the avatar state changes (idle/listening/thinking/talking) */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnStateChanged OnStateChanged;

    /** Fired when the emotion changes */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnEmotionChanged OnEmotionChanged;

    /** Fired when a user name is identified by face recognition */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnNameReceived OnNameReceived;

    /** Fired on barge-in — stop audio playback and lip sync immediately */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnStopReceived OnStopReceived;

    /** Fired when gaze target updates (~10fps) — yaw/pitch in degrees */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnGazeReceived OnGazeReceived;

    /** Fired when the UX mode indicator changes. The value is a
     *  short string ("grey"/"white"/"green"/"amber"/"red"/"purple"); the Blueprint
     *  is responsible for mapping it to a material color / pulse / brightness. */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnIndicatorChanged OnIndicatorChanged;

    /** Fired after a "gesture" message: the requested intent and the snippet that played ("None" if none did).
     *  The message itself is handled in C++ (UAvatarAnimInstance::PlayGesture); this is for display/debugging.
     *  {"type":"gesture","intent":"agree","emotion":"happy","intensity":0.4,"role":"speaking","interrupt":false}
     *  The short form {"type":"gesture","name":"nod|wave|tilt_head|thinking_pose|shrug"} (the LLM's gesture tags)
     *  maps to agree/welcome/empathize/unsure/unsure. */
    UPROPERTY(BlueprintAssignable, Category = "Avatar|Events")
    FOnGestureReceived OnGestureReceived;

    // ── Blueprint callable helpers ────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Connect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Disconnect();

    UFUNCTION(BlueprintPure, Category = "WebSocket")
    bool IsConnected() const { return bIsConnected; }

    /** Send one JSON message to the orchestrator (e.g. where her eyes are on the screen). */
    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SendJson(const FString& Json);

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TSharedPtr<IWebSocket> WebSocket;

    /** A delay on the main output (Synthesis plugin) - after the lip sync has heard the sound. */
    UPROPERTY(Transient)
    TObjectPtr<USubmixEffectDelayPreset> VoiceDelay;
    void ApplyVoiceDelay(float Ms);
    FTimerHandle ReconnectTimer;
    bool bShuttingDown = false;

    void CreateAndConnect();
    void ScheduleReconnect();
    void HandleMessage(const FString& Message);
    void SendReady();

    // Helpers
    static EAvatarState  ParseState(const FString& Value);
    static EAvatarEmotion ParseEmotion(const FString& Value);
    static FString LegacyGestureIntent(const FString& Name);
};
