#include "WebSocketManager.h"
#include "AvatarAnimInstance.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "TimerManager.h"
#include "Components/AudioComponent.h"
#include "UObject/UObjectIterator.h"
#include "HAL/PlatformTime.h"
#include "AudioMixerBlueprintLibrary.h"
#include "SubmixEffects/SubmixEffectDelay.h"

AWebSocketManager::AWebSocketManager()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AWebSocketManager::BeginPlay()
{
    Super::BeginPlay();
    CreateAndConnect();
}

void AWebSocketManager::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    bShuttingDown = true;
    if (VoiceDelay)
    {
        UAudioMixerBlueprintLibrary::RemoveMasterSubmixEffect(this, VoiceDelay);
        VoiceDelay = nullptr;
    }
    if (WebSocket)
    {
        WebSocket->OnConnected().Clear();
        WebSocket->OnConnectionError().Clear();
        WebSocket->OnClosed().Clear();
        WebSocket->OnMessage().Clear();
    }
    Disconnect();
    Super::EndPlay(EndPlayReason);
}

// ── Public API ────────────────────────────────────────────────────────────────

void AWebSocketManager::Connect()
{
    if (WebSocket && WebSocket->IsConnected())
    {
        return;
    }
    CreateAndConnect();
}

void AWebSocketManager::Disconnect()
{
    GetWorldTimerManager().ClearTimer(ReconnectTimer);
    if (WebSocket)
    {
        WebSocket->Close();
        WebSocket = nullptr;
    }
    bIsConnected = false;
}

// ── Private ───────────────────────────────────────────────────────────────────

void AWebSocketManager::ApplyVoiceDelay(float Ms)
{
    Ms = FMath::Clamp(Ms, 0.0f, 400.0f);
    if (!VoiceDelay && Ms <= 0.0f)
    {
        VoiceDelayMs = 0.0f;
        return;
    }
    FSubmixEffectDelaySettings Settings;
    Settings.MaximumDelayLength = 500.0f;
    Settings.InterpolationTime = 0.0f;
    Settings.DelayLength = Ms;
    if (!VoiceDelay)
    {
        VoiceDelay = NewObject<USubmixEffectDelayPreset>(GetTransientPackage());
        VoiceDelay->SetDefaultSettings(Settings);
        UAudioMixerBlueprintLibrary::AddMasterSubmixEffect(this, VoiceDelay);
    }
    else
    {
        VoiceDelay->SetSettings(Settings);
    }
    VoiceDelayMs = Ms;
    UE_LOG(LogTemp, Log, TEXT("[WS] her voice is held %.0f ms on its way to the speakers so it lines up with her lips"), Ms);
}

void AWebSocketManager::CreateAndConnect()
{
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
    {
        FModuleManager::Get().LoadModule("WebSockets");
    }

    WebSocket = FWebSocketsModule::Get().CreateWebSocket(ServerURL, TEXT("ws"));

    WebSocket->OnConnected().AddLambda([this]()
    {
        bIsConnected = true;
        UE_LOG(LogTemp, Log, TEXT("[WS] Connected to %s"), *ServerURL);
        SendReady();
    });

    WebSocket->OnConnectionError().AddLambda([this](const FString& Error)
    {
        if (bShuttingDown) return;
        bIsConnected = false;
        UE_LOG(LogTemp, Warning, TEXT("[WS] Connection error: %s — retrying in %.1fs"), *Error, ReconnectDelay);
        ScheduleReconnect();
    });

    WebSocket->OnClosed().AddLambda([this](int32 StatusCode, const FString& Reason, bool bWasClean)
    {
        if (bShuttingDown) return;
        bIsConnected = false;
        UE_LOG(LogTemp, Log, TEXT("[WS] Closed (code=%d, reason=%s) — retrying in %.1fs"),
               StatusCode, *Reason, ReconnectDelay);
        ScheduleReconnect();
    });

    WebSocket->OnMessage().AddLambda([this](const FString& Message)
    {
        HandleMessage(Message);
    });

    WebSocket->Connect();
}

void AWebSocketManager::ScheduleReconnect()
{
    GetWorldTimerManager().SetTimer(ReconnectTimer, [this]()
    {
        UE_LOG(LogTemp, Log, TEXT("[WS] Attempting reconnect to %s"), *ServerURL);
        CreateAndConnect();
    }, ReconnectDelay, false);
}

void AWebSocketManager::SendJson(const FString& Json)
{
    if (WebSocket && WebSocket->IsConnected())
    {
        WebSocket->Send(Json);
    }
}

void AWebSocketManager::SendReady()
{
    if (WebSocket && WebSocket->IsConnected())
    {
        WebSocket->Send(TEXT("{\"type\":\"ready\"}"));
        UE_LOG(LogTemp, Log, TEXT("[WS] → {\"type\":\"ready\"}"));
    }
}

void AWebSocketManager::HandleMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("[WS] Failed to parse JSON: %s"), *Message);
        return;
    }

    FString Type;
    if (!JsonObject->TryGetStringField(TEXT("type"), Type))
    {
        UE_LOG(LogTemp, Warning, TEXT("[WS] Message missing 'type' field"));
        return;
    }

    if (Type == TEXT("speak"))
    {
        FString Path;
        if (JsonObject->TryGetStringField(TEXT("path"), Path))
        {
            UE_LOG(LogTemp, Log, TEXT("[WS] speak → %s"), *Path);
            OnSpeakReceived.Broadcast(Path);
        }
    }
    else if (Type == TEXT("stop"))
    {
        UE_LOG(LogTemp, Log, TEXT("[WS] stop (barge-in) — stopping all audio"));
        // Stop all playing audio components in the world
        if (UWorld* World = GetWorld())
        {
            for (TObjectIterator<UAudioComponent> It; It; ++It)
            {
                if (It->IsPlaying() && It->GetWorld() == World)
                {
                    It->Stop();
                }
            }
        }
        OnStopReceived.Broadcast();
    }
    else if (Type == TEXT("state"))
    {
        FString Value;
        if (JsonObject->TryGetStringField(TEXT("value"), Value))
        {
            CurrentState = ParseState(Value);
            UE_LOG(LogTemp, Log, TEXT("[WS] state → %s"), *Value);
            OnStateChanged.Broadcast(CurrentState);
        }
    }
    else if (Type == TEXT("emotion"))
    {
        FString Value;
        if (JsonObject->TryGetStringField(TEXT("value"), Value))
        {
            CurrentEmotion = ParseEmotion(Value);
            double IntensityVal = 0.5;
            JsonObject->TryGetNumberField(TEXT("intensity"), IntensityVal);
            CurrentEmotionIntensity = FMath::Clamp(static_cast<float>(IntensityVal), 0.0f, 1.0f);
            UE_LOG(LogTemp, Log, TEXT("[WS] emotion → %s (%.2f)"), *Value, CurrentEmotionIntensity);
            OnEmotionChanged.Broadcast(CurrentEmotion, CurrentEmotionIntensity);
        }
    }
    else if (Type == TEXT("name"))
    {
        FString Value;
        if (JsonObject->TryGetStringField(TEXT("value"), Value))
        {
            CurrentUserName = Value;
            UE_LOG(LogTemp, Log, TEXT("[WS] name → %s"), *Value);
            OnNameReceived.Broadcast(Value);
        }
    }
    else if (Type == TEXT("av_sync"))
    {
        double Ms = 0.0;
        if (JsonObject->TryGetNumberField(TEXT("voice_delay_ms"), Ms))
        {
            ApplyVoiceDelay(static_cast<float>(Ms));
        }
    }
    else if (Type == TEXT("gaze"))
    {
        double Yaw = 0.0, Pitch = 0.0, Distance = 0.0, Focus = 0.0;
        bool bPresent = true;             // no "present" field: someone is there
        JsonObject->TryGetNumberField(TEXT("yaw"), Yaw);
        JsonObject->TryGetNumberField(TEXT("pitch"), Pitch);
        JsonObject->TryGetBoolField(TEXT("present"), bPresent);
        JsonObject->TryGetNumberField(TEXT("distance_in"), Distance);
        JsonObject->TryGetNumberField(TEXT("focus_in"), Focus);
        GazeYaw = static_cast<float>(Yaw);
        GazePitch = static_cast<float>(Pitch);
        bGazePresent = bPresent;
        GazeDistanceIn = static_cast<float>(Distance);
        GazeFocusIn = static_cast<float>(Focus);
        GazeReceivedAt = FPlatformTime::Seconds();
        OnGazeReceived.Broadcast(GazeYaw, GazePitch);
    }
    else if (Type == TEXT("indicator"))
    {
        // UX mode indicator. Value is one of
        // "grey"/"white"/"green"/"amber"/"red"/"purple". Blueprint maps it to
        // material color + pulse on the lapel-pin actor.
        FString Value;
        if (JsonObject->TryGetStringField(TEXT("value"), Value))
        {
            CurrentIndicator = Value;
            UE_LOG(LogTemp, Log, TEXT("[WS] indicator → %s"), *Value);
            OnIndicatorChanged.Broadcast(Value);
        }
    }
    else if (Type == TEXT("gesture"))
    {
        // {"type":"gesture","intent":"agree","emotion":"happy","intensity":0.4,"role":"speaking","interrupt":false}
        // (or the short form {"type":"gesture","name":"nod"}); emotion/intensity default to the last emotion message.
        FString Intent, Name, Emotion, GestureRole;
        if (!JsonObject->TryGetStringField(TEXT("intent"), Intent) && JsonObject->TryGetStringField(TEXT("name"), Name))
        {
            Intent = LegacyGestureIntent(Name);
        }
        JsonObject->TryGetStringField(TEXT("role"), GestureRole);
        bool bInterrupt = false;
        JsonObject->TryGetBoolField(TEXT("interrupt"), bInterrupt);
        const EAvatarEmotion Emo = JsonObject->TryGetStringField(TEXT("emotion"), Emotion) ? ParseEmotion(Emotion) : CurrentEmotion;
        double Intensity = -1.0;
        if (!JsonObject->TryGetNumberField(TEXT("intensity"), Intensity))
        {
            Intensity = -1.0;
        }
        FName Played = NAME_None;
        if (UAvatarAnimInstance* Anim = UAvatarAnimInstance::GetActive())
        {
            Played = Anim->PlayGesture(FName(*Intent), Emo, static_cast<float>(Intensity),
                                       GestureRole.IsEmpty() ? NAME_None : FName(*GestureRole), bInterrupt);
        }
        UE_LOG(LogTemp, Log, TEXT("[WS] gesture %s (role %s) -> %s"), *Intent, *GestureRole, *Played.ToString());
        OnGestureReceived.Broadcast(Intent, Played.ToString());
    }
    else
    {
        UE_LOG(LogTemp, Verbose, TEXT("[WS] Unknown message type: %s"), *Type);
    }
}

FString AWebSocketManager::LegacyGestureIntent(const FString& Name)
{
    // The LLM's five gesture tags, mapped onto intents.
    if (Name == TEXT("nod"))           return TEXT("agree");
    if (Name == TEXT("wave"))          return TEXT("welcome");
    if (Name == TEXT("tilt_head"))     return TEXT("empathize");
    if (Name == TEXT("thinking_pose")) return TEXT("unsure");
    if (Name == TEXT("shrug"))         return TEXT("unsure");
    return Name;                       // already an intent name
}

// ── Parsers ───────────────────────────────────────────────────────────────────

EAvatarState AWebSocketManager::ParseState(const FString& Value)
{
    if (Value == TEXT("listening")) return EAvatarState::Listening;
    if (Value == TEXT("thinking"))  return EAvatarState::Thinking;
    if (Value == TEXT("talking"))   return EAvatarState::Talking;
    return EAvatarState::Idle;
}

EAvatarEmotion AWebSocketManager::ParseEmotion(const FString& Value)
{
    // Primary names (match LLM directive output)
    if (Value == TEXT("happy"))     return EAvatarEmotion::Happy;
    if (Value == TEXT("sad"))       return EAvatarEmotion::Sad;
    if (Value == TEXT("disgust"))   return EAvatarEmotion::Disgust;
    if (Value == TEXT("angry"))     return EAvatarEmotion::Angry;
    if (Value == TEXT("surprised")) return EAvatarEmotion::Surprised;
    if (Value == TEXT("fearful"))   return EAvatarEmotion::Fearful;
    if (Value == TEXT("confident")) return EAvatarEmotion::Confident;
    if (Value == TEXT("excited"))   return EAvatarEmotion::Excited;
    if (Value == TEXT("bored"))     return EAvatarEmotion::Bored;
    if (Value == TEXT("playful"))   return EAvatarEmotion::Playful;
    if (Value == TEXT("confused"))  return EAvatarEmotion::Confused;

    // Synonyms / natural-language fallbacks
    if (Value == TEXT("disgusted")) return EAvatarEmotion::Disgust;
    if (Value == TEXT("anger"))     return EAvatarEmotion::Angry;
    if (Value == TEXT("surprise"))  return EAvatarEmotion::Surprised;
    if (Value == TEXT("afraid"))    return EAvatarEmotion::Fearful;
    if (Value == TEXT("fear"))      return EAvatarEmotion::Fearful;

    // Other emotion names → nearest mood
    if (Value == TEXT("curious"))   return EAvatarEmotion::Excited;
    if (Value == TEXT("concerned")) return EAvatarEmotion::Sad;
    if (Value == TEXT("amused"))    return EAvatarEmotion::Playful;

    return EAvatarEmotion::Neutral;
}
