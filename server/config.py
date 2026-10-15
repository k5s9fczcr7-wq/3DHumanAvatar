"""Server configuration (environment variables override the defaults)."""

import os


class Settings:
    # Server
    HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")   # local only; never expose the admin pages to a network
    PORT: int = int(os.getenv("SERVER_PORT", "8100"))

    # TTS
    REFERENCE_VOICE: str = os.getenv(
        "REFERENCE_VOICE",
        os.path.join(os.path.dirname(__file__), "ReferenceVoice.wav"),
    )

    # LLM — llama-server (llama.cpp)
    LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    LLM_SESSION_IDLE_TIMEOUT_S: float = float(os.getenv("LLM_SESSION_IDLE_TIMEOUT_S", "300"))

    # GPU monitoring
    NVIDIA_SMI: str = os.getenv("NVIDIA_SMI", "nvidia-smi")


settings = Settings()
