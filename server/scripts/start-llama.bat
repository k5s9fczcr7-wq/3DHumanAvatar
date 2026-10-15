@echo off
cd /d "%~dp0.."
REM Pin the LLM to the primary GPU (CUDA device 0) so auto-fit doesn't split it onto
REM the second GPU (which renders UE + runs Whisper).
REM Tested with llama.cpp build b11529 (CUDA 13.4).
REM -c 262144 = Gemma 4's full trained context. Measured on a 32 GB GPU: +5.1 GB
REM vs an 8K context (sliding-window layers keep the KV cache small), 6.8 GB
REM still free during TTS; normal-depth replies run at ~200 tok/s, as with 8K;
REM ~122 tok/s only when the context is nearly full.
REM Your llama.cpp build and model file, relative to the server folder.
set LLAMA_SERVER=llama-server\llama-server.exe
set LLM_MODEL=models\gemma-4-26B-A4B-it-UD-IQ4_NL.gguf
set CUDA_VISIBLE_DEVICES=0
%LLAMA_SERVER% -m %LLM_MODEL% -ngl 99 -c 262144 -np 1 --flash-attn on --port 8080 --host 127.0.0.1 --jinja --reasoning off > logs\llama-server.out.log 2> logs\llama-server.err.log
