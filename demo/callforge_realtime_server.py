"""One-command VibeVoice-Realtime server for CallForge.

Usage:
    python demo/callforge_realtime_server.py

Optional:
    python demo/callforge_realtime_server.py --port 3000 --model microsoft/VibeVoice-Realtime-0.5B
"""

import argparse
import os

import torch
import uvicorn


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch VibeVoice-Realtime for CallForge")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    os.environ["MODEL_PATH"] = args.model
    os.environ["MODEL_DEVICE"] = device

    print(f"[CallForge] Starting VibeVoice on http://{args.host}:{args.port}")
    print(f"[CallForge] Device: {device}")
    print("[CallForge] Keep this process running while using CallForge.")

    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
