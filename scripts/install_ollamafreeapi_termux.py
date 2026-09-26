#!/usr/bin/env python3
"""Install OllamaFreeAPI from GitHub in Termux and add its Hermes skill."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/mfoud444/ollamafreeapi.git"
REPOSITORY_DIR = Path.home() / "ollamafreeapi"
SKILL_NAME = "ollamafreeapi"

SKILL_CONTENT = """\
---
name: ollamafreeapi
description: Use the OllamaFreeAPI Python client to list currently available models and send text or streaming prompts through community-hosted Ollama servers.
---

# OllamaFreeAPI

Use this skill when asked to discover models through OllamaFreeAPI or send a
prompt to one of those models.

## Important

- The package runs with the Termux `python` command. Use that same command for
  Python examples.
- Prompts go to community-hosted Ollama servers, not a private local model.
  Do not send confidential, personal, or sensitive information without the
  user's explicit permission.
- Model availability can change. List models before choosing one and report
  connection failures instead of claiming a request succeeded.
- The client accepts a prompt string, not full OpenAI-style conversation
  history, and is not a drop-in replacement for Hermes' inference provider.

## Find available models

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; print(OllamaFreeAPI().list_models())"
```

## Send a prompt

Replace the model with one returned by `list_models()`:

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; print(OllamaFreeAPI().chat(prompt='Explain photosynthesis briefly.', model='llama3.2:3b'))"
```

## Stream a response

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; [print(part, end='', flush=True) for part in OllamaFreeAPI().stream_chat(prompt='Explain photosynthesis briefly.', model='llama3.2:3b')]"
```
"""


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a command and stop with a useful error if it fails."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_termux() -> None:
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" not in prefix:
        raise RuntimeError(
            "This installer is intended for the official Termux app. "
            "Install Termux from F-Droid, open it, and run this file there."
        )
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        print(
            "Warning: this phone does not report an ARM64 architecture. "
            "Some Python dependencies may not have Android-compatible wheels.",
            file=sys.stderr,
        )


def get_repository() -> Path:
    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "Git is missing. Run `pkg update && pkg install git`, then rerun "
            "this installer."
        )

    if REPOSITORY_DIR.exists():
        if not (REPOSITORY_DIR / ".git").is_dir():
            raise RuntimeError(
                f"{REPOSITORY_DIR} already exists but is not a Git clone. "
                "Rename that directory or move it, then rerun this installer."
            )
        remote = subprocess.run(
            [git, "-C", str(REPOSITORY_DIR), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().removesuffix(".git")
        if remote.rstrip("/") != REPOSITORY_URL.removesuffix(".git"):
            raise RuntimeError(
                f"{REPOSITORY_DIR} is a clone of a different repository "
                f"({remote}). Rename it before running this installer."
            )
        print(f"Using existing clone at {REPOSITORY_DIR}")
    else:
        run([git, "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_DIR)])
    return REPOSITORY_DIR


def install_package(repository: Path) -> None:
    try:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "."], cwd=repository)
    except subprocess.CalledProcessError:
        if not shutil.which("rustc") or not shutil.which("clang"):
            print(
                "\nPackage installation failed. Termux may need compiler tools "
                "to build an Android-specific Python dependency.\n"
                "Run `pkg install rust clang make pkg-config`, then rerun this "
                "installer.\n",
                file=sys.stderr,
            )
        raise


def install_hermes_skill() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    skill_file = hermes_home / "skills" / SKILL_NAME / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(SKILL_CONTENT, encoding="utf-8")
    return skill_file


def list_models() -> list[str]:
    from ollamafreeapi import OllamaFreeAPI

    models = OllamaFreeAPI().list_models()
    if not models:
        print(
            "The package installed, but its current model catalog is empty. "
            "Check your internet connection and try again.",
            file=sys.stderr,
        )
        return []

    print(f"\nOllamaFreeAPI found {len(models)} model entries:")
    for index, model in enumerate(models, start=1):
        print(f"{index:>2}. {model}")
    return models


def offer_chat(models: list[str]) -> None:
    if not models:
        return

    print(
        "\nPrivacy notice: prompts are sent over the internet to "
        "community-hosted Ollama servers. Do not send secrets or private data."
    )
    consent = input("Try a prompt now? Type YES to continue: ").strip()
    if consent != "YES":
        print("Setup complete. Start Hermes again to load the skill.")
        return

    selection = input(f"Choose a model number (1-{len(models)}): ").strip()
    try:
        model_index = int(selection) - 1
        model = models[model_index]
    except (ValueError, IndexError):
        print("That model number is not in the list; setup is complete.")
        return

    prompt = input("Prompt: ").strip()
    if not prompt:
        print("No prompt entered; setup is complete.")
        return

    from ollamafreeapi import OllamaFreeAPI

    print("\nResponse:\n")
    for text in OllamaFreeAPI().stream_chat(prompt=prompt, model=model):
        print(text, end="", flush=True)
    print()


def main() -> int:
    try:
        ensure_termux()
        repository = get_repository()
        install_package(repository)
        skill_file = install_hermes_skill()
        models = list_models()
        print(f"\nHermes skill saved to: {skill_file}")
        print("If Hermes is running, restart it so it can load the skill.")
        offer_chat(models)
    except subprocess.CalledProcessError as error:
        print(
            f"\nCommand failed with exit code {error.returncode}. "
            "Fix the reported issue and rerun this installer.",
            file=sys.stderr,
        )
        return error.returncode or 1
    except (OSError, RuntimeError) as error:
        print(f"\nSetup failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"\nSetup stopped with an unexpected error: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
