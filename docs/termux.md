# Samsung Android setup (Termux)

These instructions are for a Samsung Galaxy S25 (ARM64) using the official
Termux app from F-Droid. Termux is a Linux-like terminal environment for
Android; it is not the same as Samsung's Android shell.

## 1. Install Termux

Install **Termux from F-Droid** and open it once. Do not use the obsolete
Google Play build. Termux add-ons should come from the same source as Termux.

## 2. Install Git and Python

In Termux, paste:

```sh
pkg update -y
pkg install -y git python
```

## 3. Clone the repository and run the installer

Paste this block into Termux:

```sh
cd "$HOME"
git clone --depth 1 https://github.com/mfoud444/ollamafreeapi.git
cd ollamafreeapi
python scripts/install_ollamafreeapi_termux.py
```

The installer installs the Python package and its dependencies, prints the
model catalog it can currently discover, and writes the Hermes skill to
`$HOME/.hermes/skills/ollamafreeapi/SKILL.md`. If the repository is already
cloned, skip the `git clone` line and run the remaining commands from its
directory. To update the clone first, run `git pull` from inside the
`ollamafreeapi` directory.

If the dependency installation fails while building an Android-specific
Python package, install the compiler tools and try again:

```sh
pkg install -y rust clang make pkg-config
python scripts/install_ollamafreeapi_termux.py
```

## 4. Use the client

The installer lists models from the package's current catalog. To list them
again at any time:

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; print(OllamaFreeAPI().list_models())"
```

Use an exact model name returned by that command:

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; print(OllamaFreeAPI().chat(prompt='Explain photosynthesis briefly.', model='llama3.2:3b'))"
```

For streamed output:

```sh
python -c "from ollamafreeapi import OllamaFreeAPI; [print(part, end='', flush=True) for part in OllamaFreeAPI().stream_chat(prompt='Explain photosynthesis briefly.', model='llama3.2:3b')]"
```

Replace `llama3.2:3b` with a model that currently appears in `list_models()`.
The model catalog and server availability can change.

## Hermes Agent

The installer adds an OllamaFreeAPI **skill** to Hermes' default home at
`$HOME/.hermes/skills/ollamafreeapi/`. Restart Hermes after the skill is
installed so the new skill is loaded. If Hermes uses a custom home, set
`HERMES_HOME` before running the installer.

The skill teaches Hermes how to call the Python package. It does **not** add
these models to Hermes' normal model picker or change Hermes' inference
provider. The OllamaFreeAPI client accepts a prompt string and sends it through
community-hosted Ollama servers; it does not provide Hermes tool-calling or
full conversation-history semantics.

**Hermes on Termux:** Hermes' official Termux installation guide currently
warns that its Termux package is broken. This project installer does not
install Hermes. Check the
[official Hermes Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux)
for current status before attempting a separate Hermes installation.

## Privacy and troubleshooting

- Requests are sent over the internet to community-hosted Ollama servers.
  Do not send passwords, API keys, personal information, or confidential
  prompts.
- A model being listed does not guarantee that its server is currently online.
  Retry later or choose another currently listed model if a request fails.
- If Termux cannot find `git`, install it with `pkg install git`.
- If an existing `$HOME/ollamafreeapi` directory is not a clone of this
  repository, the installer stops rather than overwriting it. Rename or move
  that directory before running the clone instructions.
