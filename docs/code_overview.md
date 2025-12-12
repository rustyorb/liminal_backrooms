# Liminal Backrooms Codebase Overview

This document summarizes the major modules and flows in the project so future changes can be scoped quickly.

## Application Entry and Turn Processing (`main.py`)
- Loads runtime settings and helper APIs, then wires the GUI to worker threads for concurrent model turns.【F:main.py†L3-L135】
- `Worker` wraps `ai_turn` in a `QRunnable`, emitting progress, streaming chunks, and final results back to the UI via Qt signals to keep the interface responsive.【F:main.py†L48-L135】
- `ai_turn` prepares the per-model system prompt (identity prefix, prompt additions, temperature overrides), filters conversation history, and invokes the chosen backend model ID from `config.AI_MODELS`. It also handles branching markers ("rabbitholes"/"forks") and strips duplicate or empty messages before calling the API client utilities.【F:main.py†L133-L280】

## UI Layer (`gui.py`)
- `LiminalBackroomsApp` is a `QMainWindow` that stores conversation state, generated images/videos, and branch conversations, sets up the cyberpunk-themed interface, and connects signals for updates.【F:gui.py†L3295-L3355】
- The UI builds a split layout for the conversation view and configuration panes, manages theme/styling through a shared color palette, and loads bundled fonts for consistent rendering.【F:gui.py†L3330-L3355】【F:gui.py†L1-L69】

## Configuration and Scenario Library (`config.py`)
- Central runtime toggles (turn delay, chain-of-thought visibility, Sora defaults) plus an OpenRouter model directory map display names to provider-specific IDs used during dispatch.【F:config.py†L8-L48】
- `SYSTEM_PROMPT_PAIRS` hosts scenario templates for up to five AIs, describing available tool commands and behavioral tone (e.g., Backrooms Classic, Group Chat). Prompts are grouped per AI slot for easy GUI selection.【F:config.py†L51-L142】【F:config.py†L170-L248】

## Command Extraction (`command_parser.py`)
- Parses AI responses for structured commands like `!image`, `!video`, `!search`, `!add_ai`, `!prompt`, `!temperature`, and `!mute_self`, returning cleaned text plus a list of `AgentCommand` instances for execution.【F:command_parser.py†L1-L117】
- `format_command_result` standardizes tool feedback (success/failure) for display in the transcript.【F:command_parser.py†L120-L125】

## API Utilities (`shared_utils.py`)
- Houses wrappers for Anthropic, OpenAI, Replicate, Together, and OpenRouter calls, including streaming support and duplicate-message filtering for Claude requests. Environment variables are loaded once and clients initialized globally.【F:shared_utils.py†L1-L120】
- Also includes helpers for media generation (images/videos), HTML export, and optional web search (via DuckDuckGo Search/BeautifulSoup when installed).【F:shared_utils.py†L1-L120】

## BackroomsBench Evaluation (`backroomsbench.py`)
- Defines a multi-judge evaluation workflow using OpenRouter-hosted models to score conversations on depth, creativity, emergent themes, authenticity, and collaboration. Judges and scoring rubric are declared in constants and executed via concurrent futures.【F:backroomsbench.py†L1-L74】

## Data and Exports
- Conversations, images, and videos are stored under `exports/`, `images/`, and `videos/` (referenced by UI state). HTML exports can be opened via the helper in `shared_utils`. Logs and benchmark reports have dedicated folders (`logs/`, `backroomsbench_reports/`).

## Typical Flow
1. User configures models/scenario in the GUI and starts a run.
2. For each turn, the app enqueues a `Worker` that assembles the system prompt, cleans history, and calls the selected model backend.
3. Responses are streamed to the UI, parsed for commands (image/video/search/model management/self-modification), and appended to conversation state.
4. Media generation and evaluation utilities feed results back to the UI for rendering and export.
