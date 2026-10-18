"""One-time interactive setup: AI provider choice, local model readiness
(Ollama) or API key entry (Gemini). Runs only when no local config exists
yet, or when the user explicitly asks to reconfigure (`researchgenie config`).
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from researchgenie import ollama_manager as om
from researchgenie import scholarly_setup
from researchgenie.config import Config, load, save
from researchgenie.theme import ACCENT, MUTED, OK, WARN

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"


def _choose_provider(console: Console) -> str:
    console.print(Panel(
        "[bold]Choose your AI engine[/bold]\n\n"
        f"[{OK}]1. Ollama[/{OK}] — [{MUTED}](recommended)[/{MUTED}] runs locally, "
        "your research data never leaves this device, no API key needed.\n"
        f"[{ACCENT}]2. Gemini[/{ACCENT}] — cloud API, requires your own Google "
        "Gemini API key and an internet connection.",
        title="AI Engine", border_style=ACCENT,
    ))
    choice = Prompt.ask("Select", choices=["1", "2"], default="1")
    return "ollama" if choice == "1" else "gemini"


async def _setup_ollama(console: Console, model: str) -> bool:
    console.print(f"\n[{MUTED}]Checking for Ollama...[/{MUTED}]")
    if not om.is_ollama_on_path():
        console.print(f"[{WARN}]Ollama was not found on this system.[/{WARN}]")
        if Confirm.ask("Attempt automatic installation now?", default=True):
            result = om.attempt_automatic_install()
            if result.succeeded:
                console.print(f"[{OK}]✓[/{OK}] {result.message}")
            else:
                console.print(f"[{WARN}]{result.message}[/{WARN}]")
                console.print(
                    f"[{MUTED}]Install Ollama, then run `researchgenie` again "
                    "to continue setup.[/grey62]"
                )
                return False
        else:
            console.print(
                f"[{MUTED}]Install Ollama from https://ollama.com/download, "
                "then run `researchgenie` again.[/grey62]"
            )
            return False
    else:
        console.print(f"[{OK}]✓[/{OK}] Ollama found")

    if not await om.is_ollama_server_running():
        console.print(f"[{MUTED}]Starting the local Ollama server...[/{MUTED}]")
        om.start_ollama_server()
        for _ in range(20):
            await asyncio.sleep(1)
            if await om.is_ollama_server_running():
                break
        else:
            console.print(
                f"[{WARN}]Could not confirm the Ollama server started. "
                "Try running `ollama serve` manually in another terminal, "
                "then run `researchgenie` again.[/yellow]"
            )
            return False
    console.print(f"[{OK}]✓[/{OK}] Ollama server is running")

    if await om.is_model_installed(model):
        console.print(f"[{OK}]✓[/{OK}] Model {model!r} already installed")
        return True

    console.print(f"[{MUTED}]Downloading {model!r} — this can take a while "
                  "the first time...[/grey62]")
    last_status = {"text": ""}

    def report(status: str, completed: int, total: int) -> None:
        if status != last_status["text"]:
            last_status["text"] = status
            if total:
                pct = completed / total * 100
                console.print(f"  [{MUTED}]{status} ({pct:.0f}%)[/{MUTED}]", end="\r")
            else:
                console.print(f"  [{MUTED}]{status}[/{MUTED}]")

    success = await om.pull_model(model, on_progress=report)
    if success:
        console.print(f"\n[{OK}]✓[/{OK}] Model ready")
        return True
    console.print(
        f"\n[{WARN}]Model download did not complete. Try `ollama pull {model}` "
        "manually, then run `researchgenie` again.[/yellow]"
    )
    return False


def _setup_gemini(console: Console) -> str | None:
    console.print(Panel(
        "[bold]Gemini API Key Required[/bold]\n\n"
        "Get your key from Google AI Studio:\n"
        f"[{ACCENT}]https://aistudio.google.com/apikey[/{ACCENT}]\n\n"
        "Your key is stored locally on this device only — never committed, "
        "logged, or included in generated research output.",
        title="Gemini Setup", border_style=ACCENT,
    ))
    key = Prompt.ask("Paste your Gemini API key", password=True)
    return key.strip() or None


# Priority order (#5): (config field, prompt, one-line reason with the link, live check name).
# No Semantic Scholar key — the project will not get an S2 plan.
_SCHOLARLY_STEPS = (
    ("openalex_api_key", "OpenAlex API key",
     "Free and instant; raises your daily searches from about 100 to about 1,000. "
     f"Get one at {scholarly_setup.OPENALEX_KEY_URL}",
     "check_openalex_key"),
    ("contact_email", "Contact email",
     "Puts your searches in Crossref's faster 'polite pool', and NCBI requires one. "
     "Only the providers see it.",
     "check_contact_email"),
    ("ncbi_api_key", "NCBI (PubMed) API key",
     "Free and instant; lets PubMed searches run at 10 instead of 3 requests a second. "
     f"Get one at {scholarly_setup.NCBI_KEY_URL}",
     "check_ncbi_key"),
)


def setup_scholarly_access(console: Console, config: Config) -> None:
    """Asks for the three optional scholarly values, checks each live, and
    stores them on `config`. Enter skips (or keeps the saved value)."""
    console.print(Panel(
        "[bold]Scholarly access (optional, about two minutes)[/bold]\n\n"
        "Three free values make literature searches faster and more reliable. "
        "Press Enter to skip any of them. They are stored only on this device.",
        title="Search sources", border_style=ACCENT,
    ))
    for field, label, reason, check_name in _SCHOLARLY_STEPS:
        saved = getattr(config, field)
        console.print(f"\n[bold]{label}[/bold] — [{MUTED}]{reason}[/{MUTED}]")
        keep = " (Enter keeps the saved one)" if saved else " (Enter to skip)"
        while True:
            value = (Prompt.ask(f"{label}{keep}", password=field != "contact_email",
                                default="", show_default=False) or "").strip()
            if not value:
                break
            ok, message = getattr(scholarly_setup, check_name)(value)
            console.print(f"[{OK if ok else WARN}]{message}[/{OK if ok else WARN}]")
            if ok is not False:
                setattr(config, field, value)
                break


def run_setup_wizard(console: Console | None = None) -> Config:
    """Runs the interactive wizard and returns the resulting (already saved)
    Config: the AI engine, then the optional scholarly keys and email (#5).
    Semantic Scholar stays keyless (no S2 plan)."""
    console = console or Console()
    console.print(Panel.fit(
        "[bold]✦ ResearchGenie Setup[/bold]\n"
        "A few quick questions — this only happens once.",
        border_style=ACCENT,
    ))

    provider = _choose_provider(console)
    previous = load()
    config = Config(provider=provider, openalex_api_key=previous.openalex_api_key,
                    contact_email=previous.contact_email, ncbi_api_key=previous.ncbi_api_key)

    if provider == "ollama":
        config.ollama_model = DEFAULT_OLLAMA_MODEL
        ready = asyncio.run(_setup_ollama(console, config.ollama_model))
        config.setup_complete = ready
    else:
        api_key = _setup_gemini(console)
        config.gemini_api_key = api_key
        config.setup_complete = api_key is not None

    if config.setup_complete:
        setup_scholarly_access(console, config)
    save(config)
    if config.setup_complete:
        console.print(f"\n[{OK}]✓ Setup complete.[/{OK}]\n")
    return config
