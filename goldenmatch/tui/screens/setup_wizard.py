"""Setup wizard — guides users through GPU, API, and database configuration."""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Center, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static, RadioSet, RadioButton


class SetupWizard(Screen):
    """Interactive setup wizard that walks users through configuration."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    CSS = """
    #wizard-card {
        width: 80;
        height: auto;
        max-height: 38;
        background: #16213e;
        border: solid #d4a017;
        padding: 2 4;
    }
    .wizard-title {
        color: #d4a017;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    .wizard-step {
        color: #8892a0;
        text-align: center;
        margin-bottom: 1;
    }
    .wizard-label {
        color: #f0f0f0;
        margin-bottom: 0;
    }
    .wizard-hint {
        color: #8892a0;
        margin-bottom: 1;
    }
    .wizard-status {
        margin-top: 1;
    }
    .wizard-buttons {
        align: center middle;
        height: 3;
        margin-top: 1;
    }
    .status-ok { color: #2ecc71; }
    .status-missing { color: #e67e22; }
    .status-error { color: #e74c3c; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step = 0
        self._config = {
            "gpu_mode": "auto",
            "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "gpu_endpoint": os.environ.get("GOLDENMATCH_GPU_ENDPOINT", ""),
            "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "db_url": os.environ.get("GOLDENMATCH_DATABASE_URL", ""),
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="wizard-card"):
                yield Static(
                    "[bold #d4a017]GoldenMatch Setup Wizard[/]",
                    classes="wizard-title",
                )
                yield Static("", id="step-indicator", classes="wizard-step")
                yield Static("", id="step-content")
                yield Static("", id="step-status", classes="wizard-status")
                yield Static("")
                with Center(classes="wizard-buttons"):
                    yield Button("Back", id="btn-back")
                    yield Button("Next", variant="primary", id="btn-next")
                    yield Button("Skip", id="btn-skip")
        yield Footer()

    def on_mount(self) -> None:
        self._show_step()

    def _show_step(self) -> None:
        steps = [
            self._step_welcome,
            self._step_gpu,
            self._step_gpu_config,
            self._step_llm,
            self._step_database,
            self._step_summary,
        ]
        total = len(steps)
        indicator = self.query_one("#step-indicator", Static)
        indicator.update(f"[#8892a0]Step {self._step + 1} of {total}[/]")

        if self._step < total:
            steps[self._step]()

        # Hide back on first step
        back_btn = self.query_one("#btn-back", Button)
        back_btn.display = self._step > 0

        # Change next to "Finish" on last step
        next_btn = self.query_one("#btn-next", Button)
        next_btn.label = "Finish" if self._step == total - 1 else "Next"

    def _step_welcome(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        # Detect current state
        checks = []
        gcp = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        endpoint = os.environ.get("GOLDENMATCH_GPU_ENDPOINT", "")
        anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
        db = os.environ.get("GOLDENMATCH_DATABASE_URL", "")

        if gcp:
            checks.append(f"  [#2ecc71]✓[/] Vertex AI: project [bold]{gcp}[/]")
        elif endpoint:
            checks.append(f"  [#2ecc71]✓[/] Remote GPU: [bold]{endpoint}[/]")
        else:
            checks.append(f"  [#e67e22]○[/] GPU/Embeddings: not configured")

        if anthropic:
            checks.append(f"  [#2ecc71]✓[/] LLM Boost: API key set")
        else:
            checks.append(f"  [#8892a0]○[/] LLM Boost: not configured (optional)")

        if db:
            checks.append(f"  [#2ecc71]✓[/] Database: configured")
        else:
            checks.append(f"  [#8892a0]○[/] Database: not configured (optional)")

        content.update(
            "[bold]Welcome![/]\n\n"
            "This wizard helps you configure GoldenMatch.\n"
            "Each step is optional — skip what you don't need.\n\n"
            "[bold #d4a017]Current Status:[/]\n" +
            "\n".join(checks)
        )
        status.update("")

    def _step_gpu(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        content.update(
            "[bold]GPU / Embedding Mode[/]\n\n"
            "How should GoldenMatch compute embeddings?\n\n"
            "[bold #d4a017]A[/] [bold]Vertex AI[/] (recommended)\n"
            "  Google Cloud managed API. Best accuracy, no GPU needed.\n"
            "  Cost: ~$0.025 per 1K texts.\n\n"
            "[bold #d4a017]B[/] [bold]Remote Endpoint[/]\n"
            "  Your own GPU server or Colab notebook.\n"
            "  Free with Colab.\n\n"
            "[bold #d4a017]C[/] [bold]Local GPU[/]\n"
            "  Uses CUDA/MPS if available.\n\n"
            "[bold #d4a017]D[/] [bold]CPU-Safe[/]\n"
            "  No embeddings. Uses fuzzy string matching only.\n\n"
            "Press [bold]A[/], [bold]B[/], [bold]C[/], or [bold]D[/] to select."
        )
        status.update("")

    def _step_gpu_config(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        mode = self._config["gpu_mode"]

        if mode == "vertex":
            project = self._config.get("gcp_project", "")
            content.update(
                "[bold]Vertex AI Configuration[/]\n\n"
                "You need a Google Cloud project with Vertex AI enabled.\n\n"
                "[bold #d4a017]Setup steps:[/]\n"
                "  1. Create project: [link]console.cloud.google.com/projectcreate[/link]\n"
                "  2. Enable API:\n"
                "     [dim]gcloud services enable aiplatform.googleapis.com[/dim]\n"
                "  3. Authenticate:\n"
                "     [dim]gcloud auth application-default login[/dim]\n"
                "  4. Set environment variable:\n"
                "     [dim]export GOOGLE_CLOUD_PROJECT=your-project-id[/dim]\n\n"
                f"Current project: [bold]{project or '(not set)'}[/]"
            )
            if project:
                status.update("[#2ecc71]✓ Project configured[/]")
            else:
                status.update("[#e67e22]Set GOOGLE_CLOUD_PROJECT and restart[/]")

        elif mode == "remote":
            endpoint = self._config.get("gpu_endpoint", "")
            content.update(
                "[bold]Remote GPU Configuration[/]\n\n"
                "[bold #d4a017]Option 1: Google Colab (free)[/]\n"
                "  Upload [bold]scripts/gpu_colab_notebook.ipynb[/] to Colab\n"
                "  Set runtime to GPU, run all cells, copy the URL\n\n"
                "[bold #d4a017]Option 2: Self-hosted[/]\n"
                "  Run on any GPU machine:\n"
                "  [dim]python scripts/gpu_endpoint.py --port 8090[/dim]\n\n"
                "Set environment variable:\n"
                "  [dim]export GOLDENMATCH_GPU_ENDPOINT=http://your-server:8090[/dim]\n\n"
                f"Current endpoint: [bold]{endpoint or '(not set)'}[/]"
            )
            if endpoint:
                status.update("[#2ecc71]✓ Endpoint configured[/]")
            else:
                status.update("[#e67e22]Set GOLDENMATCH_GPU_ENDPOINT and restart[/]")

        elif mode == "local":
            content.update(
                "[bold]Local GPU[/]\n\n"
                "GoldenMatch will use your local GPU automatically.\n\n"
                "Requirements:\n"
                "  [dim]pip install goldenmatch[embeddings][/dim]\n"
                "  CUDA or Apple MPS must be available.\n"
            )
            try:
                import torch
                if torch.cuda.is_available():
                    status.update(f"[#2ecc71]✓ CUDA available: {torch.cuda.get_device_name(0)}[/]")
                else:
                    status.update("[#e67e22]No CUDA detected. Install GPU drivers.[/]")
            except Exception:
                status.update("[#e67e22]torch not installed. Run: pip install goldenmatch[embeddings][/]")

        else:  # cpu_safe
            content.update(
                "[bold]CPU-Safe Mode[/]\n\n"
                "Embedding features are disabled.\n"
                "GoldenMatch will use lightweight scorers:\n"
                "  exact, jaro_winkler, levenshtein,\n"
                "  token_sort, soundex_match, ensemble\n\n"
                "No additional configuration needed."
            )
            status.update("[#2ecc71]✓ Ready[/]")

    def _step_llm(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

        content.update(
            "[bold]LLM Boost (Optional)[/]\n\n"
            "Use an LLM to label training pairs and fine-tune\n"
            "the embedding model for your specific data.\n\n"
            "Cost: ~$0.30 per dataset. Only runs once.\n\n"
            "[bold #d4a017]Setup:[/]\n"
            "  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]\n"
            "  or\n"
            "  [dim]export OPENAI_API_KEY=sk-...[/dim]\n\n"
            "Then run with:\n"
            "  [dim]goldenmatch dedupe file.csv --llm-boost[/dim]\n"
        )

        if has_key:
            status.update("[#2ecc71]✓ API key configured[/]")
        else:
            status.update("[#8892a0]○ No API key set (skip if not needed)[/]")

    def _step_database(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        db_url = os.environ.get("GOLDENMATCH_DATABASE_URL", "")

        content.update(
            "[bold]Database Sync (Optional)[/]\n\n"
            "Match new records against a live Postgres database.\n\n"
            "[bold #d4a017]Setup:[/]\n"
            "  [dim]pip install goldenmatch[postgres][/dim]\n"
            "  [dim]export GOLDENMATCH_DATABASE_URL=postgresql://user:pass@host/db[/dim]\n\n"
            "Then run:\n"
            "  [dim]goldenmatch sync --table customers[/dim]\n\n"
            "Or watch continuously:\n"
            "  [dim]goldenmatch watch --table customers --interval 30[/dim]\n"
        )

        if db_url:
            status.update(f"[#2ecc71]✓ Database configured[/]")
        else:
            status.update("[#8892a0]○ No database configured (skip if not needed)[/]")

    def _step_summary(self) -> None:
        content = self.query_one("#step-content", Static)
        status = self.query_one("#step-status", Static)

        lines = ["[bold]Setup Summary[/]\n"]

        # GPU
        mode = self._config["gpu_mode"]
        if mode == "vertex":
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "(not set)")
            lines.append(f"  [#d4a017]GPU:[/] Vertex AI (project: {project})")
        elif mode == "remote":
            endpoint = os.environ.get("GOLDENMATCH_GPU_ENDPOINT", "(not set)")
            lines.append(f"  [#d4a017]GPU:[/] Remote ({endpoint})")
        elif mode == "local":
            lines.append(f"  [#d4a017]GPU:[/] Local")
        else:
            lines.append(f"  [#d4a017]GPU:[/] CPU-safe (no embeddings)")

        # LLM
        if os.environ.get("ANTHROPIC_API_KEY"):
            lines.append(f"  [#d4a017]LLM:[/] Anthropic (configured)")
        elif os.environ.get("OPENAI_API_KEY"):
            lines.append(f"  [#d4a017]LLM:[/] OpenAI (configured)")
        else:
            lines.append(f"  [#d4a017]LLM:[/] Not configured")

        # DB
        if os.environ.get("GOLDENMATCH_DATABASE_URL"):
            lines.append(f"  [#d4a017]Database:[/] Configured")
        else:
            lines.append(f"  [#d4a017]Database:[/] Not configured")

        lines.append("\n[bold]To save these settings permanently,[/]")
        lines.append("add the environment variables to your shell profile")
        lines.append("or create a [bold].env[/] file (see [bold].env.example[/]).")

        content.update("\n".join(lines))
        status.update("[#2ecc71]Press Finish to start using GoldenMatch.[/]")

    # ── Event Handlers ────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-next":
            self._next_step()
        elif event.button.id == "btn-back":
            self._prev_step()
        elif event.button.id == "btn-skip":
            self._next_step()

    def on_key(self, event) -> None:
        if self._step == 1:  # GPU selection step
            key = event.key.lower() if hasattr(event, 'key') else ""
            if key == "a":
                self._config["gpu_mode"] = "vertex"
                self._next_step()
            elif key == "b":
                self._config["gpu_mode"] = "remote"
                self._next_step()
            elif key == "c":
                self._config["gpu_mode"] = "local"
                self._next_step()
            elif key == "d":
                self._config["gpu_mode"] = "cpu_safe"
                self._next_step()

    def _next_step(self) -> None:
        total = 6
        if self._step >= total - 1:
            self.dismiss("done")
        else:
            self._step += 1
            self._show_step()

    def _prev_step(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._show_step()

    def action_cancel(self) -> None:
        self.dismiss("cancel")
