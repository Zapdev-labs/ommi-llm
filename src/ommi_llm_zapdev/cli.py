"""
CLI interface for ommi llm.
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .core.auto_model import AutoModel
from .persistence.sharder import ModelSharder
from .utils.memory import MemoryManager
from .cli_compression import app as compress_app

app = typer.Typer(
    name="ommi", help="Ommi LLM - Run 70B+ models on consumer GPUs", no_args_is_help=True
)

# Add compress subcommand
app.add_typer(compress_app, name="compress", help="Compress and optimize models")
console = Console()

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ommi-llm"


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded models."""
    cache_dir = Path(os.environ.get("OMMI_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]ommi-llm[/] version [green]{__version__}[/]")


@app.command(name="download")
def download_model(
    model_id: str = typer.Argument(
        ..., help="HuggingFace model ID (e.g., meta-llama/Llama-2-7b-chat-hf)"
    ),
    local_dir: Optional[Path] = typer.Option(
        None, "--local-dir", "-o", help="Local directory to save model"
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", help="Cache directory for downloads"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="HuggingFace API token (for gated models)"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume incomplete downloads"),
):
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    # Use default cache dir if not specified
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Determine output directory
    if local_dir:
        output_path = local_dir
    else:
        output_path = cache_dir / model_id.replace("/", "--")

    output_path.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold blue]Downloading Model[/]\n"
            f"Model: [cyan]{model_id}[/]\n"
            f"Output: [green]{output_path}[/]",
            title="ommi-llm",
            border_style="blue",
        )
    )

    try:
        # Download with progress
        console.print("[yellow]Starting download...[/]")

        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path) if local_dir else None,
            cache_dir=str(cache_dir) if not local_dir else None,
            resume_download=resume,
            token=token,
            local_files_only=False,
        )

        console.print(f"[green]✓ Model downloaded successfully![/]")
        console.print(f"[dim]Location: {downloaded_path}[/]")

        # Show model info
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(downloaded_path)

            table = Table(title="Model Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Model ID", model_id)
            table.add_row("Architecture", str(getattr(config, "architectures", ["Unknown"])[0]))
            table.add_row("Model Type", getattr(config, "model_type", "Unknown"))
            table.add_row("Hidden Size", str(getattr(config, "hidden_size", "Unknown")))
            table.add_row("Num Layers", str(getattr(config, "num_hidden_layers", "Unknown")))
            table.add_row(
                "Num Attention Heads", str(getattr(config, "num_attention_heads", "Unknown"))
            )

            console.print(table)
        except Exception:
            pass

        console.print(f"\n[bold]To use this model:[/]")
        console.print(f"[dim]ommi load {downloaded_path}[/] or")
        console.print(f'[dim]ommi generate {downloaded_path} "Your prompt"[/]')

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/]")
        raise typer.Exit(1)


@app.command(name="search")
def search_models(
    query: str = typer.Argument(..., help="Search query (e.g., 'llama 70b', 'mistral 7b')"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    filter_tag: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Filter by tag (e.g., 'text-generation')"
    ),
):
    """Search for models on HuggingFace Hub."""
    from huggingface_hub import list_models

    console.print(f"[blue]Searching for: {query}[/]")

    try:
        models = list_models(
            search=query,
            limit=limit,
            sort="downloads",
            direction=-1,
            filter=filter_tag,
        )

        table = Table(title=f"Search Results: '{query}'")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Downloads", style="green", justify="right")
        table.add_column("Tags", style="yellow", max_width=40)

        model_list = list(models)
        if not model_list:
            console.print("[yellow]No models found.[/]")
            return

        for model in model_list:
            tags = ", ".join(model.tags[:3]) if model.tags else ""
            downloads = f"{model.downloads:,}" if model.downloads else "0"
            table.add_row(model.id, downloads, tags)

        console.print(table)

        console.print(f"\n[dim]To download a model: ommi download <model_id>[/]")

    except Exception as e:
        console.print(f"[red]Error searching models: {e}[/]")
        raise typer.Exit(1)


@app.command(name="list")
def list_models():
    """List downloaded models in cache."""
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        console.print("[yellow]No models downloaded yet.[/]")
        return

    # Find all model directories
    models = []
    for item in cache_dir.iterdir():
        if item.is_dir():
            # Check if it's a model directory (contains config.json)
            config_file = item / "config.json"
            if config_file.exists():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_gb = size / (1024**3)
                models.append((item.name, size_gb, item))

    if not models:
        console.print("[yellow]No models found in cache.[/]")
        console.print(f"[dim]Cache directory: {cache_dir}[/]")
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Location", style="dim")

    for name, size_gb, path in sorted(models, key=lambda x: x[1], reverse=True):
        table.add_row(name, f"{size_gb:.2f} GB", str(path))

    console.print(table)
    console.print(f"\n[dim]Cache directory: {cache_dir}[/]")
    console.print(f"[dim]Total models: {len(models)}[/]")


@app.command(name="rm")
def remove_model(
    model_name: str = typer.Argument(..., help="Model name or path to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a downloaded model from cache."""
    cache_dir = get_cache_dir()

    # Try to find the model
    model_path = cache_dir / model_name
    if not model_path.exists():
        # Try with -- separator
        model_path = cache_dir / model_name.replace("/", "--")

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_name}[/]")
        console.print(f"[dim]Use 'ommi list' to see downloaded models[/]")
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"Remove {model_path}? This cannot be undone.")
        if not confirm:
            console.print("[yellow]Cancelled.[/]")
            return

    try:
        import shutil

        shutil.rmtree(model_path)
        console.print(f"[green]✓ Removed {model_name}[/]")
    except Exception as e:
        console.print(f"[red]Error removing model: {e}[/]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]ommi-llm[/] version [green]{__version__}[/]")


@app.command()
def load(
    model: str = typer.Argument(..., help="HuggingFace model ID or local path"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu/mps)"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type (float16/bfloat16/float32)"),
    prefetching: bool = typer.Option(
        True, "--prefetching/--no-prefetching", help="Enable async prefetching"
    ),
    compression: Optional[str] = typer.Option(
        None, "--compression", "-c", help="Quantization (4bit/8bit)"
    ),
):
    """Load a model for inference."""
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            model = AutoModel.from_pretrained(
                model, device=device, dtype=dtype, prefetching=prefetching, compression=compression
            )

            progress.update(task, completed=True)

            # Show model info
            info = model.get_model_info()
            table = Table(title="Model Loaded")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in info.items():
                table.add_row(key, str(value))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/]")
            raise typer.Exit(1)


@app.command()
def generate(
    model: str = typer.Argument(..., help="HuggingFace model ID or local path"),
    prompt: str = typer.Argument(..., help="Input prompt"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-n", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
):
    """Generate text from a prompt."""
    console.print(f"[blue]Loading model: {model}[/]")

    try:
        llm = AutoModel.from_pretrained(model, device=device)

        console.print(f"[blue]Generating with prompt: {prompt[:50]}...[/]")

        # Tokenize
        input_tokens = llm.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=4096,
        )

        # Generate
        import torch

        with torch.no_grad():
            output_ids = llm.generate(
                input_tokens["input_ids"].to(llm.device),
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        # Decode
        output_text = llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        console.print("[green]Output:[/]")
        console.print(output_text)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


@app.command()
def shard(
    model: str = typer.Argument(..., help="Model to shard"),
    output: Path = typer.Argument(..., help="Output directory"),
    compression: Optional[str] = typer.Option(
        None, "--compression", "-c", help="Quantization mode"
    ),
    delete_original: bool = typer.Option(
        False, "--delete-original", help="Delete original after sharding"
    ),
):
    """Shard a model into layer-wise files."""
    console.print(f"[blue]Sharding model: {model}[/]")
    console.print(f"[blue]Output: {output}[/]")

    try:
        sharder = ModelSharder(model, output, compression=compression)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Sharding model...", total=None)

            shard_paths = sharder.shard_model(delete_original=delete_original)

            progress.update(task, completed=True)

        console.print(f"[green]Model sharded into {len(shard_paths)} layers[/]")

    except Exception as e:
        console.print(f"[red]Error sharding model: {e}[/]")
        raise typer.Exit(1)


@app.command()
def memory():
    """Show memory usage statistics."""
    mm = MemoryManager("cuda")
    stats = mm.get_memory_stats()

    table = Table(title="Memory Statistics")
    table.add_column("Type", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    # RAM stats
    ram = stats["ram"]
    table.add_row("RAM", "Total", f"{ram['total']:.2f} GB")
    table.add_row("RAM", "Available", f"{ram['available']:.2f} GB")
    table.add_row("RAM", "Used %", f"{ram['percent']:.1f}%")

    # VRAM stats
    if "vram" in stats:
        vram = stats["vram"]
        table.add_row("VRAM", "Total", f"{vram['total']:.2f} GB")
        table.add_row("VRAM", "Allocated", f"{vram['allocated']:.2f} GB")
        table.add_row("VRAM", "Reserved", f"{vram['reserved']:.2f} GB")
        table.add_row("VRAM", "Free", f"{vram['free']:.2f} GB")

    console.print(table)


@app.command()
def list_architectures():
    """List supported model architectures."""
    from .utils.constants import SUPPORTED_ARCHITECTURES

    table = Table(title="Supported Architectures")
    table.add_column("Architecture", style="cyan")

    for arch in SUPPORTED_ARCHITECTURES:
        table.add_row(arch)

    console.print(table)


@app.command()
def tui():
    """Launch the interactive TUI (Terminal User Interface)."""
    from .tui_launcher import launch_tui

    console.print(
        Panel.fit(
            "[bold blue]Launching Ommi LLM TUI...[/]\n"
            "[dim]A modern terminal interface for managing models[/]",
            title="ommi-llm",
            border_style="blue",
        )
    )

    exit_code = launch_tui()
    raise typer.Exit(exit_code)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
