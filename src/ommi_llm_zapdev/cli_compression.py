"""
CLI commands for model compression.

Adds `ommi compress` subcommand with various compression options.
"""

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .compression.compressor import ModelCompressor

console = Console()


app = typer.Typer(help="Compress and optimize models")


@app.command(name="quantize")
def quantize_cmd(
    model: str = typer.Argument(..., help="HuggingFace model ID or local path"),
    output: Path = typer.Argument(..., help="Output directory"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (4 or 8)", min=4, max=8),
    nested: bool = typer.Option(
        True, "--nested/--no-nested", help="Use nested quantization (4-bit only)"
    ),
):
    """
    Quantize model to 4-bit or 8-bit.

    Reduces model size significantly:
    - 4-bit: ~75% smaller (4x reduction)
    - 8-bit: ~50% smaller (2x reduction)

    Example:
        ommi compress quantize meta-llama/Llama-2-7b-chat-hf ./llama-7b-4bit --bits 4
    """
    compressor = ModelCompressor(model)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task(f"Quantizing to {bits}-bit...", total=None)

        try:
            if bits == 4:
                result = compressor.quantize_4bit(output, nested_quantization=nested)
            elif bits == 8:
                result = compressor.quantize_8bit(output)
            else:
                console.print("[red]Only 4-bit and 8-bit quantization supported[/]")
                raise typer.Exit(1)

            progress.update(task, completed=True)

            console.print(f"[green]✓ Model quantized and saved to {result}[/]")

            # Show how to use it
            console.print(f"\n[dim]To use this model:[/]")
            console.print(f"[dim]  ommi load {result}[/]")
            console.print(f'[dim]  ommi generate {result} "Your prompt"[/]')

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            raise typer.Exit(1)


@app.command(name="prune")
def prune_cmd(
    model: str = typer.Argument(..., help="Model to prune"),
    output: Path = typer.Argument(..., help="Output directory"),
    layers: int = typer.Option(4, "--layers", "-l", help="Number of layers to remove", min=1),
    strategy: str = typer.Option(
        "last", "--strategy", "-s", help="Pruning strategy: last/first/alternating"
    ),
):
    """
    Remove transformer layers to reduce model size.

    Removing layers makes the model smaller and faster but may reduce quality.

    Example:
        # Remove last 4 layers from a 32-layer model (keeping 28)
        ommi compress prune meta-llama/Llama-2-7b-chat-hf ./llama-7b-28l --layers 4
    """
    compressor = ModelCompressor(model)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Pruning layers...", total=None)

        try:
            result = compressor.prune_layers(layers, output, strategy)
            progress.update(task, completed=True)

            console.print(f"[green]✓ Pruned model saved to {result}[/]")
            console.print(f"[yellow]⚠ Note: Pruned models may have reduced quality[/]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            raise typer.Exit(1)


@app.command(name="export")
def export_cmd(
    model: str = typer.Argument(..., help="Model to export"),
    output: Path = typer.Argument(..., help="Output file/directory"),
    format: str = typer.Option("gguf", "--format", "-f", help="Export format: gguf/onnx"),
    quant: str = typer.Option(
        "Q4_K_M", "--quant", "-q", help="GGUF quantization (Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0)"
    ),
):
    """
    Export model to different formats.

    Supports:
    - GGUF: For llama.cpp (CPU inference, edge devices)
    - ONNX: For cross-platform deployment (TensorRT, CoreML)

    Example:
        ommi compress export meta-llama/Llama-2-7b-chat-hf ./model.gguf --format gguf --quant Q4_K_M
    """
    compressor = ModelCompressor(model)

    try:
        if format == "gguf":
            result = compressor.export_gguf(output, quantization=quant)
            console.print(f"[green]✓ Exported to GGUF: {result}[/]")
            console.print(f'[dim]Use with llama.cpp: ./main -m {result} -p "Hello"[/]')

        elif format == "onnx":
            result = compressor.export_onnx(output)
            console.print(f"[green]✓ Exported to ONNX: {result}[/]")

        else:
            console.print(f"[red]Unknown format: {format}. Use 'gguf' or 'onnx'[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


@app.command(name="compare")
def compare_cmd(
    original: str = typer.Argument(..., help="Original model"),
    variants: List[str] = typer.Argument(..., help="Compressed model paths to compare"),
):
    """
    Compare sizes of original vs compressed models.

    Example:
        ommi compress compare meta-llama/Llama-2-7b-chat-hf ./llama-4bit ./llama-8bit ./llama-pruned
    """
    compressor = ModelCompressor(original)

    table = Table(title="Model Size Comparison")
    table.add_column("Variant", style="cyan")
    table.add_column("Size (GB)", justify="right", style="green")
    table.add_column("Reduction", justify="right", style="yellow")

    # Get original size
    orig_size = compressor._get_model_size(original)
    table.add_row("Original (FP16)", f"{orig_size:.2f}", "0%")

    # Compare variants
    for i, variant in enumerate(variants, 1):
        variant_path = Path(variant)
        if variant_path.exists():
            size = compressor._get_dir_size(variant_path)
            reduction = (1 - size / orig_size) * 100
            table.add_row(f"Variant {i} ({variant_path.name})", f"{size:.2f}", f"{reduction:.1f}%")
        else:
            table.add_row(f"Variant {i} ({variant})", "Not found", "-")

    console.print(table)


@app.command(name="auto")
def auto_compress(
    model: str = typer.Argument(..., help="Model to compress"),
    output_dir: Path = typer.Option(
        Path("./compressed-models"), "--output", "-o", help="Output directory"
    ),
    target_size: float = typer.Option(
        None, "--target-size", "-t", help="Target size in GB (e.g., 4.0)"
    ),
    quality: str = typer.Option(
        "balanced", "--quality", "-q", help="Quality priority: speed/balanced/quality"
    ),
):
    """
    Automatically compress model to target size.

    Picks the best compression strategy based on your constraints.

    Example:
        # Compress to fit on 4GB GPU
        ommi compress auto meta-llama/Llama-2-7b-chat-hf --target-size 4.0

        # Prioritize speed over quality
        ommi compress auto meta-llama/Llama-2-13b-chat-hf --quality speed
    """
    console.print(f"[blue]Analyzing {model}...[/]")

    compressor = ModelCompressor(model)
    orig_size = compressor._get_model_size(model)

    console.print(f"Original size: {orig_size:.2f} GB")

    if target_size:
        target_reduction = (1 - target_size / orig_size) * 100
        console.print(f"Target: {target_size:.2f} GB ({target_reduction:.1f}% reduction)")

    # Determine strategy
    if quality == "speed" or (target_size and target_size < orig_size * 0.25):
        # Aggressive compression
        console.print("[yellow]Using aggressive compression (4-bit)...[/]")
        output = output_dir / f"{Path(model).name}-4bit"
        compressor.quantize_4bit(output)

    elif quality == "balanced" or (target_size and target_size < orig_size * 0.5):
        # Moderate compression
        console.print("[yellow]Using balanced compression (8-bit)...[/]")
        output = output_dir / f"{Path(model).name}-8bit"
        compressor.quantize_8bit(output)

    else:
        # Quality priority - try 8-bit first
        console.print("[yellow]Using quality-preserving compression (8-bit)...[/]")
        output = output_dir / f"{Path(model).name}-8bit"
        compressor.quantize_8bit(output)

    final_size = compressor._get_dir_size(output)
    console.print(
        f"\n[green]✓ Compressed to {final_size:.2f} GB ({(1 - final_size / orig_size) * 100:.1f}% smaller)[/]"
    )
    console.print(f"Saved to: {output}")


# Main entry point for the compress subcommand
def compress_main():
    """Entry point for `ommi compress` subcommand."""
    app()
