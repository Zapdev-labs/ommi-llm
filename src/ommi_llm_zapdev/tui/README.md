# Ommi LLM TUI

A modern terminal user interface for Ommi LLM built with [OpenTUI](https://opentui.com).

## Features

- **🖥️ Model Manager**: Browse, download, and manage HuggingFace models
- **💬 Interactive Chat**: Chat with downloaded models in a beautiful interface
- **🧠 Memory Monitor**: Real-time system and GPU memory statistics
- **📦 Model Sharding**: Pre-shard models for faster layer-wise loading
- **⌨️ Keyboard Navigation**: Vim-style keybindings for power users

## Screenshots

The TUI features a dark, modern interface with:
- Clean, bordered panels with rounded corners
- Color-coded sections (primary teal, secondary purple, accent coral)
- Smooth keyboard navigation
- Real-time memory statistics display

## Requirements

- [Bun](https://bun.sh) runtime (for running the TUI)
- Python 3.10+ (for the underlying ommi-llm package)

## Installation

The TUI is bundled with `ommi-llm`. To use it:

```bash
# Make sure you have Bun installed
curl -fsSL https://bun.sh/install | bash

# Install ommi-llm
pip install ommi-llm

# Launch the TUI
ommi tui
```

## Usage

### Launching

```bash
# Via the CLI command
ommi tui

# Or directly via Python
python -m ommi_llm_zapdev.tui_launcher
```

### Navigation

| Key | Action |
|-----|--------|
| `↑` / `↓` or `k` / `j` | Navigate menus |
| `Enter` | Select item |
| `ESC` | Go back to previous screen |
| `Ctrl+C` | Exit application |

### Screens

#### Home Screen
The main menu provides access to all features:
- Model Manager
- Chat
- Memory
- Shard Model
- Help

#### Model Manager
- Search models on HuggingFace
- View downloaded models
- Download new models
- Remove existing models

#### Chat
- Interactive conversation with loaded models
- View chat history in a scrollable pane
- Type messages at the bottom input field

#### Memory
- System RAM statistics (total, available, usage %)
- GPU VRAM statistics (if CUDA available)
- Information about layer-wise memory usage

#### Shard Model
- Pre-shard models for faster inference
- Supports 4-bit and 8-bit quantization
- Custom output directory selection

## Development

### Structure

```
tui/
├── index.ts          # Main TUI application
├── package.json      # Node.js dependencies
├── tsconfig.json     # TypeScript configuration
└── README.md         # This file
```

### Running in Development Mode

```bash
cd src/ommi_llm_zapdev/tui
bun install
bun run dev
```

### Building

```bash
cd src/ommi_llm_zapdev/tui
bun run build
```

## Architecture

The TUI is built with OpenTUI, a Zig-based terminal UI framework with TypeScript bindings. It communicates with the Python backend via subprocess calls, allowing it to leverage the full ommi-llm inference engine while providing a modern, interactive interface.

### Communication

The TUI launches Python commands to perform operations:
- Model downloads via `ommi download`
- Model loading via `ommi load`
- Memory stats via `ommi memory`
- Text generation via `ommi generate`

This architecture keeps the TUI lightweight while providing full access to all ommi-llm features.

## License

MIT License - same as ommi-llm
