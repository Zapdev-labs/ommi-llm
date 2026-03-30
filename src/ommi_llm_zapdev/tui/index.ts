/**
 * Ommi LLM TUI - Terminal User Interface
 * 
 * A modern TUI for managing and running large language models
 * with layer-wise inference for consumer GPUs.
 */

import {
  createCliRenderer,
  Box,
  Text,
  Input,
  Select,
  Textarea,
  ScrollBox,
  t,
  bold,
  fg,
  bg,
  dim,
  italic,
} from "@opentui/core"
import { execSync, spawn } from "child_process"
import { homedir } from "os"
import { join } from "path"

// Types
interface ModelInfo {
  id: string
  name: string
  architecture: string
  size: string
  downloaded: boolean
}

interface MemoryStats {
  ram: {
    total: number
    available: number
    percent: number
  }
  vram?: {
    total: number
    allocated: number
    reserved: number
    free: number
  }
}

// Colors
const COLORS = {
  primary: "#00D4AA",
  secondary: "#6B7BFF",
  accent: "#FF6B6B",
  background: "#1a1a2e",
  surface: "#16213e",
  text: "#eaeaea",
  muted: "#8b8b9a",
  success: "#4CAF50",
  warning: "#FFC107",
  error: "#F44336",
}

// Run Python CLI command and return parsed result
function runPythonCommand(command: string[]): any {
  try {
    const pythonCmd = process.env.OMMI_PYTHON || "python"
    const args = ["-m", "ommi_llm_zapdev.cli", ...command, "--output", "json"]
    const result = execSync(`${pythonCmd} ${args.join(" ")}`, {
      encoding: "utf-8",
      cwd: process.cwd(),
      timeout: 30000,
    })
    return JSON.parse(result)
  } catch (error) {
    console.error("Command failed:", error)
    return { error: String(error) }
  }
}

// Run async Python command with streaming output
function runPythonAsync(
  command: string[],
  onData: (data: string) => void,
  onError: (error: string) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.env.OMMI_PYTHON || "python"
    const args = ["-m", "ommi_llm_zapdev.cli", ...command]
    
    const proc = spawn(pythonCmd, args, {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
    })

    let output = ""

    proc.stdout?.on("data", (data) => {
      const chunk = data.toString()
      output += chunk
      onData(chunk)
    })

    proc.stderr?.on("data", (data) => {
      const error = data.toString()
      onError(error)
    })

    proc.on("close", (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`Process exited with code ${code}`))
      }
    })

    proc.on("error", reject)
  })
}

// Generate a model card
function ModelCard(props: {
  model: ModelInfo
  onSelect: () => void
  isSelected: boolean
}) {
  const { model, isSelected } = props
  const borderColor = isSelected ? COLORS.primary : COLORS.surface
  const bgColor = isSelected ? COLORS.surface : COLORS.background

  return Box(
    {
      borderStyle: "rounded",
      borderColor,
      backgroundColor: bgColor,
      padding: 1,
      margin: 1,
      width: 50,
      flexDirection: "column",
      gap: 1,
    },
    Text({
      content: t`${bold(fg(COLORS.primary)(model.name))}`,
    }),
    Text({
      content: t`${fg(COLORS.muted)("Architecture: ")}${model.architecture}`,
    }),
    Text({
      content: t`${fg(COLORS.muted)("Size: ")}${model.size}`,
    }),
    Text({
      content: model.downloaded
        ? t`${fg(COLORS.success)("✓ Downloaded")}`
        : t`${fg(COLORS.muted)("Not downloaded")}`,
    })
  )
}

// Main App
async function main() {
  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    title: "Ommi LLM TUI",
  })

  // State
  let currentScreen = "home"
  let selectedModel: ModelInfo | null = null
  let models: ModelInfo[] = []
  let memoryStats: MemoryStats | null = null
  let searchQuery = ""
  let chatHistory: Array<{ role: "user" | "assistant"; content: string }> = []
  let isGenerating = false

  // Screens

  // Home Screen
  function HomeScreen() {
    const menuOptions = [
      { name: "Model Manager", description: "Download and manage models", value: "models" },
      { name: "Chat", description: "Chat with downloaded models", value: "chat" },
      { name: "Memory", description: "View system memory statistics", value: "memory" },
      { name: "Shard Model", description: "Pre-shard models for faster loading", value: "shard" },
      { name: "Help", description: "View help and documentation", value: "help" },
      { name: "Exit", description: "Exit the application", value: "exit" },
    ]

    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 2,
      },
      // Header
      Box(
        {
          flexDirection: "column",
          alignItems: "center",
          marginBottom: 2,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("⚡ Ommi LLM"))}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("Run 70B+ models on consumer GPUs")}`,
        })
      ),
      // Main menu
      Box(
        {
          borderStyle: "rounded",
          borderColor: COLORS.primary,
          backgroundColor: COLORS.surface,
          padding: 2,
        },
        Select({
          id: "main-menu",
          width: 50,
          height: 12,
          options: menuOptions,
          selectedBackgroundColor: COLORS.primary,
          selectedTextColor: COLORS.background,
          textColor: COLORS.text,
          backgroundColor: "transparent",
          focusedBackgroundColor: COLORS.surface,
        })
      ),
      // Footer
      Box(
        {
          marginTop: 2,
        },
        Text({
          content: t`${dim(fg(COLORS.muted)("Use ↑↓ to navigate, Enter to select, Ctrl+C to exit"))}`,
        })
      )
    )
  }

  // Model Manager Screen
  function ModelManagerScreen() {
    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        padding: 1,
        gap: 1,
      },
      // Header
      Box(
        {
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderStyle: "single",
          borderColor: COLORS.primary,
          padding: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("📦 Model Manager"))}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("Press ESC to go back")}`,
        })
      ),
      // Search
      Box(
        {
          flexDirection: "row",
          gap: 1,
          alignItems: "center",
        },
        Text({
          content: t`${fg(COLORS.text)("Search: ")}`,
        }),
        Input({
          id: "model-search",
          width: 40,
          placeholder: "Search models on HuggingFace...",
          backgroundColor: COLORS.surface,
          focusedBackgroundColor: COLORS.primary,
          textColor: COLORS.text,
          cursorColor: COLORS.primary,
        })
      ),
      // Models list
      Box(
        {
          flexGrow: 1,
          borderStyle: "rounded",
          borderColor: COLORS.surface,
          padding: 1,
        },
        Select({
          id: "models-list",
          width: "100%",
          height: "100%",
          options: models.map((m) => ({
            name: m.name,
            description: `${m.architecture} | ${m.size} ${m.downloaded ? "✓" : ""}`,
            value: m.id,
          })),
          selectedBackgroundColor: COLORS.primary,
          selectedTextColor: COLORS.background,
          textColor: COLORS.text,
          backgroundColor: "transparent",
          focusedBackgroundColor: COLORS.surface,
        })
      ),
      // Actions
      Box(
        {
          flexDirection: "row",
          gap: 2,
          justifyContent: "center",
          padding: 1,
        },
        Text({
          content: t`${fg(COLORS.muted)("[d] Download  [r] Remove  [Enter] Details")}`,
        })
      )
    )
  }

  // Chat Screen
  function ChatScreen() {
    if (!selectedModel) {
      return Box(
        {
          width: "100%",
          height: "100%",
          backgroundColor: COLORS.background,
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 2,
        },
        Text({
          content: t`${bold(fg(COLORS.warning)("No model selected"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("Please select a model from the Model Manager first")}`,
        }),
        Text({
          content: t`${dim(fg(COLORS.muted)("Press ESC to go back"))}`,
        })
      )
    }

    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        padding: 1,
        gap: 1,
      },
      // Header
      Box(
        {
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderStyle: "single",
          borderColor: COLORS.primary,
          padding: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("💬 Chat"))} ${fg(COLORS.muted)("| " + selectedModel.name)}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("ESC: back  Ctrl+C: exit")}`,
        })
      ),
      // Chat history
      ScrollBox(
        {
          id: "chat-history",
          flexGrow: 1,
          borderStyle: "rounded",
          borderColor: COLORS.surface,
          padding: 1,
          backgroundColor: COLORS.surface,
        },
        ...chatHistory.map((msg, i) =>
          Box(
            {
              flexDirection: "column",
              marginBottom: 1,
            },
            Text({
              content: t`${bold(
                fg(msg.role === "user" ? COLORS.primary : COLORS.secondary)(
                  msg.role === "user" ? "You:" : "Assistant:"
                )
              )}`,
            }),
            Text({
              content: msg.content,
              fg: COLORS.text,
            })
          )
        ),
        isGenerating
          ? Text({
              content: t`${italic(fg(COLORS.muted)("Generating..."))}`,
            })
          : null
      ),
      // Input
      Box(
        {
          flexDirection: "row",
          gap: 1,
          alignItems: "center",
        },
        Text({
          content: t`${fg(COLORS.primary)(">")}`,
        }),
        Input({
          id: "chat-input",
          flexGrow: 1,
          placeholder: "Type your message...",
          backgroundColor: COLORS.surface,
          focusedBackgroundColor: COLORS.surface,
          textColor: COLORS.text,
          cursorColor: COLORS.primary,
        })
      )
    )
  }

  // Memory Screen
  function MemoryScreen() {
    const stats = memoryStats || {
      ram: { total: 0, available: 0, percent: 0 },
      vram: { total: 0, allocated: 0, reserved: 0, free: 0 },
    }

    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        padding: 1,
        gap: 1,
      },
      // Header
      Box(
        {
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderStyle: "single",
          borderColor: COLORS.primary,
          padding: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("🧠 Memory Statistics"))}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("ESC: back")}`,
        })
      ),
      // RAM Stats
      Box(
        {
          borderStyle: "rounded",
          borderColor: COLORS.secondary,
          backgroundColor: COLORS.surface,
          padding: 2,
          flexDirection: "column",
          gap: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.secondary)("System RAM"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)(`Total: ${stats.ram.total.toFixed(2)} GB`)}`,
        }),
        Text({
          content: t`${fg(COLORS.text)(`Available: ${stats.ram.available.toFixed(2)} GB`)}`,
        }),
        Text({
          content: t`${fg(COLORS.text)(`Used: ${stats.ram.percent.toFixed(1)}%`)}`,
        })
      ),
      // VRAM Stats
      stats.vram
        ? Box(
            {
              borderStyle: "rounded",
              borderColor: COLORS.accent,
              backgroundColor: COLORS.surface,
              padding: 2,
              flexDirection: "column",
              gap: 1,
            },
            Text({
              content: t`${bold(fg(COLORS.accent)("GPU VRAM"))}`,
            }),
            Text({
              content: t`${fg(COLORS.text)(`Total: ${stats.vram.total.toFixed(2)} GB`)}`,
            }),
            Text({
              content: t`${fg(COLORS.text)(`Allocated: ${stats.vram.allocated.toFixed(2)} GB`)}`,
            }),
            Text({
              content: t`${fg(COLORS.text)(`Reserved: ${stats.vram.reserved.toFixed(2)} GB`)}`,
            }),
            Text({
              content: t`${fg(COLORS.text)(`Free: ${stats.vram.free.toFixed(2)} GB`)}`,
            })
          )
        : Box(
            {
              borderStyle: "rounded",
              borderColor: COLORS.muted,
              backgroundColor: COLORS.surface,
              padding: 2,
            },
            Text({
              content: t`${fg(COLORS.muted)("No GPU detected or CUDA not available")}`,
            })
          ),
      // Info
      Box(
        {
          borderStyle: "single",
          borderColor: COLORS.muted,
          padding: 1,
          marginTop: 1,
        },
        Text({
          content: t`${dim(
            fg(COLORS.muted)(
              "Layer-wise inference uses ~4GB VRAM for 70B models (1.6GB layer + 0.4GB activations)"
            )
          )}`,
        })
      )
    )
  }

  // Shard Model Screen
  function ShardScreen() {
    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        padding: 1,
        gap: 1,
      },
      // Header
      Box(
        {
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderStyle: "single",
          borderColor: COLORS.primary,
          padding: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("📦 Shard Model"))}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("ESC: back")}`,
        })
      ),
      // Model input
      Box(
        {
          flexDirection: "column",
          gap: 1,
          borderStyle: "rounded",
          borderColor: COLORS.surface,
          padding: 2,
        },
        Text({
          content: t`${fg(COLORS.text)("Model ID or path:")}`,
        }),
        Input({
          id: "shard-model-input",
          width: "100%",
          placeholder: "e.g., meta-llama/Llama-2-70b-chat",
          backgroundColor: COLORS.surface,
          focusedBackgroundColor: COLORS.primary,
          textColor: COLORS.text,
          cursorColor: COLORS.primary,
        }),
        Text({
          content: t`${fg(COLORS.text)("Output directory:")}`,
        }),
        Input({
          id: "shard-output-input",
          width: "100%",
          value: join(homedir(), ".cache", "ommi-llm", "sharded"),
          backgroundColor: COLORS.surface,
          focusedBackgroundColor: COLORS.primary,
          textColor: COLORS.text,
          cursorColor: COLORS.primary,
        }),
        Text({
          content: t`${fg(COLORS.text)("Compression (optional):")}`,
        }),
        Select({
          id: "shard-compression",
          width: 30,
          height: 4,
          options: [
            { name: "None", description: "No compression", value: "" },
            { name: "4-bit", description: "4-bit quantization", value: "4bit" },
            { name: "8-bit", description: "8-bit quantization", value: "8bit" },
          ],
          selectedIndex: 0,
          backgroundColor: COLORS.surface,
          selectedBackgroundColor: COLORS.primary,
        })
      ),
      // Actions
      Box(
        {
          flexDirection: "row",
          gap: 2,
          justifyContent: "center",
          padding: 1,
        },
        Text({
          content: t`${fg(COLORS.primary)("[Enter]")} ${fg(COLORS.text)("Start sharding")}`,
        })
      )
    )
  }

  // Help Screen
  function HelpScreen() {
    return Box(
      {
        width: "100%",
        height: "100%",
        backgroundColor: COLORS.background,
        flexDirection: "column",
        padding: 1,
        gap: 1,
      },
      // Header
      Box(
        {
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderStyle: "single",
          borderColor: COLORS.primary,
          padding: 1,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("❓ Help"))}`,
        }),
        Text({
          content: t`${fg(COLORS.muted)("ESC: back")}`,
        })
      ),
      // Content
      ScrollBox(
        {
          flexGrow: 1,
          borderStyle: "rounded",
          borderColor: COLORS.surface,
          padding: 2,
          backgroundColor: COLORS.surface,
        },
        Text({
          content: t`${bold(fg(COLORS.primary)("Ommi LLM TUI"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("Run 70B+ language models on consumer GPUs with layer-wise inference.")}`,
        }),
        Text({
          content: "",
        }),
        Text({
          content: t`${bold(fg(COLORS.secondary)("Navigation"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• ↑/↓ or k/j: Navigate menus")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Enter: Select item")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• ESC: Go back")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Ctrl+C: Exit application")}`,
        }),
        Text({
          content: "",
        }),
        Text({
          content: t`${bold(fg(COLORS.secondary)("Features"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Model Manager: Download and manage models from HuggingFace")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Chat: Interactive chat with loaded models")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Memory: Monitor system and GPU memory usage")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• Shard: Pre-shard models for faster layer-wise loading")}`,
        }),
        Text({
          content: "",
        }),
        Text({
          content: t`${bold(fg(COLORS.secondary)("Performance"))}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• 70B models: ~4GB VRAM, 35-100s per token")}`,
        }),
        Text({
          content: t`${fg(COLORS.text)("• With 4-bit quantization: ~3x speedup")}`,
        }),
        Text({
          content: "",
        }),
        Text({
          content: t`${dim(fg(COLORS.muted)("For more information, visit: https://github.com/zapdev-labs/ommi-llm"))}`,
        })
      )
    )
  }

  // Render current screen
  function render() {
    renderer.root.removeAll()

    switch (currentScreen) {
      case "home":
        renderer.root.add(HomeScreen())
        break
      case "models":
        renderer.root.add(ModelManagerScreen())
        break
      case "chat":
        renderer.root.add(ChatScreen())
        break
      case "memory":
        renderer.root.add(MemoryScreen())
        break
      case "shard":
        renderer.root.add(ShardScreen())
        break
      case "help":
        renderer.root.add(HelpScreen())
        break
    }
  }

  // Initial render
  render()

  // Keyboard navigation
  renderer.keyInput.on("keypress", (key) => {
    if (key.name === "escape") {
      if (currentScreen !== "home") {
        currentScreen = "home"
        render()
      }
    }
  })

  // Handle main menu selection
  const mainMenu = renderer.root.get("main-menu")
  if (mainMenu) {
    mainMenu.on("select", (value: string) => {
      if (value === "exit") {
        process.exit(0)
      } else if (value === "models") {
        currentScreen = "models"
        // Load models
        models = [
          {
            id: "meta-llama/Llama-2-70b-chat",
            name: "Llama 2 70B Chat",
            architecture: "Llama",
            size: "140 GB",
            downloaded: false,
          },
          {
            id: "mistralai/Mistral-7B-v0.1",
            name: "Mistral 7B",
            architecture: "Mistral",
            size: "14 GB",
            downloaded: true,
          },
        ]
        render()
      } else if (value === "chat") {
        currentScreen = "chat"
        render()
      } else if (value === "memory") {
        currentScreen = "memory"
        // Load memory stats
        memoryStats = {
          ram: {
            total: 32,
            available: 16,
            percent: 50,
          },
          vram: {
            total: 8,
            allocated: 2,
            reserved: 1,
            free: 5,
          },
        }
        render()
      } else if (value === "shard") {
        currentScreen = "shard"
        render()
      } else if (value === "help") {
        currentScreen = "help"
        render()
      }
    })
  }
}

main().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
