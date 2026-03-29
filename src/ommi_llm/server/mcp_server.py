"""
MCP (Model Context Protocol) server for ommi llm.

Provides remote access to model operations via the MCP protocol.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    EmbeddedResource,
    ImageContent,
)

logger = logging.getLogger(__name__)


class OmmiLLMServer:
    """
    MCP server for managing large language model inference.

    Provides tools for:
    - Loading models
    - Running inference
    - Managing model sharding
    - Configuring optimizations
    """

    def __init__(self):
        self.server = Server("ommi-llm")
        self._models: Dict[str, any] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP handlers."""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="ommi://models",
                    name="Available Models",
                    description="List of loaded and available models",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ommi://memory",
                    name="Memory Status",
                    description="Current memory usage statistics",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI."""
            if uri == "ommi://models":
                return self._get_models_info()
            elif uri == "ommi://memory":
                return self._get_memory_info()
            raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="load_model",
                    description="Load a model from HuggingFace or local path",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "HuggingFace model ID or local path",
                            },
                            "device": {
                                "type": "string",
                                "enum": ["cuda", "cpu", "mps"],
                                "description": "Device to load model on",
                                "default": "cuda",
                            },
                            "dtype": {
                                "type": "string",
                                "enum": ["float16", "bfloat16", "float32"],
                                "description": "Data type for computation",
                                "default": "float16",
                            },
                            "prefetching": {
                                "type": "boolean",
                                "description": "Enable async layer prefetching",
                                "default": True,
                            },
                            "compression": {
                                "type": "string",
                                "enum": [None, "4bit", "8bit"],
                                "description": "Quantization mode",
                                "default": None,
                            },
                        },
                        "required": ["model_name"],
                    },
                ),
                Tool(
                    name="generate",
                    description="Generate text using loaded model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "Model identifier from load_model",
                            },
                            "prompt": {"type": "string", "description": "Input prompt text"},
                            "max_new_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                                "default": 100,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature",
                                "default": 0.7,
                            },
                            "top_p": {
                                "type": "number",
                                "description": "Nucleus sampling threshold",
                                "default": 0.9,
                            },
                        },
                        "required": ["model_id", "prompt"],
                    },
                ),
                Tool(
                    name="shard_model",
                    description="Shard a model into layer-wise files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "HuggingFace model ID or local path",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save sharded model",
                            },
                            "compression": {
                                "type": "string",
                                "enum": [None, "4bit", "8bit"],
                                "description": "Apply quantization during sharding",
                                "default": None,
                            },
                        },
                        "required": ["model_name", "output_path"],
                    },
                ),
                Tool(
                    name="get_model_info",
                    description="Get information about a loaded model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string", "description": "Model identifier"}
                        },
                        "required": ["model_id"],
                    },
                ),
                Tool(
                    name="unload_model",
                    description="Unload a model to free memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "Model identifier to unload",
                            }
                        },
                        "required": ["model_id"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Execute a tool."""
            try:
                if name == "load_model":
                    return await self._handle_load_model(arguments)
                elif name == "generate":
                    return await self._handle_generate(arguments)
                elif name == "shard_model":
                    return await self._handle_shard_model(arguments)
                elif name == "get_model_info":
                    return await self._handle_get_model_info(arguments)
                elif name == "unload_model":
                    return await self._handle_unload_model(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_load_model(self, args: dict) -> List[TextContent]:
        """Handle load_model tool."""
        from ..core.auto_model import AutoModel

        model_name = args["model_name"]
        device = args.get("device", "cuda")
        dtype = args.get("dtype", "float16")
        prefetching = args.get("prefetching", True)
        compression = args.get("compression")

        model = AutoModel.from_pretrained(
            model_name, device=device, dtype=dtype, prefetching=prefetching, compression=compression
        )

        model_id = f"{model_name.replace('/', '_')}_{len(self._models)}"
        self._models[model_id] = model

        return [TextContent(type="text", text=f"Model loaded successfully. ID: {model_id}")]

    async def _handle_generate(self, args: dict) -> List[TextContent]:
        """Handle generate tool."""
        import torch

        model_id = args["model_id"]
        prompt = args["prompt"]
        max_new_tokens = args.get("max_new_tokens", 100)
        temperature = args.get("temperature", 0.7)
        top_p = args.get("top_p", 0.9)

        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        model = self._models[model_id]

        # Tokenize
        input_tokens = model.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=4096,
        )

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tokens["input_ids"].to(model.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode
        output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return [TextContent(type="text", text=output_text)]

    async def _handle_shard_model(self, args: dict) -> List[TextContent]:
        """Handle shard_model tool."""
        from ..persistence.sharder import ModelSharder

        model_name = args["model_name"]
        output_path = Path(args["output_path"])
        compression = args.get("compression")

        sharder = ModelSharder(model_name, output_path, compression=compression)

        shard_paths = sharder.shard_model()

        return [
            TextContent(
                type="text", text=f"Model sharded into {len(shard_paths)} layers at {output_path}"
            )
        ]

    async def _handle_get_model_info(self, args: dict) -> List[TextContent]:
        """Handle get_model_info tool."""
        model_id = args["model_id"]

        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        model = self._models[model_id]
        info = model.get_model_info()

        return [TextContent(type="text", text=str(info))]

    async def _handle_unload_model(self, args: dict) -> List[TextContent]:
        """Handle unload_model tool."""
        model_id = args["model_id"]

        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        del self._models[model_id]
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

        return [TextContent(type="text", text=f"Model {model_id} unloaded successfully")]

    def _get_models_info(self) -> str:
        """Get information about loaded models."""
        if not self._models:
            return "No models loaded"

        info = []
        for model_id, model in self._models.items():
            model_info = model.get_model_info()
            info.append(f"- {model_id}: {model_info}")

        return "\n".join(info)

    def _get_memory_info(self) -> str:
        """Get memory usage information."""
        from ..utils.memory import MemoryManager

        mm = MemoryManager("cuda")
        stats = mm.get_memory_stats()

        return str(stats)

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ommi-llm",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(),
                ),
            )


def main() -> None:
    """Entry point for MCP server."""
    server = OmmiLLMServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
