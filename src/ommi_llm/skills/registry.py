"""
Skill registry for pluggable optimizations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class Skill(ABC):
    """
    Base class for ommi llm skills.

    Skills are pluggable optimizations that can be applied to
    the inference engine for specific use cases.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def apply(self, engine: Any) -> None:
        """
        Apply the skill to an inference engine.

        Args:
            engine: The inference engine to modify
        """
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the skill with parameters.

        Args:
            config: Configuration dictionary
        """
        pass


class QuantizationSkill(Skill):
    """
    Skill for enabling quantization.

    Configures 4-bit or 8-bit quantization for faster inference
    with reduced memory usage.
    """

    name = "quantization"
    description = "4-bit or 8-bit weight quantization"

    def __init__(self):
        self.mode: Optional[str] = None
        self.blocksize: int = 64

    def configure(self, config: Dict[str, Any]) -> None:
        self.mode = config.get("mode", "4bit")
        self.blocksize = config.get("blocksize", 64)

    def apply(self, engine: Any) -> None:
        engine.compression = self.mode
        logger.info(f"Applied quantization skill: {self.mode}")


class FlashAttentionSkill(Skill):
    """
    Skill for enabling Flash Attention.

    Uses Flash Attention 2 for faster and more memory-efficient attention.
    """

    name = "flash_attention"
    description = "Flash Attention 2 for efficient attention computation"

    def __init__(self):
        self.enabled: bool = True

    def configure(self, config: Dict[str, Any]) -> None:
        self.enabled = config.get("enabled", True)

    def apply(self, engine: Any) -> None:
        if self.enabled:
            # Enable will be handled during model initialization
            logger.info("Flash Attention skill enabled")


class KVCacheSkill(Skill):
    """
    Skill for optimizing KV cache.

    Configures KV cache strategies for efficient autoregressive generation.
    """

    name = "kv_cache"
    description = "Optimized KV cache management"

    def __init__(self):
        self.max_cache_size: int = 4096
        self.offload_to_cpu: bool = False

    def configure(self, config: Dict[str, Any]) -> None:
        self.max_cache_size = config.get("max_cache_size", 4096)
        self.offload_to_cpu = config.get("offload_to_cpu", False)

    def apply(self, engine: Any) -> None:
        engine.kv_cache_config = {
            "max_cache_size": self.max_cache_size,
            "offload_to_cpu": self.offload_to_cpu,
        }
        logger.info(f"KV cache skill applied: max_size={self.max_cache_size}")


class SkillRegistry:
    """
    Registry for managing skills.

    Allows registration, configuration, and application of skills
    to inference engines.
    """

    def __init__(self):
        self._skills: Dict[str, Type[Skill]] = {}
        self._active_skills: Dict[str, Skill] = {}

        # Register built-in skills
        self.register_skill(QuantizationSkill)
        self.register_skill(FlashAttentionSkill)
        self.register_skill(KVCacheSkill)

    def register_skill(self, skill_class: Type[Skill]) -> None:
        """
        Register a skill class.

        Args:
            skill_class: Skill class to register
        """
        self._skills[skill_class.name] = skill_class
        logger.debug(f"Registered skill: {skill_class.name}")

    def get_skill(self, name: str) -> Optional[Type[Skill]]:
        """
        Get a skill class by name.

        Args:
            name: Skill name

        Returns:
            Skill class or None
        """
        return self._skills.get(name)

    def list_skills(self) -> List[Dict[str, str]]:
        """
        List all registered skills.

        Returns:
            List of skill information dictionaries
        """
        return [
            {"name": name, "description": skill.description} for name, skill in self._skills.items()
        ]

    def apply_skill(self, name: str, engine: Any, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Apply a skill to an engine.

        Args:
            name: Skill name
            engine: Inference engine
            config: Optional skill configuration

        Returns:
            True if skill was applied successfully
        """
        skill_class = self._skills.get(name)
        if not skill_class:
            logger.error(f"Unknown skill: {name}")
            return False

        try:
            skill = skill_class()
            if config:
                skill.configure(config)
            skill.apply(engine)
            self._active_skills[name] = skill
            return True
        except Exception as e:
            logger.error(f"Failed to apply skill {name}: {e}")
            return False

    def apply_skills(self, engine: Any, skills_config: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Apply multiple skills from configuration.

        Args:
            engine: Inference engine
            skills_config: Dictionary mapping skill names to configurations

        Returns:
            List of successfully applied skill names
        """
        applied = []
        for name, config in skills_config.items():
            if self.apply_skill(name, engine, config):
                applied.append(name)
        return applied


# Global skill registry instance
_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry
