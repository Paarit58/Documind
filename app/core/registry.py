"""
Engine registry for managing OCR engine instances.

The registry provides a centralized location for registering and
retrieving OCR engines, enabling runtime engine selection and
supporting the modular architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.engines.base import OCREngine

logger = logging.getLogger(__name__)


class EngineRegistry:
    """
    Singleton registry for OCR engines.
    
    The registry maintains a mapping of engine names to engine instances,
    allowing the pipeline to select engines by name at runtime.
    
    Usage:
        registry = EngineRegistry()
        registry.register("deepseek_torch", engine_instance)
        engine = registry.get("deepseek_torch")
    """
    
    _instance: EngineRegistry | None = None
    _engines: dict[str, OCREngine]
    _default_engine: str | None
    
    def __new__(cls) -> EngineRegistry:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._engines = {}
            cls._instance._default_engine = None
        return cls._instance
    
    def register(
        self,
        name: str,
        engine: OCREngine,
        set_default: bool = False
    ) -> None:
        """
        Register an OCR engine with the given name.
        
        Args:
            name: Unique identifier for the engine
            engine: OCR engine instance
            set_default: If True, set this engine as the default
            
        Raises:
            ValueError: If an engine with this name is already registered
        """
        if name in self._engines:
            raise ValueError(f"Engine '{name}' is already registered")
        
        self._engines[name] = engine
        logger.info(f"Registered OCR engine: {name}")
        
        if set_default or self._default_engine is None:
            self._default_engine = name
            logger.info(f"Set default OCR engine: {name}")
    
    def get(self, name: str | None = None) -> OCREngine:
        """
        Get an OCR engine by name.
        
        Args:
            name: Engine name, or None to get the default engine
            
        Returns:
            The requested OCR engine
            
        Raises:
            KeyError: If no engine with this name is registered
            RuntimeError: If no engines are registered
        """
        if name is None:
            if self._default_engine is None:
                raise RuntimeError("No OCR engines registered")
            name = self._default_engine
        
        if name not in self._engines:
            available = list(self._engines.keys())
            raise KeyError(
                f"Engine '{name}' not found. Available engines: {available}"
            )
        
        return self._engines[name]
    
    def unregister(self, name: str) -> None:
        """
        Unregister an OCR engine.
        
        Args:
            name: Engine name to unregister
            
        Raises:
            KeyError: If no engine with this name is registered
        """
        if name not in self._engines:
            raise KeyError(f"Engine '{name}' not found")
        
        del self._engines[name]
        logger.info(f"Unregistered OCR engine: {name}")
        
        if self._default_engine == name:
            self._default_engine = next(iter(self._engines), None)
            if self._default_engine:
                logger.info(f"New default OCR engine: {self._default_engine}")
    
    def list_engines(self) -> list[str]:
        """Get a list of all registered engine names."""
        return list(self._engines.keys())
    
    def has_engine(self, name: str) -> bool:
        """Check if an engine with the given name is registered."""
        return name in self._engines
    
    @property
    def default_engine_name(self) -> str | None:
        """Get the name of the default engine."""
        return self._default_engine
    
    def clear(self) -> None:
        """
        Clear all registered engines.
        
        Warning: This is primarily for testing. Use with caution.
        """
        self._engines.clear()
        self._default_engine = None
        logger.warning("Cleared all registered OCR engines")


# Global registry instance
_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    """
    Get the global engine registry instance.
    
    Returns:
        The singleton EngineRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry

