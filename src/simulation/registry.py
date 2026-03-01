import os
import importlib.util
import inspect
from typing import Dict, List, Type, Any
from .strategy import Strategy

class StrategyRegistry:
    def __init__(self, strategies_dir: str = "src/simulation/strategies"):
        self.strategies_dir = strategies_dir
        self._strategies: Dict[str, Type[Strategy]] = {}
        self.load_strategies()

    def load_strategies(self):
        """Scans the directory for Python files and loads classes inheriting from Strategy."""
        self._strategies = {}
        if not os.path.exists(self.strategies_dir):
            return

        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".py") and not filename.startswith("_") and filename != "TEMPLATE.py":
                module_name = filename[:-3]
                file_path = os.path.join(self.strategies_dir, filename)

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for classes inheriting from Strategy
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if issubclass(obj, Strategy) and obj is not Strategy:
                                # Prioritize explicitly defined NAME, fallback to class name
                                strategy_name = getattr(obj, "NAME", name)
                                if strategy_name == "":
                                    strategy_name = name
                                self._strategies[strategy_name] = obj
                except Exception as e:
                    print(f"Error loading strategy {filename}: {e}")

    def get_all(self) -> Dict[str, Type[Strategy]]:
        return self._strategies

    def get_by_name(self, name: str) -> Type[Strategy]:
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found in registry.")
        return self._strategies[name]

    def list_names(self) -> List[str]:
        return list(self._strategies.keys())

    def get_strategy_metadata(self) -> List[Dict[str, Any]]:
        metadata = []
        for name, cls in self._strategies.items():
            metadata.append({
                "name": getattr(cls, "NAME", name) or name,
                "class_name": cls.__name__,
                "description": getattr(cls, "DESCRIPTION", ""),
                "version": getattr(cls, "VERSION", "1.0.0"),
                "author": getattr(cls, "AUTHOR", ""),
                "supported_timeframes": getattr(cls, "SUPPORTED_TIMEFRAMES", []),
                "parameters": cls.get_param_schema() if hasattr(cls, "get_param_schema") else {}
            })
        return metadata

# Global registry instance
registry = StrategyRegistry()
