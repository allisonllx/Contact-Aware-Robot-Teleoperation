from .config import SCENARIOS

__all__ = ["FrankaForceEnv", "SCENARIOS"]


def __getattr__(name):
    if name == "FrankaForceEnv":
        from .env import FrankaForceEnv

        return FrankaForceEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
