"""KARMA brain — the control plane that decides how to serve a model on vLLM.

Milestone M1 (this package today): the FACTS tier and the action space.
See brain/PLAN.md for the full plan and milestone status.
"""
from brain.facts import Facts, default_facts, load_facts
from brain.feasibility import check, fit_table, to_vllm_command
from brain.schema import SLA, Deployment, EngineConfig, FleetConfig, PatchError, RoutingConfig

__all__ = [
    "Facts", "default_facts", "load_facts",
    "check", "fit_table", "to_vllm_command",
    "SLA", "Deployment", "EngineConfig", "FleetConfig", "RoutingConfig", "PatchError",
]
