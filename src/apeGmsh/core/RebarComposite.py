"""Facade: ``g.rebar`` moved to :mod:`apeGmsh.rebar._composite` (#1809).

Retires per ``internal_docs/program/retirement_ledger.toml`` row ``facade-core-rebar-composite``.
"""
from ..rebar._composite import RebarComposite, RebarMember, RebarPlacement

__all__ = ["RebarComposite", "RebarMember", "RebarPlacement"]
