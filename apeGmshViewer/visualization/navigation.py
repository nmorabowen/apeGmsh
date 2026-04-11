"""
Navigation — Camera control for apeGmshViewer.

Re-exports ``install_navigation`` from the core viewer module
so all viewers share the same implementation.
"""
from apeGmsh.viewers.core.navigation import install_navigation

__all__ = ["install_navigation"]
