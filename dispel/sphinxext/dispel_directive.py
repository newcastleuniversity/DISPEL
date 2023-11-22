"""Core module for library specific Sphinx directives."""
from abc import ABC

from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util import nested_parse_with_titles
from sphinx.util.docutils import SphinxDirective


class DispelDirective(SphinxDirective, ABC):
    """Base directive from which library specific directives should inherit."""

    def _parse(self, rst_text, annotation):
        result = ViewList()
        for line in rst_text.split("\n"):
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children
