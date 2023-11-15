"""Extensions to Sphinx to automatically document features."""
from importlib import import_module

from docutils.parsers.rst import directives  # type: ignore
from sphinx.application import Sphinx

from dispel.processing.trace import (
    FeatureTrace,
    get_ancestor_source_graph,
    get_edge_parameters,
    get_traces,
    inspect,
)
from dispel.sphinxext.dispel_directive import DispelDirective
from dispel.sphinxext.templates import FEATURES_DETAIL


def get_variable(path):
    """Get the actual variable for a specified string.

    Parameters
    ----------
    path
        The full module path and variable name of the variable of interest.

    Returns
    -------
    object
        The actual value at the specified ``path``.

    """
    module_path, variable = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, variable)


class FeatureListDirective(DispelDirective):
    """A directive to automatically document feature lists."""

    option_spec = {
        "steps": directives.unchanged_required,
        "name": directives.unchanged_required,
    }

    def run(self):
        """Run the directive."""
        # determine features
        steps_path = self.options["steps"]
        steps = get_variable(steps_path)

        graph = inspect(steps)  # type: ignore

        def _trace_to_tuple(trace: FeatureTrace):
            agg = get_ancestor_source_graph(graph, trace)
            parameters = get_edge_parameters(agg)
            return trace.step, trace.feature, parameters

        features = list(_trace_to_tuple(t) for t in get_traces(graph, FeatureTrace))

        rst_text = FEATURES_DETAIL.render(
            name=self.options["name"], features=features, graph=graph
        )

        return self._parse(rst_text, "<feature-list>")


def setup(app: Sphinx):
    """Run the set-up of the extension."""
    app.add_directive("feature-list", FeatureListDirective)
