"""Extensions to Sphinx to automatically document measures."""
from importlib import import_module

from docutils.parsers.rst import directives  # type: ignore
from sphinx.application import Sphinx

from dispel.processing.trace import (
    MeasureTrace,
    get_ancestor_source_graph,
    get_edge_parameters,
    get_traces,
    inspect,
)
from dispel.sphinxext.dispel_directive import DispelDirective
from dispel.sphinxext.templates import MEASURES_DETAIL


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


class MeasureListDirective(DispelDirective):
    """A directive to automatically document measure lists."""

    option_spec = {
        "steps": directives.unchanged_required,
        "name": directives.unchanged_required,
    }

    def run(self):
        """Run the directive."""
        # determine measures
        steps_path = self.options["steps"]
        steps = get_variable(steps_path)

        graph = inspect(steps)  # type: ignore

        def _trace_to_tuple(trace: MeasureTrace):
            agg = get_ancestor_source_graph(graph, trace)
            parameters = get_edge_parameters(agg)
            return trace.step, trace.measure, parameters

        measures = list(_trace_to_tuple(t) for t in get_traces(graph, MeasureTrace))

        rst_text = MEASURES_DETAIL.render(
            name=self.options["name"], measures=measures, graph=graph
        )

        return self._parse(rst_text, "<measure-list>")


def setup(app: Sphinx):
    """Run the set-up of the extension."""
    app.add_directive("measure-list", MeasureListDirective)
