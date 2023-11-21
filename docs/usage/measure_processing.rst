.. _measure_processing:

Measure processing
==================

The library provides a basic framework to extract measures from raw data
captured in :class:`dispel.data.core.Reading`\ s. All functionality can be found
under the :mod:`dispel.processing` module. This section provides a brief
introduction on using the framework.

For a comprehensive list of measures that are produced please see
:ref:`here <measures>`.

Measure definitions
-------------------

In order to standardize how measures are represented the library comes with a
few base classes that handle the definition of measures.

The following example shows how to create a basic measure definition

.. code-block:: python

    from dispel.data.validators import RangeValidator
    from dispel.data.values import ValueDefinition

    definition = ValueDefinition(
        id_='reaction-time',
        name='Reaction time',
        unit='s',
        description='The time it takes the subject to respond to the stimulus',
        data_type='float64',
        validator=RangeValidator(lower_bound=0)
    )

Later on this definition is used to tie both definition and values together
using the :class:`~dispel.data.measures.MeasureValue`.

A common use case is to have a group of related measures, e.g. the same metric
aggregated using different descriptive statistics. The library offers a
prototype definition

.. doctest:: usage-prototype

    >>> from dispel.data.validators import RangeValidator
    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> prototype = ValueDefinitionPrototype(
    ...     id_='{method}-reaction-time',
    ...     name='{method} reaction time',
    ...     unit='s',
    ...     description='The {method} time it takes the subject to respond '
    ...                 'to all stimuli',
    ...     data_type='float64',
    ...     validator=RangeValidator(lower_bound=0)
    ... )

    Given this prototype one can quickly create definitions
    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> prototype = ValueDefinitionPrototype(
    ...     id_='{method}-reaction-time',
    ...     name='{method} reaction time',
    ...     unit='s',
    ...     description='The {method} time it takes the subject to respond '
    ...                 'to all stimuli',
    ...     data_type='float64',
    ...     validator=RangeValidator(lower_bound=0)
    ... )

Given this prototype one can quickly create definitions

.. doctest:: usage-prototype

    >>> prototype.create_definition(method='median')
    <ValueDefinition: median-reaction-time (median reaction time, s)>

The prototypes can consume as many placeholders as needed and use python's
:meth:`str.format` method to create the actual definition.

The measure's ``id`` is represented using the
:class:`~dispel.data.values.DefinitionId` class. This allows to standardize
measure ids. In the above examples the definition creates simply an instance
of :class:`~dispel.data.values.DefinitionId` by using
:meth:`~dispel.data.values.DefinitionId.from_str`. One can provide their own
standard or use one of the more complex ones like
:class:`dispel.data.measures.MeasureId`:

.. doctest::

    >>> from dispel.data.measures import MeasureId
    >>> from dispel.data.values import AbbreviatedValue as AV, ValueDefinition
    >>> measure_name = AV('reaction time', 'rt')
    >>> definition = ValueDefinition(
    ...     id_=MeasureId(
    ...         task_name=AV('test', 'tst'),
    ...         measure_name=measure_name
    ...     ),
    ...     name=measure_name
    ... )
    >>> definition
    <ValueDefinition: tst-rt (reaction time)>

Since this is a common use case the library provides two additional classes
:class:`~dispel.data.measures.MeasureValueDefinition` and
:class:`~dispel.data.measures.MeasureValueDefinitionPrototype`. These two
classes allow to structure the definitions into tasks, modalities/variants of
the task, measure name, aggregation method, and running ids:

.. doctest::

    >>> from dispel.data.measures import MeasureValueDefinition
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> definition = MeasureValueDefinition(
    ...     task_name=AV('Cognitive Processing Speed test', 'CPS'),
    ...     measure_name=AV('correct responses', 'cr'),
    ...     modalities=[
    ...         AV('digit-to-digit', 'dtd'),
    ...         AV('predefined key 1', 'key1')
    ...     ],
    ...     aggregation=AV('standard deviation', 'std')
    ... )
    >>> definition
    <MeasureValueDefinition: cps-dtd_key1-cr-std (CPS digit-to-digit ...>

.. _measure-extraction:

Measure extraction
------------------

Measure extraction methods are organized in modules per test, e.g. the
*Cognitive Processing Speed* (CPS) test measures extraction is available in
the :mod:`dispel.providers.generic.tasks.cps.steps`. See also
:ref:`here <contribute_new_test_module>` for details on how to contribute new
processing modules for tests.

Measure extraction is typically comprised of two generic tasks: (1)
*transforming* raw signals (e.g. computing the magnitude of a signal); and (2)
*extracting* a measure (e.g. the maximum magnitude value of the signal). To
ensure re-usability of some generic building blocks the library provides a
framework around handling these steps.

The basic class :class:`~dispel.processing.core.ProcessingStep` represents one
step that consumes a :class:`~dispel.data.core.Reading` and
yields a :class:`~dispel.processing.core.ProcessingResult` that wraps one of
:class:`~dispel.data.raw.RawDataSet`, :class:`~dispel.data.measures.MeasureValue`,
or :class:`~dispel.data.levels.Level`.

Measure extractions can be defined by
providing a list of :class:`~dispel.processing.core.ProcessingStep`\ s to the
function :func:`~dispel.processing.process`:

.. code-block:: python

    import pandas as pd
    from dispel.data.core import Reading
    from dispel.data.levels import Level
    from dispel.data.measures import MeasureValue
    from dispel.data.raw import (RawDataSetSource, RawDataValueDefinition,
                              RawDataSetDefinition, RawDataSet)
    from dispel.data.values import ValueDefinition

    from dispel.processing import ErrorHandling, ProcessingStep
    from dispel.processing.data_set import RawDataSetProcessingResult
    from dispel.processing.level import LevelProcessingResult
    from dispel.signal import euclidean_norm

    class EuclideanNorm(ProcessingStep):
        def __init__(self, data_set_id, level_id):
            self.data_set_id = data_set_id
            self.level_id = level_id

        def process_reading(self, reading: Reading):
            input = reading.get_level(self.level_id).get_raw_data_set(
                self.data_set_id
            )
            res = euclidean_norm(input.data)
            yield RawDataSetProcessingResult(
                step=self,
                sources=input,
                level=reading.get_level(self.level_id),
                result=RawDataSet(
                    RawDataSetDefinition(
                        f'{self.data_set_id}-euclidean-norm',
                        RawDataSetSource('konectom'),
                        [RawDataValueDefinition('mag', 'magnitude')],
                        True  # is computed!
                    ),
                    pd.DataFrame(res.rename('mag'))
                )
            )

    class MaxValue(ProcessingStep):
        def __init__(self, data_set_id, level_id, measure_value_definition):
            self.data_set_id = data_set_id
            self.level_id = level_id
            self.measure_value_definition = measure_value_definition

        def process_reading(self, reading: Reading, **kwargs):
            input = reading.get_level(self.level_id).get_raw_data_set(
                self.data_set_id
            )
            yield LevelProcessingResult(
                step=self,
                sources=input,
                level=reading.get_level(self.level_id),
                result=MeasureValue(
                    self.measure_value_definition,
                    input.data.max().max()
                )
            )

    steps = [
        EuclideanNorm('accelerometer_ts', level_id),
        MaxValue(
            'accelerometer_ts-euclidean-norm',
            level_id,
            ValueDefinition(
                'max-acc',
                'Maximum magnitude of acceleration',
                'm/s^2'
            )
        )
    ]

The actual processing is done by calling :func:`~dispel.processing.process`
on a reading. The following example assumes you have a
:class:`~dispel.data.core.Reading` in the variable ``reading``. For details on
reading data sets see :ref:`here <reading_data_sets>`.

.. doctest-skip::

    >>> from dispel.processing import process
    >>> res = process(example, steps)
    >>> res
    <DataTrace of <Reading: 2 levels (0 flags)>: (11 entities, 2 ...
    >>> reading = res.get_reading()
    >>> reading.get_measure_set(level_id).get_raw_value('max-acc')
    0.012348961

The results will then be available in the ``measure_set`` attribute of the
returned :class:`~dispel.data.core.Reading` or from the attribute of
the :class:`~dispel.data.levels.Level` available with
:meth:`~dispel.data.core.Reading.get_level`.

Transformation & Extraction
```````````````````````````

Since the two examples above represent two common scenarios of consuming one
or more raw data sets to transform and consuming one or more raw data sets to
extract one or more measures the following convenience classes exist:
:class:`~dispel.processing.transform.TransformStep`,
:class:`~dispel.processing.extract.ExtractStep`, and
:class:`~dispel.processing.extract.ExtractMultipleStep`. This simplifies the
definition of the above examples as follows:

.. doctest-skip::

    >>> from dispel.processing.extract import ExtractStep
    >>> from dispel.processing.transform import TransformStep
    >>> transform_step = TransformStep(
    ...     'accelerometer_ts',
    ...     euclidean_norm,
    ...     'accelerometer-euclidean-norm',
    ...     [RawDataValueDefinition('mag', 'magnitude')]
    ... )
    >>> extract_step = ExtractStep(
    ...     'accelerometer-euclidean-norm',
    ...     lambda data: data.max().max(),
    ...     ValueDefinition(
    ...         'max-acc',
    ...         'Maximum magnitude of acceleration',
    ...         'm/s^2'
    ...     )
    ... )
    >>> steps = [
    ...     transform_step,
    ...     extract_step
    ... ]
    >>> res = process(example, steps).get_reading()

One can also use supplementary information on top of the automatically passed
data frame inside the transformation functions. This functionality can be used
by passing either ``level`` and/or ``reading`` as parameters of the
transformation function and they will be automatically provided.

.. doctest-skip::

    >>> from dispel.processing.extract import ExtractStep
    >>> from dispel.processing.transform import TransformStep
    >>> def reaction_time(data, level):
    ...     return (
    ...         data['ts'].min() - level.start
    ...     ).total_seconds()
    >>> extract_step = ExtractStep(
    ...     'accelerometer',
    ...     reaction_time,
    ...     ValueDefinition(
    ...         'rt',
    ...         'Reaction time',
    ...         's'
    ...     )
    ... )
    >>> steps = [extract_step]
    >>> res = process(example, steps).get_reading()

Often transform and extract steps are defined as classes to ensure steps can
be reused:

.. doctest-skip::

    >>> from dispel.processing.data_set import transformation
    >>> class MyExtractStep(ExtractStep):
    ...     data_set_ids = 'accelerometer'
    ...     definition = ValueDefinition(
    ...         'rt',
    ...         'Reaction time',
    ...         's'
    ...     )
    ...
    ...     @transformation
    ...     def reaction_time(self, data, level):
    ...         return (
    ...             data['ts'].min() - level.start
    ...         ).total_seconds()
    >>> steps = [MyExtractStep()]
    >>> res = process(example, steps).get_reading()

The above example shows some additional concepts that allow specify arguments,
such as the data set ids, via class variables. Furthermore, class routines
can be decorated with ``@transformation`` to specify the transformation
applied to the data sets. Further details and more advanced use cases can be
found in the documentation of :class:`~dispel.processing.transform.TransformStep`
and :class:`~dispel.processing.extract.ExtractStep`.

Grouping
````````

Another common scenario is to extract measures for a specific task and
sub-task. :class:`~dispel.processing.extract.ExtractStep` allows to pass a
:class:`~dispel.data.values.ValueDefinitionPrototype` instead of the
concrete definition. The helper class
:class:`~dispel.processing.level.ProcessingStepGroup` can be used to provide
additional arguments to the prototype:

.. doctest-skip::

    >>> from dispel.data.measures import MeasureValueDefinitionPrototype
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.processing.level import ProcessingStepGroup
    >>> steps = [
    ...     ProcessingStepGroup([
    ...         transform_step,
    ...         ExtractStep(
    ...             'accelerometer-euclidean-norm',
    ...             lambda data: data.max().max(),
    ...             MeasureValueDefinitionPrototype(
    ...                 measure_name=AV('measure 1', 'f'),
    ...                 description='{task_name} measure 1 description',
    ...                 unit='s'
    ...             )
    ...         )],
    ...         task_name=AV('U-turn test', 'UTT')
    ...     )
    ... ]
    >>> res = process(example, steps).get_reading()

This is achieved by passing all named parameters from ``ProcessingStepGroup``
to the ``process`` function of each step.

Filtering
`````````

Often one wants to process specific levels of a reading. Each level-based
processing step allows to specify a
:class:`~dispel.processing.level.LevelFilter` that allows to determine which
level will be considered during processing.

The supported processing step classes are

- :class:`~dispel.processing.level.LevelProcessingStep`
- :class:`~dispel.processing.level.ProcessingStepGroup`
- :class:`~dispel.processing.transform.TransformStep`
- :class:`~dispel.processing.transform.ConcatenateLevels`
- :class:`~dispel.processing.extract.ExtractStep`
- any other processing step inheriting from
  :class:`~dispel.processing.level.LevelFilterProcessingStepMixin`.

Parameters
----------

Some processing might be contingent on the context used.
:class:`~dispel.processing.core.Parameter` allows to specify configurable
values that can be used to configure behavior of processing steps and linked
to extracted measures. This is important to keep lineage of any dimension
affecting the measure.

Parameters automatically create a unique id based on their location of
specification and the provided name. To link a parameter to a processing
step it has to be either defined directly in the processing step or assigned
to an attribute. :meth:`dispel.processing.core.ProcessingStep.get_parameters`
automatically determines all parameters of a step through inspection.

Assuming we had a module called ``example.module`` that defines a parameter on
a module level and on a processing step level. The typical pattern of usage
would be as following:

.. doctest-skip::

    >>> # example.module
    >>> from dispel.data.core import Reading
    >>> from dispel.data.validators import GREATER_THAN_ZERO
    >>> from dispel.processing.core import Parameter
    >>> from dispel.processing.data_set import transformation
    >>> from dispel.processing.transform import TransformStep
    >>> PARAM_A = Parameter(
    ...     id_='param_a',
    ...     default_value=10,
    ...     validator=GREATER_THAN_ZERO,
    ...     description='A description explaining the influence of the param.'
    ... )
    >>> def transform(data, param_a, param_b):
    ...     return ...
    >>> class MyTransformStep(TransformStep):
    ...     param_a = PARAM_A
    ...     param_b = Parameter('param_b')
    ...     @transformation
    ...     def _transform(self, data):
    ...         return transform(data, self.param_a, self.param_b)

The above specification will lead to two parameters called
``example.module.param_a`` and ``example.module.MyTransformStep.param_b``.
The values can be modified by either using their id or reference, e.g.,
``PARAM_A.value = 5`` or ``Parameter.set_value('example.module.param_a', 5)``.

Data trace graph
----------------

The data trace constitutes a
`DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ like
representation of the main data entities of each evaluation i.e.

    - :class:`~dispel.data.core.Reading`,
    - :class:`~dispel.data.levels.Level`,
    - :class:`~dispel.data.levels.LevelEpoch`.
    - :class:`~dispel.data.raw.RawDataSet`,
    - :class:`~dispel.data.measures.MeasureValue`.

The links between entities are the processing steps that were applied on
the source and led to the target entity.

The goal of the data trace graph is to keep tabs on transformation and
extraction steps in order to trace which raw data has led to the creation
on which measure.

Every entity is wrapped in a :class:`~dispel.processing.data_trace.Node`
class that links both parent and child nodes related to it. All nodes are
then stored in the :class:`~dispel.processing.data_trace.DataTrace` class.

In order to populate the data trace graph one can use the
:meth:`~dispel.processing.data_trace.DataTrace.populate` dispatch method.
