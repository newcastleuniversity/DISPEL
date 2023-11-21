.. _data_model:

Data model
==========

.. _reading_data_sets:

Reading
-------

The basic container for all data processed is a :class:`~dispel.data.core.Reading`.
_providers extend the library with reading functionality for specific file formats.
The function :func:`~dispel.io.read` allows automatically parse registered formats into
the reading data model. The providers too come with functions that allow to read from
files directly, if the format is known.

As an example, we will read a file using the parsing for Ad Scientiam's file
format. First you have to import and read in given a path:

.. code-block:: python

    from dispel.providers.ads.io import read_ads

    reading = read_ads('path/to/file.json')

Level
-----

Level definition
~~~~~~~~~~~~~~~~

Tasks used to assess the performance of a subject can be composed of multiple
sub-tasks used to assess multiple aspects within a domain. To provide a
logical structure for data capture, these sub-tasks are structured in so
called *levels*. From a data analysis perspective a level is a logical unit of
analysis. For example, the *DRAW* task asks the subject to draw four different
shapes twice with both their left and right hand. Each attempt to draw a
shape with one hand is considered a level as one would derive measures for
analyses for each and everyone of them.


Identification of levels
~~~~~~~~~~~~~~~~~~~~~~~~

Levels are identified by their id that allows to uniquely select one scenario
out of the above mentioned modalities. The ids are derived from the lower case
name of the modality. Multiple modalities are separated with a dash
(\ ``-``\ ).

CPS
^^^


* Symbol-to-digit/digit-to-digit

  * Symbol-to-digit (\ ``symbol_to_digit``\ )
  * Digit-to-digit (\ ``digit_to_digit``\ )

PINCH / DRAW / GRIP
^^^^^^^^^^^^^^^^^^^


* Hand

  * Left (\ ``left``\ )
  * Right (\ ``right``\ )

* Shape

  * Square counter-clock (\ ``square_counter_clock``\ )
  * Square (\ ``square``\ )
  * Figure 8 (\ ``figure_eight``\ )
  * Spiral (\ ``spiral``\ )

* Attempt

  * 1st attempt (\ ``1``\ )
  * 2nd attempt (\ ``2``\ )

An example for *DRAW* would be ``left_square_1``.

Relationships
~~~~~~~~~~~~~

The following diagram illustrated the class relationships between ``Reading``\
, ``Level``\ , ``RawDataSet``\ , ``LevelEpoch`` and ``MeasureSet``.

.. mermaid::

   classDiagram
       class Reading {
           -levels: Dict[str, Level]
           ...
           +get_level(id_: Union[str, LevelId]): Level
       }
       class LevelId{
           +id: str
           +modalities: Tuple[str, ...]
           +from_modalities(*modalities): LevelId
       }
       class LevelEpoch{
           ...
       }
       class Level{
           +id: LevelId
           -_raw_data_sets: Dict[str, RawDataSet]
           -_measure_set: MeasureSet
           +get_raw_data_set(id_: str): RawDataSet
           +epochs: List[LevelEpoch]
           +get_measure_set(): MeasureSet
           +get_measure(id_: str): MeasureValue
       }
       Level "1" --> "1" LevelId
       Level "1" --> "0..*" LevelEpoch
       class RawDataSet
       class MeasureSet
       Reading "1" --> "1..*" Level
       Level "1" --> "0..*" RawDataSet
       Reading "1" --> "0..1" MeasureSet
       Level "1" --> "0..1" MeasureSet



Extracting a Level
~~~~~~~~~~~~~~~~~~

To get access to one of the :class:`dispel.data.levels.Level` use the method
:meth:`~dispel.data.core.Reading.get_level`:

.. code-block:: python

    level = reading.get_level('<level_id>')

where ``<level_id>`` has to be replaced with the desired ``level_id``. e.g.
if you have a reading of the CPS test, you can use
``level_id = 'digit_to_symbol'``.

One can also extract all levels from a :class:`~dispel.data.core.Reading` using
:attr:`~dispel.data.core.Reading.levels`:

.. code-block:: python

    levels = reading.levels


LevelEpoch
----------

:class:`~dispel.data.levels.LevelEpoch` allow to describe specific time periods
in levels with measures. They can be used to both process and extract data
and measures for those specific epochs in time.

RawDataSet
----------

A :class:`~dispel.data.raw.RawDataSet` is a data structure with a pandas
data frame encapsulated with a :class:`~dispel.data.raw.RawDataSetDefinition`
composed with information about the data source, and a short description of
the values in RawDataSet (see
:class:`~dispel.data.raw.RawDataValueDefinition`).

.. mermaid ::

    classDiagram
        class RawDataSet{
            +definition: RawDataSetDefinition
            +data: pandas.DataFrame}
        class RawDataSetDefinition{
            +id_: str
            +source: RawDataSource
            +value_definitions: Iterable[ RawDataValueDefinition]
            +is_computed: bool
        }
        RawDataSet "1" --> "1" RawDataSetDefinition


Extracting a RawDataSet
~~~~~~~~~~~~~~~~~~~~~~~

To get access to one of the :class:`~dispel.data.raw.RawDataSet` s contained in
the Level, simply call

.. code-block:: python

    data = level.get_raw_data_set('<id>')

where ``<id>`` has to be replaced with the data set ids, for the cps example
with `level_id = 'digit_to_symbol'` one may use ``id = 'userInput'``.

Minimum working example
-----------------------

We illustrate how to get a ``pandas.DataFrame`` with formatted data from a
json file with an example, reading data from a CPS experiment at the level
`symbol-to-digit` with key ``level_id = 'digit_to_symbol'`` and
:class:`dispel.data.raw.RawDataSet` id set to ``id = 'userInput'``.

.. code-block:: python

    from dispel.io.ads import read_ads

    # path to json
    path = "./tests/io/_resources/ads/CPS/example.json"

    # read ads
    reading = read_ads(path)

    # extract dataset
    data_set = reading.get_level('digit_to_symbol').get_raw_data_set('userInput')

    # get dataframe from data_set
    df = data_set.data

Flag
------------

:class:`~dispel.data.flags.Flag`\ s provide a structured way to
mark entities as valid. Those flags can originate from both technical
issues (:attr:`~dispel.data.flags.FlagType.TECHNICAL`) with the
underlying data capture or behavioural aspects (
:attr:`~dispel.data.flags.FlagType.BEHAVIORAL`)
of subjects performing tests not according to their respective protocol.

Flags are supported for :class:`~dispel.data.core.Reading`\ s,
:class:`~dispel.data.levels.Level`\ s, :class:`~dispel.data.raw.RawDataSet`\ s and
:class:`~dispel.data.measures.MeasureValue`\ s.

:meth:`~dispel.data.flags.FlagMixIn.is_valid` indicates if a
particular entity is valid and
:meth:`~dispel.data.flags.FlagMixIn.get_flags` allows to
retrieve all reasons why a particular entity was flagged.

Flags contain both an identifier and a reason. The ``id`` contains
three pieces of information:

    * ``task_name`` e.g. `CPS` etc.
    * ``flag_type`` e.g. `technical`, `behavioral` etc.
    * ``flag_severity`` e.g. `deviation` or `invalidation`
    * ``flag_name`` e.g. `tilt_angle`, `pressure_range` etc.

And the ``reason`` contains a more descriptive message of the data
flag.

Flags can be defined as follows:

.. doctest:: flag

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.flags import Flag, FlagId, FlagSeverity, FlagType
    >>> flag_id = FlagId(
    ...     task_name=AV('Pinch test', 'pinch'),
    ...     flag_name=AV('Tilt angle', 'ta'),
    ...     flag_type=FlagType.BEHAVIORAL,
    ...     flag_severity=FlagSeverity.DEVIATION,
    ... )
    >>> flag_id
    pinch-behavioral-deviation-ta

    >>> flag = Flag(
    ...     id_=flag_id,
    ...     reason='The tilt angle of the phone is too flat during the Pinch '
    ...            'test.'
    ... )
    >>> flag
    Flag(id=pinch-behavioral-deviation-ta, reason='The tilt angle of the phone is too flat during the ...
