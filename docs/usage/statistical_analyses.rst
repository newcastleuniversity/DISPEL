.. _statistical_analyses:

Statistical analyses
====================

.. _feature-collection:

Feature collection
------------------

Feature collections can be used in order ease the handling of multiple
readings and obtain a collection of features wrapped in a class
:class:`~dispel.data.collections.FeatureCollection`.

In order to initialize a feature collection, one can use the following methods:

    - :meth:`~dispel.data.collections.FeatureCollection.from_reading` or
      :meth:`~dispel.data.collections.FeatureCollection.from_readings` by
      directly feeding it one or multiple :class:`~dispel.data.core.Reading`\ s.

    - :meth:`~dispel.data.collections.FeatureCollection.from_data_frame` by
      feeding it a pandas data frame containing a feature export.

    - :meth:`~dispel.data.collections.FeatureCollection.from_csv` by feeding it a
      csv file path to a feature export.

    - :meth:`~dispel.data.collections.FeatureCollection.from_feature_set` by
      feeding it a :class:`~dispel.data.features.FeatureSet`, an
      :class:`~dispel.data.core.Evaluation` and a
      :class:`~dispel.data.core.Session`.

Feature collections can also be extended using the method
:meth:`~dispel.data.collections.FeatureCollection.extend` or simply by the `+`
operator.

Feature collections can also be exported into a dictionary, a JSON file or a
csv file using the following methods
:meth:`~dispel.data.collections.FeatureCollection.to_dict`,
:meth:`~dispel.data.collections.FeatureCollection.to_json` and
:meth:`~dispel.data.collections.FeatureCollection.to_csv`.

Assuming you have a few readings

>>> EXAMPLE_PATHS = [
...     'raw-data-example-01.json',
...     'raw-data-example-02.json',
...     'raw-data-example-03.json',
... ]

The can be read into the data model using the reading functionality

.. doctest-skip::

    >>> from dispel.io.ads import read_ads
    >>> readings = [
    ...     read_ads(path) for path in EXAMPLE_PATHS
    ... ]

Assuming the readings have not been processed one can use
:func:`~dispel.providers.auto_process` to do so:

.. doctest-skip::

    >>> from dispel.providers import auto_process
    >>> processed = [auto_process(r).get_reading() for r in readings]
    >>> processed = [auto_process(r).get_reading() for r in readings]

.. doctest-skip::

    >>> from dispel.data.collections import FeatureCollection
    >>> FeatureCollection.from_reading(processed[0])
    <FeatureCollection: 1 subject, 1 evaluation>
    >>> collection2 = FeatureCollection.from_reading(processed[0])
    >>> collection2
    <FeatureCollection: 1 subject, 1 evaluation>
    >>> collection = FeatureCollection.from_readings(processed)
    >>> collection
    <FeatureCollection: 3 subjects, 3 evaluations>

The collection can also be turned into a dictionary:

.. doctest-skip::

    >>> dict_export = collection.to_dict()

Or stored as JSON or CSV files:

.. doctest-skip::

    >>> collection.to_json('feature_export.json')
    >>> collection.to_csv('feature_export.csv')


Reliability analyses
--------------------

The :mod:`dispel.stats.reliability` module provides tools to perform feature
reliability analysis.

1. Load the CSV export file containing the features values into a
   :class:`~dispel.data.collections.FeatureCollection` using the class method
   :meth:`~dispel.data.collections.FeatureCollection.from_data_frame`.
2. Computing all the ICC test-retest score for all features using the
   :func:`~dispel.stats.reliability.icc_set_test_retest`. It creates
   a :class:`~dispel.stats.reliability.ICCResultSet` object, which contains
   :class:`~dispel.stats.reliability.ICCResult` icc score for each feature.

.. doctest-skip::

    >>> from dispel.stats.reliability import icc_set_test_retest
    >>> from dispel.data.collections import FeatureCollection
    >>> feature_collection = FeatureCollection.from_csv(EXAMPLE_PATH)
    >>> feature_iccs = icc_set_test_retest(feature_collection)

An :class:`~dispel.stats.reliability.ICCResult` object is composed of all the
information relative to the ICC test retest analyses which have been performed.
The kind of reliability (Test retest), the model used (two way mixed, absolute
agreement, average measurement), the ICC score value, the bounds
(upper and lower), the p-value and the sample size and the power of the test.

Learning analysis
-----------------

The :mod:`~dispel.stats.learning` module provides tools to perform feature
learning analyses.
A learning analysis is performed over multiple sessions performed by a subject
related to a feature extracted from its corresponding task (i.e. CPS
task for the feature ``cps-std-rt-mean``).

In order to extract the relevant parameters from a learning analysis (i.e.
learning rate,...), one has to provide the required data wrapped in a
:class:`~dispel.data.collections.FeatureCollection`.



.. code-block:: python

    from dispel.data.collections import FeatureCollection
    import pkg_resources

    EXAMPLE_PATH = pkg_resources.resource_filename(
        'tests.stats', '_resources/single-user-learning-example.csv'
    )
    # Extract a FeatureCollection from a .csv export.
    collection = FeatureCollection.from_csv(EXAMPLE_PATH)


Extract learning parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the feature collection is created, one can extract the learning
parameters. This can be done by passing the feature collection to the
:func:`~dispel.stats.learning.extract_learning_for_one_subject` function:

.. doctest-skip::

    >>> from dispel.stats.learning import extract_learning_for_one_subject
    >>> learning_result = extract_learning_for_one_subject(collection)

The function will then return a :class:`~dispel.stats.learning.LearningResult`
class containing the whole learning analysis results.

One can access the learning parameters with the method
:meth:`~dispel.stats.learning.LearningResult.get_parameters` that outputs a
pandas object (data frame or series) containing the following information:

  * ``subject_id``: the subject identifier,
  * ``feature_id``: the feature identifier,
  * ``optimal_performance``: the optimal performance of the subject for the
    feature in question,
  * ``slope_coefficient``: the slope coefficient of the learning curve of the
    subject for the feature in question,
  * ``learning_rate``: the learning rate of the subject for the feature in
    question,
  * ``warm_up``: the minimum number of sessions needed for the subject to
    attain `90%` of their optimal performance,
  * ``r2_score``: the R squared measure that represents the goodness of the
    fit of the learning model,
  * ``nb_outliers``: the number of rejected outliers during the fit of the
    learning model,
  * ``delay_mean``: the average delay between sessions in days,
  * ``delay_median``: the median delay between sessions in days,
  * ``delay_mean``: the maximum delay between sessions in days,

One can also access the new data points without outliers with the method
:meth:`~dispel.stats.learning.LearningResult.get_new_data`.

Note that if one wants to extract learning parameters for all present subjects
in the :class:`~dispel.data.collections.FeatureCollection`, it can be done by
passing this :class:`~dispel.data.collections.FeatureCollection` and a
``'feature_id'`` to the
:func:`~dispel.stats.learning.extract_learning_for_all_subjects` function.
This function will in the same way return a
:class:`~dispel.stats.learning.LearningResult` class containing the same types
of information as previously described.

.. doctest-skip::

    >>> from dispel.stats.learning import extract_learning_for_all_subjects
    >>> learning_result = extract_learning_for_all_subjects(
    ...     collection, 'CPS-dtd-rt-mean-01')

One can then explore the learning parameters for all users as well as plot
relevant results.
