"""SensorLog provider module.

See http://sensorlog.berndthomas.net for details about the SensorLog app.

Examples
--------
To read the data from a SensorLog json file one can simply use :func:`read_sensor_log`:

.. testsetup:: sensorlog

    >>> import pkg_resources
    >>> path = pkg_resources.resource_filename(
    ...     'tests.providers.sensorlog', '_resources/example.json'
    ... )

.. doctest:: sensorlog

    >>> from dispel.providers.sensorlog.io import read_sensor_log
    >>> reading = read_sensor_log(path)

Note that the SensorLog format does not know about levels. The reading introduces an
artifical level called ``sensorlog``. It can be accessed like this:

.. doctest:: sensorlog

    >>> level = reading.get_level('sensorlog')

Assuming the data set contains gyroscope data one can access it in the following way:

.. doctest:: sensorlog
    :options: +NORMALIZE_WHITESPACE

    >>> from dispel.providers.sensorlog.data import SensorLogSensorType
    >>> data_set = level.get_raw_data_set(SensorLogSensorType.GYRO)
    >>> data_set.data
                               gyroRotationX  gyroRotationY  gyroRotationZ
    gyroTimestamp_sinceReboot
    0 days 08:09:55.293038          0.054116      -0.004393       0.068387
    0 days 08:09:55.312870          0.089527       0.023450       0.008785
    0 days 08:09:55.332701         -0.077681       0.029209      -0.007083

Other properties, such as the vendor ID recorded for Apple apps can be accessed via the
:data:`~dispel.data.core.Reading.device` property:

.. doctest:: sensorlog

    >>> reading.device.uuid
    'E142433C-C949-4965-97E0-475F3131BA6B'

For further details on :class:`~dispel.data.core.Reading` and the data model take a look at
:mod:`dispel.data.core`.

"""

PROVIDER_ID = "sensorlog"
PROVIDER_NAME = "SensorLog"
