.. _providers:

Providers
=========

#. Mobilize-D
#. Konectom
#. Sensorlog
#. APDM Wearables
#. Digital Artefacts


Mobilize-D
----------

Mobilize-D YAR .mat files are already supported !
Each level corresponds to a recording: e.g.: `Recording1` and encapsulates the
following datasets:

* SU_INDIP-LowerBack
* SU_INDIP-LeftFoot
* SU_INDIP-RightFoot
* SU_INDIP-RightWrist
* SU-LowerBack
* Standards-PressureInsoles_raw-LeftFoot
* Standards-PressureInsoles_raw-RightFoot
* Standards-DistanceModule_raw-LeftFoot
* Standards-DistanceModule_raw-RightFoot

All datasets starting with **SU_INDIP** have accelerometer, gyroscope,
magnetometer and timestamp data.
**SU** dataset has an extra column for the Bar data.
Standards Datasets including **PressureInsoles_raw** have 16 pressure columns.
Standards Datasets including **DistanceModule_raw** have 2 distance columns.



.. TODO: list all providers

.. TODO: explain the basic concept for providers and how they extend the functionality


Konectom
--------

Levels for Konectom tasks
~~~~~~~~~~~~~~~~~~~~~~~~~

.. FIXME: Complete levels with what is currently available

The following table indicates the modalities that lead to levels and
configurations that are just properties of the level, i.e. are not
handled as separate levels. Each task has at least one level.

.. list-table::
   :header-rows: 1

   * - Task
     - Modalities
     - Configuration
   * - Mood
     - None
     - None
   * - MSIS-29
     - None
     - None
   * - CPS
     - Symbol-to-digit/digit-to-digit
     - Predefined/random sequence/keys
   * - PINCH
     - Hand
     - None
   * - DRAW
     - Hand, shape, attempt
     - None
   * - GRIP
     - Hand
     - Sequence
   * - SBT
     - None
     - None
   * - UTT
     - None
     - None
   * - 6MWT
     - None
     - None
