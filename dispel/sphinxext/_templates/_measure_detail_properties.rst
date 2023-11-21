.. index::
    pair: {{ measure.id }};{{ measure.name }}

.. _{{ measure.id }}:

{{ measure.name }}
{{ '^' * measure.name|length }}

.. list-table::
    :widths: 20 80
    :header-rows: 1
    :stub-columns: 1

    * - Property
      - Value
    * - Id
      - ``{{ measure.id }}``
    * - Produced by
      - :class:`~{{ step.__class__.__module__ }}.{{ step.__class__.__name__ }}`
    {% if measure.modalities %}
    * - Modalities
      - {% for modality in measure.modalities %}
        * {{ modality }}
        {% endfor %}
    {% endif %}
    {% if measure.description %}
    * - Description
      - {{ measure.description }}
    {% endif %}
    {% if measure.unit %}
    * - Unit
      - {{ measure.unit }}
    {% endif %}
    {% if measure.validator is range_validator %}
    * - Constraints
      - Values are:
          {% if measure.validator.lower_bound is not none -%}* equal or greater than ``{{ measure.validator.lower_bound }}``{% endif %}
          {% if measure.validator.upper_bound is not none -%}* equal or less than ``{{ measure.validator.upper_bound }}``{% endif %}
    {% elif measure.validator is set_validator %}
    * - Constraints
      - Values are one of:
          {% for allowed_value in measure.validator.allowed_values %}
          * ``{{ allowed_value }}``{% if measure.validator.get_label(allowed_value) %}: {{ measure.validator.get_label(allowed_value) }}{% endif %}
          {% endfor %}
    {% endif %}
    {% if parameters %}
    * - Parameters
      - {% for pstep, params in parameters.items() %}
        :class:`~{{ pstep.__class__.__module__ }}.{{ pstep.__class__.__name__ }}`
        {% for _, param in params %}
        * ``{{ param.id }}``: {{ param.description }} (default: {{ param.default_value }}, validator: {{ param.validator }})
        {% endfor %}
        {% endfor %}
    {% endif %}
