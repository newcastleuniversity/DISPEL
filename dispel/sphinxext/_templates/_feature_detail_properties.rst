.. index::
    pair: {{ feature.id }};{{ feature.name }}

.. _{{ feature.id }}:

{{ feature.name }}
{{ '^' * feature.name|length }}

.. list-table::
    :widths: 20 80
    :header-rows: 1
    :stub-columns: 1

    * - Property
      - Value
    * - Id
      - ``{{ feature.id }}``
    * - Produced by
      - :class:`~{{ step.__class__.__module__ }}.{{ step.__class__.__name__ }}`
    {% if feature.modalities %}
    * - Modalities
      - {% for modality in feature.modalities %}
        * {{ modality }}
        {% endfor %}
    {% endif %}
    {% if feature.description %}
    * - Description
      - {{ feature.description }}
    {% endif %}
    {% if feature.unit %}
    * - Unit
      - {{ feature.unit }}
    {% endif %}
    {% if feature.validator is range_validator %}
    * - Constraints
      - Values are:
          {% if feature.validator.lower_bound is not none -%}* equal or greater than ``{{ feature.validator.lower_bound }}``{% endif %}
          {% if feature.validator.upper_bound is not none -%}* equal or less than ``{{ feature.validator.upper_bound }}``{% endif %}
    {% elif feature.validator is set_validator %}
    * - Constraints
      - Values are one of:
          {% for allowed_value in feature.validator.allowed_values %}
          * ``{{ allowed_value }}``{% if feature.validator.get_label(allowed_value) %}: {{ feature.validator.get_label(allowed_value) }}{% endif %}
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
