{{ name|title }}
{{ '-' * name|length }}

Overview
~~~~~~~~

{% with measures_generic=measures|selectattr('1.modalities', 'none')|list %}
{% with measures_modalities=measures|rejectattr('1.modalities', 'none')|list %}
{% if measures_generic %}
Generic Measures
````````````````
{% with measures=measures_generic %}{% include '_measure_detail_link_list.rst' %}{% endwith %}
{% endif %}

{% if measures_modalities %}
{% for modalities, measures in measures_modalities|groupby('1.modalities') %}
{% with modalities_title=modalities|join(' ') %}
{{ modalities_title|title }}
{{ '`' * modalities_title|length }}
{% endwith %}
{% include '_measure_detail_link_list.rst' %}
{% endfor %}
{% endif %}

Details
~~~~~~~

{% if measures_generic %}
Generic Measures
````````````````
{% for step, measure, parameters in measures_generic %}
{% with measures=measures_generic %}{% include '_measure_detail_properties.rst' %}{% endwith %}
{% endfor %}
{% endif %}

{% if measures_modalities %}
{% for modalities, measures in measures_modalities|groupby('1.modalities') %}
{% with modalities_title=modalities|join(' ') %}
{{ modalities_title|title }}
{{ '`' * modalities_title|length }}
{% endwith %}
{% for step, measure, parameters in measures %}
{% include '_measure_detail_properties.rst' %}
{% endfor %}
{% endfor %}
{% endif %}
{% endwith %}
{% endwith %}
