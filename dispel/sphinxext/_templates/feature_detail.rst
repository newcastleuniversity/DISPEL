{{ name|title }}
{{ '-' * name|length }}

Overview
~~~~~~~~

{% with features_generic=features|selectattr('1.modalities', 'none')|list %}
{% with features_modalities=features|rejectattr('1.modalities', 'none')|list %}
{% if features_generic %}
Generic Features
````````````````
{% with features=features_generic %}{% include '_feature_detail_link_list.rst' %}{% endwith %}
{% endif %}

{% if features_modalities %}
{% for modalities, features in features_modalities|groupby('1.modalities') %}
{% with modalities_title=modalities|join(' ') %}
{{ modalities_title|title }}
{{ '`' * modalities_title|length }}
{% endwith %}
{% include '_feature_detail_link_list.rst' %}
{% endfor %}
{% endif %}

Details
~~~~~~~

{% if features_generic %}
Generic Features
````````````````
{% for step, feature, parameters in features_generic %}
{% with features=features_generic %}{% include '_feature_detail_properties.rst' %}{% endwith %}
{% endfor %}
{% endif %}

{% if features_modalities %}
{% for modalities, features in features_modalities|groupby('1.modalities') %}
{% with modalities_title=modalities|join(' ') %}
{{ modalities_title|title }}
{{ '`' * modalities_title|length }}
{% endwith %}
{% for step, feature, parameters in features %}
{% include '_feature_detail_properties.rst' %}
{% endfor %}
{% endfor %}
{% endif %}
{% endwith %}
{% endwith %}
