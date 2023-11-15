.. hlist::
    :columns: 2

    {% for step, feature, _ in features %}
    * :ref:`{{ feature.name }} <{{ feature.id }}>`
    {% endfor %}
