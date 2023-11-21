.. hlist::
    :columns: 2

    {% for step, measure, _ in measures %}
    * :ref:`{{ measure.name }} <{{ measure.id }}>`
    {% endfor %}
