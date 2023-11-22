.. highlight:: shell

============
Contributing
============

Get Started!
------------

Ready to contribute? Here's how to set up `dispel` for local development.

1. Clone `dispel` locally::

    $ git clone git@github.com/ # TODO UPDATE THE LINK

2. Install your local copy and development dependencies into a virtualenv::

    $ cd DISPEL

   Using venv::

    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install build
    $ pip install -e ".[docs,dev]"

3. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-measure

   Now you can make your changes locally.

4. When you're done making changes, check that your changes pass flake8 and
   the tests::

    $ make lint
    $ make lint-docs
    $ make test-typing
    $ make test-docs
    $ make test

   To ensure adhering to correct import of modules and formatting, you can run ``make
   format``.

5. Commit your changes and push your branch to GitLab::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-measure

6. Submit a pull request through the GitLab website.

Pre-commit
----------

You can run all the aforementioned styling checks manually as described.
However, we encourage you to use `pre-commit hooks <https://pre-commit.com/>`_
instead to automatically run ``isort``, ``flake8``, ``mypy`` as well as other
linting functionalities. This can be done by installing ``pre-commit``::

    $ pip install pre-commit

and then running::

    $ pre-commit install

from the root of the dispel repository. Now all of the styling checks will be run
each time you commit changes without your needing to run each one manually.
In addition, using ``pre-commit`` will also allow you to more easily remain
up-to-date with code checks as they evolve.

Note that if needed, you can skip these checks with ``git commit --no-verify``.

If you don’t want to use ``pre-commit`` as part of your workflow, you can
still use it to run its checks with::

    $ pre-commit run --files <files you have modified>

without needing to have done ``pre-commit install`` beforehand.
