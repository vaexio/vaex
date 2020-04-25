Running benchmarks
------------------

Vaex benchmarks are run with `Airspeed Velocity <https://asv.readthedocs.io/en/stable/>`__ on dedicated hardware. To develop or test benchmarks locally follow these steps:

1. Install ASV:

.. code:: bash

    conda install -c conda-forge asv

2. Implement or change the benchmarks next to existing ones. See `Writing benchmarks <https://asv.readthedocs.io/en/stable/writing_benchmarks.html>`__.

3. Run them in dev mode: this will run them in the current Python environment and will repeat each test only once, for a given benchmarks suite:

.. code:: bash

    asv dev --bench Strings

4. Or run them fully on your laptop, again in the current Python environment:

.. code:: bash

    asv run --python=$(which python)

On the dedicated hardware ASV will create a new Conda environment for each run.
