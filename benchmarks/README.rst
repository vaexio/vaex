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

Benchmarks runner
-----------------

The benchmarks are executed on dedicated hardware at a predefined schedule. The HTML results are published to http://asv.vaex.io/ , while the raw performance numbers are saved in the the `vaexio/vaex-asv` repo, for the history. The process works as follows.

1. A cron job triggers the `bin/benchmark_new_commits.sh` script.
2. The script checks out `vaexio/vaex/master`: the benchmark tests are taken from `master`.
3. First the tests are run for the new commits on `master` branch.
4. Then if there are any `vaexio/vaex/bench*` branches they are also tested: for any new commits on top of current `master`. Note that the benchmark tests are still taken from the `master` branch, so new benchmarks from e.g. a `bench_new_feature` branch would not run yet. So this process can be useful to improve existing benchmarks rather than developing new ones.
5. The results are merged with those from `vaexio/vaex-asv/.asv/results/**/*.json` and updated in that branch. This is done to keep the history of benchmark runs.
6. Static HTML files are generated and copied to the www folder on the benchmarking machine, making them available at http://asv.vaex.io/ .

Initial setup of the machine
****************************

To setup the benchmarks runner on a new machine you can do the following.

1. Setup a unix user which will run the benchmarks: e.g. "`github`".
2. Generate an SSH key and `add it as a Deploy key <https://github.com/vaexio/vaex-asv/settings/keys>`__ in the repo, so that the user can push new commits to this repo.
3. Clone the repos:

    git clone git@github.com:vaexio/vaex.git
    git clone git@github.com:vaexio/vaex-asv.git

4. Configure git user name and email to be used in benchmark result commits:

    git config --global user.name "Vaex GitHub Integration"
    git config --global user.email github@vaex.io

5. Install Miniconda and then install ASV:

    conda install asv -c conda-forge

6. Configure the machine parameters for ASV, they will be stored in `~/.asv-machine.json`:

    asv machine

7. Run ASV for the first time, to make sure that it works and to initialize the results history.

    cd vaex
    bin/benchmark_new_commits.sh --only-last

8. Add a cron job to run the benchmarks, e.g. every 8 hours:

    mkdir /home/github/log/
    crontab -e
    <add the following content in the editor>
    PATH=/home/github/miniconda3/bin:/home/github/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    0 */8 * * * /home/github/vaex/bin/benchmark_new_commits.sh >/home/github/log/benchmark-run-$(date "+\%Y-\%m-\%d_\%H-\%M-\%S").log 2>&1
