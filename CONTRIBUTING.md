# Contributing Guidelines

Thank you for wanting to explore and contribute to _vaex_!

Below are the instructions on how to setup a development version of vaex, but contributions to the documentation and issue-resolving are very welcome.

## Setup for Development

* First, clone (with submodules!) the repo:
    
    ``git clone --recursive https://github.com/vaexio/vaex && cd vaex``

* Next, create and activate a virtual python environment (using `conda` or `venv`).
 
* Finally, install vaex for development (this might take a long time):

    ``make init`` or ``pip install -e ".[dev]"``

### Common Errors

 * Did you clone with `--recursive`? If not, run `git submodule update --init`.
 * If you're on Windows, make sure that your command line/terminal has administrator privileges.

#### MacOS Compilation Error

If the installation failed, and the error said _vaex-core_ couldn't be installed, and somewhere in the (long) log you received an error which looks something like:

`error: $MACOSX_DEPLOYMENT_TARGET mismatch: now "10.9" but "10.15" during configure`

Try running the following line and retry to install:

```
export MACOSX_DEPLOYMENT_TARGET=`python -c "from distutils.util import get_platform;print(get_platform().split('-')[1], end='')"`
```

#### Nothing Works?

Did nothing work for you? Contact one of the repo's contributors for help.
If you have encountered and solved another installation issue, you are very welcome to add it do the documentation here and submit a PR.

## Submitting a Pull Request

We work using forks and Pull Requests.

In order to submit a PR for vaex, one must fork the repository and submit the PR from there.

Please make sure your fork is up-to-date with the vaex's latest _master_ branch.

