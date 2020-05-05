# Make vaex releases

## Required tool
 * pip install releash

## General idea

The vaex repo is a mono-repo, meaning that it contains many Python packages in the `packages` directory. We use the [releash](https://github.com/maartenbreddels/releash) tool to orchestrate this.

## Get status

Get an overview of the status of all packages by running `$ releash status`

## Example

Releasing a new version of vaex-core

### Bump version number
    $ releash bump vaex-core -n  # dry run
    $ releash bump vaex-core -n --what=minor  # to go from 2.3.x to 2.4.0
    $ releash bump vaex-core -f --what=minor  # force it, in case the worktree is dirty

### Push

By pushing the tag (that releash created for you)

    $ git push && git push --tags

It will trigger the GitHub Action to build the wheels and upload them to pypi.

