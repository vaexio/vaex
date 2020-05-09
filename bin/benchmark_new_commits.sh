#!/bin/bash
# Run benchmarks for the new commits on "master" and "bench*" branches;
# Save the results to the vaex-asv repo, and copy produced
# HTML results to the /var/www/asv.vaex.io/ folder on disk

set -e

VAEX_REPO_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/../"
cd "$VAEX_REPO_DIR"

# get all remote branches
git fetch --all

# force master to point to origin
git checkout master -f
git reset --hard origin/master


function restore_results_from_git () {
  # copy results from vaex-asv/.asv/results/**.json to vaex/.asv/results/**.json
  (cd ../vaex-asv && git pull -f)
  (mkdir -p .asv)
  (mkdir -p ../vaex-asv/.asv/results/)
  echo Restoring the following results from vaex-asv repo:
  find ../vaex-asv/.asv/results/
  cp -r ../vaex-asv/.asv/results/ .asv/
}

function store_and_push_results_in_git () {
  # copy results to vaex-asv/.asv/results/**.json
  echo Storing the following results into vaex-asv:
  find .asv/results
  (mkdir -p ../vaex-asv/.asv/results)
  cp -rf .asv/results ../vaex-asv/.asv
  # cp -rf .asv/html ../vaex-asv/asv/ (should we also do the HTML?)
  (cd ../vaex-asv && git add .asv && git commit -m 'Commit benchmark results' && git push)
}

function copy_results_to_www () {
  echo Copying HTML results to www
  cp -rf .asv/html/* /var/www/asv.vaex.io/html/
}

function get_additional_branches_to_benchmark () {
  # find branches which start with "bench*" in the "origin"
  git branch -a | grep origin/bench | sed -e 's/.*origin\/\(.*\)/\1/'
}


restore_results_from_git

if [[ "$1" == "--only-last" ]]
then

  echo Benchmarking last commit on master
  asv run

else

  echo Benchmarking new commits on master
  (asv run NEW) || (echo Maybe no new commits to benchmark?)

  for branch in $(get_additional_branches_to_benchmark)
  do
    echo Benchmarking commits on "${branch}"
    asv run origin/master..origin/${branch}
  done

fi

# generate HTML
asv publish

store_and_push_results_in_git
copy_results_to_www
