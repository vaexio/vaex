#!/bin/bash
# Run benchmarks for the new commits on "master" and "bench*" branches;
# Save the results to the "gh-pages" branch, and copy produced
# HTML results to the /var/www/asv.vaex.io/ folder on disk

set -e

VAEX_REPO_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/../"
cd "$VAEX_REPO_DIR"
git pull origin master

function restore_results_from_git () {
  # copy results from gh-pages/benchmarks/results/**.json to master/.asv/results/**.json
  (git checkout origin/gh-pages -- benchmarks/results/ && git reset -- benchmarks/results/) ||
  (mkdir -p benchmarks/results/)
  echo Restoring the following results from git branch:
  find benchmarks/results/
  cp -r benchmarks/results/ .asv/
}

function store_and_push_results_in_git () {
  # copy results to gh-pages/benchmarks/results/**.json and HTML to gh-pages/benchmarks/index.html
  echo Storing the following results into git branch:
  find .asv/results
  git checkout -f gh-pages
  cp -rf .asv/results benchmarks/
  cp -rf .asv/html/* benchmarks/
  git add benchmarks
  git commit -m 'Commit benchmark results'
  git push
  git checkout -
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
