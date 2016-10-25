 * Make sure unittest run (on travis) (and for the moment run the gui unittest locally)
 * Change version number in vaex/version.py
 * $ git commit -m "bumbed version number" vaex/version.py && git push 
 * $ git tag \`python -c 'import vaex as vx; print(vx.version.versionstring)'\` && git push --tags
 * $ python setup.py sdist upload
 * From vaex-wheels
   * Edit .travis.yml to reflect version #
   * git commit -m "new release" .travis.yml && git push
 * twine upload ~/Dropbox/Apps/vaex/vaex-wheels/wheelhouse/vaex-X*
 * Download the tarball, or use the one that was uploaded, and calculate the sth like
 * openssl sha256 dist/vaex-1.0.0b3.tar.gz
 * Edit vaex-feedstock/recipe/meta.yaml
  * Add sha hash
  * Edit version number
  * Set build.number to 0

