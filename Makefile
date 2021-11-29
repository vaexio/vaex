nose:
	nosetests --with-coverage --cover-package=vaex --cover-html test/dataset.py
# test/ui.py
coverage:
	-coverage run --parallel-mode --source=vaex -m vaex.test.ui
	-coverage run --parallel-mode --source=vaex -m vaex.test.dataset
	-coverage run --parallel-mode --source=vaex -m vaex.test.plot
	coverage combine
	#test/dataset.py
	coverage html -d cover
	open cover/index.html

test:
	VAEX_DEV=1 python -m pytest tests/

clean-test:
	rm -rf smæll2.parquet smæll2.yaml tests/data/parquet/ tests/data/smæll2.csv.hdf5 tests/data/unittest.parquet
	find . -wholename "./tests/data/parquet_dataset_partitioned_*" -delete

generate-docs:
	make -C ./docs/ html

init:
	python -m pip install -e .[dev]

uninstall:
	 @pip uninstall -y `pip list | grep -o -E "^vaex([-a-zA-Z0-9]+)?"` 2> /dev/null ||\
	 	echo "Can't uninstall: Vaex is not installed"

clean:
	make uninstall clean-test
	rm -rf packages/vaex-core/build/
	find . -wholename "*egg-info*" -delete

release:
	echo ""
