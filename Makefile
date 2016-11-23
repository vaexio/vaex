


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
	python test/dataset.py

release:
	echo ""
