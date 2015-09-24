__author__ = 'breddels'


def pca(dataset, expressions, jobs_manager):
	# pca
	#  - first calculate mean
	#  - subtract means
	#   - do cov matrix,
	means = jobs_manager.calculate_mean(dataset, *expressions)
	#print "means", means
	#dsa
	cov_expressions = []
	N = len(columns)
	for i in range(N-1):
		for j in range(i+1,N):
			expression = "((%s)-%40f) * ((%s)-%40f)" % (columns[i], means[i], columns[j], means[j])
			cov_expressions.append(expression)

	#jobs_manager.calculate_pca(columns, means)

