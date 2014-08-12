#include <Python.h>
#include <math.h>
#include <stdexcept>
#include <cstdio>
//#include <omp.h>

#include <numpy/arrayobject.h>




template<typename T>
void object_to_numpy1d_nocopy(T* &ptr, PyObject* obj, int &count, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 1)
			throw std::runtime_error("array is not 1d");
		int size = PyArray_DIMS(obj)[0];
		if((count >= 0) && (size != count))
			throw std::runtime_error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw std::runtime_error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		if(strides[0] != PyArray_ITEMSIZE(obj)) {
			throw std::runtime_error("stride is not 1");
		}
		
		ptr = (T*)PyArray_DATA(obj);
		count = size;
}

template<typename T>
void object_to_numpy2d_nocopy(T* &ptr, PyObject* obj, int &count_x, int &count_y, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 2)
			throw std::runtime_error("array is not 2d");
		int size_x = PyArray_DIMS(obj)[0];
		if((count_x >= 0) && (size_x != count_x))
			throw std::runtime_error("arrays not of equal size");
		int size_y = PyArray_DIMS(obj)[1];
		if((count_y >= 0) && (size_y != count_y))
			throw std::runtime_error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw std::runtime_error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		printf("strides: %d %d (%d %d)\n", strides[0],strides[1], size_x, size_y);
		if(strides[1] != PyArray_ITEMSIZE(obj)) {
			throw std::runtime_error("stride[0] is not 1");
		}
		if(strides[0] != PyArray_ITEMSIZE(obj)*size_x) {
			throw std::runtime_error("stride[1] is not 1");
		}
		
		ptr = (T*)PyArray_DATA(obj);
		count_x = size_x;
		count_y = size_y;
}




void range_check(double* const block_ptr, unsigned char * const mask_ptr, int length, double min, double max) {
	for(int i = 0; i < length; i++) {
		mask_ptr[i] = (block_ptr[i] > min) & (block_ptr[i] <= max);
	}
}

PyObject* range_check_(PyObject* self, PyObject *args) {
	PyObject* result = NULL;
	PyObject* block, *mask;
	double min, max;
	if(PyArg_ParseTuple(args, "OOdd", &block, &mask, &min, &max)) {
		int length = -1;
		double *block_ptr = NULL;
		unsigned char *mask_ptr = NULL;
		object_to_numpy1d_nocopy(block_ptr, block, length);
		object_to_numpy1d_nocopy(mask_ptr, mask, length, NPY_BOOL);
		range_check(block_ptr, mask_ptr, length, min, max);
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}



void find_nan_min_max(double* const block_ptr, int length, double &min_, double &max_) {
	double min = min_, max = max_;
	//*double min = min_, max = max_; // no using the reference but a local var seems easier for the compiler to optimize
	//printf("length: %d\n", length);
	 
	/*/
	int thread_index = 0;
	int counts = 0;
	double maxima[16], minima[16];
	double local_max, local_min;
	int k;
	#pragma omp parallel private(thread_index, counts, k, local_max, local_min)
	{
		thread_index = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		
		printf("threads: %d\n", nthreads);
		counts = 0;
		
		#pragma omp single
		{
			//maxima = (double*)malloc(sizeof(double) * nthreads);
			//minima = (double*)malloc(sizeof(double) * nthreads);
			for(int i = 0; i < nthreads; i++) {
				maxima[i] = block_ptr[0];
				minima[i] = block_ptr[0];
			}
		}
		local_min = minima[thread_index];
		local_max = maxima[thread_index];
		for(int k = 0; k < length/nthreads; k++) {
			double value = block_ptr[k*nthreads + thread_index];
			local_min = fmin(value, local_min);
			local_max = fmax(value, local_max);
		}
		//schedule(static, 1000)
		#pragma omp for
		for(int i = 0; i < length; i++) {
			//mask_ptr[i] = (block_ptr[i] > min) & (block_ptr[i] <= max);
			double value = block_ptr[i];
			//printf("%d %d\n", thread_index, i);
			//#if(!isnan(value)) {
			if(value == value) {  // nan checking
				//minima[thread_index] = (value < minima[thread_index]) ? value : minima[thread_index];
				//maxima[thread_index] = (value > maxima[thread_index]) ? value : maxima[thread_index];
				local_min = fmin(value, local_min);
				local_max = fmax(value, local_max);
			}
			counts++;
		}
		minima[thread_index] = local_min;
		maxima[thread_index] = local_max;
		
		printf("thread %d did %d\n", thread_index, counts);
		#pragma omp single
		{
			min = minima[0];
			max = maxima[0];
			for(int i = 1; i < nthreads; i++) {
				min = fmin(minima[i], min);
				max = fmax(maxima[i], max);
			}
			//free(maxima);
			//free(minima);
		}

	}
	/*/
	for(int i = 0; i < length; i++) {
		double value = block_ptr[i];
		if(value == value) {  // nan checking
			min = fmin(value, min);
			max = fmax(value, max);
		}
	}
	/**/
	min_ = min;
	max_ = max;
}





PyObject* find_nan_min_max_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject* block;
	if(PyArg_ParseTuple(args, "O", &block)) {

		int length = -1;
		double *block_ptr = NULL;
		double min=0., max=1.;
		object_to_numpy1d_nocopy(block_ptr, block, length);
		find_nan_min_max(block_ptr, length, min, max);
		result = Py_BuildValue("dd", min, max); 
	}
	return result;
}


void histogram1d(const double* const block, const double* const weights, int block_length, double* counts, int counts_length, double min, double max) {
	for(int i = 0; i < block_length; i++) {
		double value = block[i];
		double scaled = (value - min) / (max-min);
		int index = (int)(scaled * counts_length);
		if( (index >= 0) & (index < counts_length) )
			counts[index] += weights == NULL ? 1 : weights[i];
	}
}

PyObject* histogram1d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* block, *weights, *counts;
	double min, max;
	if(PyArg_ParseTuple(args, "OOOdd", &block, &weights, &counts, &min, &max)) {
		int block_length = -1;
		int counts_length = -1;
		double *block_ptr = NULL;
		double *counts_ptr = NULL;
		double *weights_ptr = NULL;
		object_to_numpy1d_nocopy(block_ptr, block, block_length);
		object_to_numpy1d_nocopy(counts_ptr, counts, counts_length);
		if(weights != Py_None) {
			object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
		}
		histogram1d(block_ptr, weights_ptr, block_length, counts_ptr, counts_length, min, max);
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}


void histogram2d(const double* const blockx, const double* const blocky, const double* const weights, int block_length, double* counts, int counts_length_x, int counts_length_y, double xmin, double xmax, double ymin, double ymax) {
	for(int i = 0; i < block_length; i++) {
		double value_x = blockx[i];
		double scaled_x = (value_x - xmin) / (xmax-xmin);
		int index_x = (int)(scaled_x * counts_length_x);

		double value_y = blocky[i];
		double scaled_y = (value_y - ymin) / (ymax-ymin);
		int index_y = (int)(scaled_y * counts_length_y);
		if( (index_x >= 0) & (index_x < counts_length_x)  & (index_y >= 0) & (index_y < counts_length_y) )
			counts[index_y + counts_length_y*index_x] += weights == NULL ? 1 : weights[i];
	}
}

PyObject* histogram2d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* blockx, *blocky, *weights, *counts;
	double xmin, xmax, ymin, ymax;
	if(PyArg_ParseTuple(args, "OOOOdddd", &blockx, &blocky, &weights, &counts, &xmin, &xmax, &ymin, &ymax)) {
		int block_length = -1;
		int counts_length_x = -1;
		int counts_length_y = -1;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		double *weights_ptr = NULL;
		double *counts_ptr = NULL;
		object_to_numpy1d_nocopy(blockx_ptr, blockx, block_length);
		object_to_numpy1d_nocopy(blocky_ptr, blocky, block_length);
		object_to_numpy2d_nocopy(counts_ptr, counts, counts_length_x, counts_length_y);
		if(weights != Py_None) {
			object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
		}
		histogram2d(blockx_ptr, blocky_ptr, weights_ptr, block_length, counts_ptr, counts_length_x, counts_length_y, xmin, xmax, ymin, ymax);
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}


void pnpoly(double *vertx, double *verty, int nvert, const double* const blockx, const double* const blocky, unsigned char* const mask, int length, double meanx, double meany, double radius) {
	double radius_squared = radius*radius;
	for(int k= 0; k < length; k++){
		double testx = blockx[k];
		double testy = blocky[k];
		int i, j, c = 0;
		mask[k] = 0;
		double distancesq = pow(testx - meanx, 2) + pow(testy - meany, 2);
		if(distancesq < radius_squared) 
		{
			for (i = 0, j = nvert-1; i < nvert; j = i++) {
				if ( ((verty[i]>testy) != (verty[j]>testy)) &&
					(testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
					c = !c;
			}
				mask[k] = c;
		}
	}
}

static PyObject* pnpoly_(PyObject* self, PyObject *args) {
//object x, object y, object blockx, object blocky, object mask, double meanx, double meany, double radius)
	PyObject* result = NULL;
	PyObject *x, *y, *blockx, *blocky, *mask;
	double meanx, meany, radius;
	if(PyArg_ParseTuple(args, "OOOOOddd", &x, &y, &blockx, &blocky, &mask, &meanx, &meany, &radius)) {
		unsigned char *mask_ptr = NULL;
		int polygon_length = -1, length = -1;
		double *x_ptr = NULL;
		double *y_ptr = NULL;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		object_to_numpy1d_nocopy(x_ptr, x, polygon_length);
		object_to_numpy1d_nocopy(y_ptr, y, polygon_length);
		object_to_numpy1d_nocopy(blockx_ptr, blockx, length);
		object_to_numpy1d_nocopy(blocky_ptr, blocky, length);
		object_to_numpy1d_nocopy(mask_ptr, mask, length, NPY_BOOL);
		pnpoly(x_ptr, y_ptr, polygon_length, blockx_ptr, blocky_ptr, mask_ptr, length, meanx, meany, radius);
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}



static PyMethodDef pygavi_functions[] = {
        {"histogram1d", (PyCFunction)histogram1d_, METH_VARARGS, ""},
        {"histogram2d", (PyCFunction)histogram2d_, METH_VARARGS, ""},
        {"find_nan_min_max", (PyCFunction)find_nan_min_max_, METH_VARARGS, ""},
        {"pnpoly", (PyCFunction)pnpoly_, METH_VARARGS, ""},
        {"range_check", (PyCFunction)range_check_, METH_VARARGS, ""},
    { NULL, NULL, 0 }
};


PyMODINIT_FUNC
initgavifast(void)
{
	PyObject *mod;
	//PyObject *c_api_object;

	///import_libnumarray();
	import_array();

	mod = Py_InitModule("gavifast", pygavi_functions);
	//INIT_TYPE(PyKaplotImage_Type);
	//PyModule_AddObject(mod, "Image",  (PyObject *)&PyKaplotImage_Type);

	//PyKaplotFont_API[0] = (void *)PyKaplot_Font_Outline;
	//c_api_object = PyCObject_FromVoidPtr((void *)&api, NULL);
	//if (c_api_object != NULL)
	//      PyModule_AddObject(mod, "_C_API", c_api_object);
}