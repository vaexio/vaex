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
		int size_x = PyArray_DIMS(obj)[1];
		if((count_x >= 0) && (size_x != count_x))
			throw std::runtime_error("arrays not of equal size");
		int size_y = PyArray_DIMS(obj)[0];
		if((count_y >= 0) && (size_y != count_y))
			throw std::runtime_error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw std::runtime_error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		//printf("strides: %d %d (%d %d)\n", strides[0],strides[1], size_x, size_y);
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

template<typename T>
void object_to_numpy3d_nocopy(T* &ptr, PyObject* obj, int &count_x, int &count_y, int &count_z, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 3)
			throw std::runtime_error("array is not 3d");
		int size_x = PyArray_DIMS(obj)[2];
		if((count_x >= 0) && (size_x != count_x))
			throw std::runtime_error("arrays not of equal size");
		int size_y = PyArray_DIMS(obj)[1];
		if((count_y >= 0) && (size_y != count_y))
			throw std::runtime_error("arrays not of equal size");
		int size_z = PyArray_DIMS(obj)[0];
		if((count_z >= 0) && (size_z != count_z))
			throw std::runtime_error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw std::runtime_error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		//printf("strides: %d %d %d(%d %d %d)\n", strides[0], strides[1], strides[2], size_x, size_y, size_z);
		if(strides[2] != PyArray_ITEMSIZE(obj)) {
			throw std::runtime_error("stride[0] is not 1");
		}
		if(strides[1] != PyArray_ITEMSIZE(obj)*size_y) {
			throw std::runtime_error("stride[1] is not 1");
		}
		if(strides[0] != PyArray_ITEMSIZE(obj)*size_y*size_x) {
			throw std::runtime_error("stride[2] is not 1");
		}
		
		ptr = (T*)PyArray_DATA(obj);
		count_x = size_x;
		count_y = size_y;
		count_z = size_z;
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
		Py_BEGIN_ALLOW_THREADS
		range_check(block_ptr, mask_ptr, length, min, max);
		Py_END_ALLOW_THREADS
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
	min = block_ptr[0];
	max = block_ptr[0];
	for(int i = 1; i < length; i++) {
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
		Py_BEGIN_ALLOW_THREADS
		find_nan_min_max(block_ptr, length, min, max);
		Py_END_ALLOW_THREADS
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
		Py_BEGIN_ALLOW_THREADS
		histogram1d(block_ptr, weights_ptr, block_length, counts_ptr, counts_length, min, max);
		Py_END_ALLOW_THREADS
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}



void histogram2d(const double* const blockx, const double* const blocky, const double* const weights, const int block_length, double* const counts, const int counts_length_x, const int counts_length_y, const double xmin, const double xmax, const double ymin, const double ymax, const long long offset_x, const long long offset_y) {
	long long i_x = offset_x;
	long long i_y = offset_y;
	for(long long i = 0; i < block_length; i++) {
		//double value_x = blockx[(i+ offset_x + block_length)  % block_length];
		//double value_x = blockx[i];
		double value_x = blockx[i_x];
		double scaled_x = (value_x - xmin) / (xmax-xmin);
		int index_x = (int)(scaled_x * counts_length_x);

		//double value_y = blocky[(i+ offset_y + block_length)  % block_length];
 		//double value_y = blocky[i];

		/*
		if( (index_x >= 0) & (index_x < counts_length_x)  & (index_y >= 0) & (index_y < counts_length_y) )
			counts[index_y + counts_length_y*index_x] += weights == NULL ? 1 : weights[i];
		*/
		if( (index_x >= 0) & (index_x < counts_length_x)) {
			double value_y = blocky[i_y];
			
			double scaled_y = (value_y - ymin) / (ymax-ymin);
			int index_y = (int)(scaled_y * counts_length_y);
			if ( (index_y >= 0) & (index_y < counts_length_y) ) {
				counts[index_x + counts_length_x*index_y] += weights == NULL ? 1 : weights[i];
			}
		}
		i_x = i_x >= block_length-1 ? 0 : i_x+1;
		i_y = i_y >= block_length-1 ? 0 : i_y+1;
	}
}

PyObject* histogram2d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* blockx, *blocky, *weights, *counts;
	double xmin, xmax, ymin, ymax;
	long long offset_x = 0;
	long long offset_y = 0;
	if(PyArg_ParseTuple(args, "OOOOdddd|LL", &blockx, &blocky, &weights, &counts, &xmin, &xmax, &ymin, &ymax, &offset_x, &offset_y)) {
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
		Py_BEGIN_ALLOW_THREADS
		histogram2d(blockx_ptr, blocky_ptr, weights_ptr, block_length, counts_ptr, counts_length_x, counts_length_y, xmin, xmax, ymin, ymax, offset_x, offset_y);
		Py_END_ALLOW_THREADS
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}

void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, double* counts, int counts_length_x, int counts_length_y, int counts_length_z, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, long long const offset_x, long long const offset_y, long long const offset_z) {
	for(long long i = 0; i < block_length; i++) {
		double value_x = blockx[(i+ offset_x + block_length)  % block_length];
		double scaled_x = (value_x - xmin) / (xmax-xmin);
		int index_x = (int)(scaled_x * counts_length_x);

		double value_y = blocky[(i+ offset_y + block_length)  % block_length];
		double scaled_y = (value_y - ymin) / (ymax-ymin);
		int index_y = (int)(scaled_y * counts_length_y);
		
		double value_z = blockz[(i+ offset_z + block_length)  % block_length];
		double scaled_z = (value_z - zmin) / (zmax-zmin);
		int index_z = (int)(scaled_z * counts_length_z);
		
		if( (index_x >= 0) & (index_x < counts_length_x)  & (index_y >= 0) & (index_y < counts_length_y)  & (index_z >= 0) & (index_z < counts_length_z) )
			//counts[index_z + counts_length_z*index_y + counts_length_z*counts_length_y*index_x] += weights == NULL ? 1 : weights[i];
			counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += weights == NULL ? 1 : weights[i];
	}
}

PyObject* histogram3d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* blockx, *blocky, *blockz, *weights, *counts;
	double xmin, xmax, ymin, ymax, zmin, zmax;
	long long offset_x = 0, offset_y = 0, offset_z = 0;
	if(PyArg_ParseTuple(args, "OOOOOdddddd|LLL", &blockx, &blocky, &blockz, &weights, &counts, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &ymax, &offset_x, &offset_y, &offset_z)) {
		int block_length = -1;
		int counts_length_x = -1;
		int counts_length_y = -1;
		int counts_length_z = -1;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		double *blockz_ptr = NULL;
		double *weights_ptr = NULL;
		double *counts_ptr = NULL;
		object_to_numpy1d_nocopy(blockx_ptr, blockx, block_length);
		object_to_numpy1d_nocopy(blocky_ptr, blocky, block_length);
		object_to_numpy1d_nocopy(blockz_ptr, blockz, block_length);
		object_to_numpy3d_nocopy(counts_ptr, counts, counts_length_x, counts_length_y, counts_length_z);
		if(weights != Py_None) {
			object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
		}
		Py_BEGIN_ALLOW_THREADS
		histogram3d(blockx_ptr, blocky_ptr, blockz_ptr, weights_ptr, block_length, counts_ptr, counts_length_x, counts_length_y, counts_length_z, xmin, xmax, ymin, ymax, zmin, zmax, offset_x, offset_y, offset_z);
		Py_END_ALLOW_THREADS
		Py_INCREF(Py_None);
		result = Py_None;
	}
	return result;
}

void project(double* cube_, const int cube_length_x, const int cube_length_y, const int cube_length_z, double* surface_, const int surface_length_x, const int surface_length_y, const double* const projection_, const double* const offset_)
{
	double* const surface = surface_;
	const double* const cube = cube_;
	const double* const projection = projection_;
	const double* const offset = offset_;
	for(int i = 0; i < cube_length_x; i++) {
	for(int j = 0; j < cube_length_y; j++) {
	for(int k = 0; k < cube_length_z; k++) {
			const double x = projection[0]*(i+offset[0]) + projection[1]*(j+offset[1]) + projection[2]*(k+offset[2]) + projection[3];
			const double y = projection[4]*(i+offset[0]) + projection[5]*(j+offset[1]) + projection[6]*(k+offset[2]) + projection[7];
			const int binNox = int(x);
			const int binNoy = int(y);
			if( (binNox >= 0) && (binNox < surface_length_x) && (binNoy >= 0) && (binNoy < surface_length_y))
				surface[binNox + binNoy*surface_length_x] += cube[i + cube_length_x*j + cube_length_x*cube_length_y*k];
	}}}
}

PyObject* project_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* cube, *surface, *projection, *offset;
	if(PyArg_ParseTuple(args, "OOOO", &cube, &surface, &projection, &offset)) {
		int cube_length_x = -1;
		int cube_length_y = -1;
		int cube_length_z = -1;
		double *cube_ptr = NULL;
		
		int surface_length_x = -1;
		int surface_length_y = -1;
		double *surface_ptr = NULL;
		
		int projection_length = -1;
		double *projection_ptr = NULL;

		int offset_length = -1;
		double *offset_ptr = NULL;

		object_to_numpy3d_nocopy(cube_ptr, cube, cube_length_x, cube_length_y, cube_length_z);
		object_to_numpy2d_nocopy(surface_ptr, surface, surface_length_x, surface_length_y);
		object_to_numpy1d_nocopy(projection_ptr, projection, projection_length);
		object_to_numpy1d_nocopy(offset_ptr, offset, offset_length);
		//if(weights != Py_None) {
		//	object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
		//}
		if(projection_length != 8)
			throw std::runtime_error("projection array should be of length 8");
		if(offset_length != 3)
			throw std::runtime_error("center array should be of length 3");
		Py_BEGIN_ALLOW_THREADS
		project(cube_ptr, cube_length_x, cube_length_y, cube_length_z, surface_ptr, surface_length_x, surface_length_y, projection_ptr, offset_ptr);
		Py_END_ALLOW_THREADS
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
		Py_BEGIN_ALLOW_THREADS
		pnpoly(x_ptr, y_ptr, polygon_length, blockx_ptr, blocky_ptr, mask_ptr, length, meanx, meany, radius);
		Py_END_ALLOW_THREADS
		Py_INCREF(Py_None);
		result = Py_None;
	}
	
	return result;
}





static PyMethodDef pygavi_functions[] = {
        {"histogram1d", (PyCFunction)histogram1d_, METH_VARARGS, ""},
        {"histogram2d", (PyCFunction)histogram2d_, METH_VARARGS, ""},
        {"histogram3d", (PyCFunction)histogram3d_, METH_VARARGS, ""},
        {"project", (PyCFunction)project_, METH_VARARGS, ""},
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




