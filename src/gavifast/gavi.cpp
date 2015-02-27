#include <Python.h>
#include <math.h>
#include <stdexcept>
#include <cstdio>
//#include <omp.h>

#include <numpy/arrayobject.h>


// from http://stackoverflow.com/questions/12261915/howto-throw-stdexceptions-with-variable-messages
struct Error : std::exception
{
    char text[1000];

    Error(char const* fmt, ...) __attribute__((format(printf,2,3))) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(text, sizeof text, fmt, ap);
        va_end(ap);
    }

    char const* what() const throw() { return text; }
};

int stride_default = 1;

template<typename T>
void object_to_numpy1d_nocopy(T* &ptr, PyObject* obj, long long  &count, int& stride=stride_default, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 1)
			throw std::runtime_error("array is not 1d");
		long long size = PyArray_DIMS(obj)[0];
		if((count >= 0) && (size != count))
			throw std::runtime_error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw std::runtime_error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		if(stride == -1) {
			stride = strides[0];
		} else {
			if(strides[0] != stride*PyArray_ITEMSIZE(obj)) {
				throw Error("stride is not equal to %d", stride);
			}
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
template<typename T>
void object_to_numpyNd_nocopy(T* &ptr, PyObject* obj, int max_dimension, int& dimension, int* sizes, int* strides, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		//printf("dim = %i maxdim = %i %i\n", dimension, max_dimension,  (int)PyArray_NDIM(obj));
		dimension = (int)PyArray_NDIM(obj);
		if(dimension > max_dimension) {
			printf("dim = %i maxdim = %i\n", dimension, max_dimension);
			throw std::runtime_error("array dimension is bigger than allowed");
		}

		for(int i = 0; i < dimension; i++) {
			sizes[i] = PyArray_DIMS(obj)[i];
			strides[i] = PyArray_STRIDES(obj)[i];
		}
		ptr = (T*)PyArray_DATA(obj);
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
		long long length = -1;
		double *block_ptr = NULL;
		unsigned char *mask_ptr = NULL;
		try {
			object_to_numpy1d_nocopy(block_ptr, block, length);
			object_to_numpy1d_nocopy(mask_ptr, mask, length, stride_default, NPY_BOOL);
			Py_BEGIN_ALLOW_THREADS
			range_check(block_ptr, mask_ptr, length, min, max);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		} catch(...) {
			PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}
	return result;
}



void find_nan_min_max(const double* const block_ptr, const long long length, double &min_, double &max_) {
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
	for(long long i = 1; i < length; i++) {
		if(isnan(min))
			min = block_ptr[i];
		else
			break;
	}
	for(long long i = 1; i < length; i++) {
		if(isnan(max))
			max = block_ptr[i];
		else
			break;
	}
	for(long long i = 1; i < length; i++) {
		const double value = block_ptr[i];
		//if(value == value)// nan checking
		{  
			//min = fmin(value, min);
			//max = fmax(value, max);
			if(value < min) {
				min = value;
			} else if (value > max) {
				max = value;
			}
			//min = value < min ? value : min;
			//max = value > max ? value : max;
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

		long long length = -1;
		double *block_ptr = NULL;
		double min=0., max=1.;
		try {
			object_to_numpy1d_nocopy(block_ptr, block, length);
			Py_BEGIN_ALLOW_THREADS
			find_nan_min_max(block_ptr, length, min, max);
			Py_END_ALLOW_THREADS
			result = Py_BuildValue("dd", min, max); 
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}




void histogram1d(const double* const __restrict__ block, const long long block_stride, const double* const weights, const int weights_stride, long long block_length, double* __restrict__ counts, const int counts_length, const double min, const double max) {
	const double* __restrict__ block_ptr = block;
	

	/*
	
	const int BLOCK_SIZE = (2<<10);
	long long int indices[BLOCK_SIZE];
	
	int subblocks = int(block_length * 1./ BLOCK_SIZE + 1.);
	for(long long b = 0; b < subblocks ; b++) {
		long long i1 = b * BLOCK_SIZE;
		long long i2 = (b+1) * BLOCK_SIZE;
		if(i2 > block_length) i2 = block_length;
		long long length = i2-i1;
		long long index_count = 0;
		if(weights == NULL) {
			for(long long i = 0; i < length; i++) {
				const double value = block[i];
				//block_ptr += block_stride;
				const double scaled = (value - min) / (max-min);
				const long long index = (long long)(scaled * counts_length);
				if( (index >= 0) & (index < counts_length) )
					indices[index_count++] = index;
			}
			for(long long i = 0; i < index_count; i++) {
				const long long index = indices[i];
				counts[index] += 1;
			}
		} else {
			for(long long i = 0; i < length; i++) {
				const double value = block[i];
				//block_ptr += block_stride;
				const double scaled = (value - min) / (max-min);
				const long long index = (long long)(scaled * counts_length);
				indices[index_count++] = index;
			}
			for(long long i = 0; i < index_count; i++) {
				const long long index = indices[i];
				if( (index >= 0) & (index < counts_length) )
					counts[index] += weights[i1+i];
			}
		}
	}
	/*/
	
	const double scale = counts_length / (max-min);;
	for(long long i = 0; i < block_length; i++) {
		const double value = block[i]; //block[i*block_stride];
		//block_ptr++;
		//const double value = *block_ptr;
		//block_ptr += block_stride;
		//__builtin_prefetch(block_ptr, 1, 1); // read, and no temporal locality
		//__builtin_prefetch(block_ptr+block_stride*10, 1, 1); // read, and no temporal locality
		//if( (index >= 0) & (index < counts_length) )
		if((value > min) & (value < max)) {
			const double scaled = (value - min) * scale;
			const long long index = (long long)(scaled);
			counts[index] += weights == NULL ? 1 : weights[i];
		}
	}
	/**/
}




PyObject* histogram1d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* block, *weights, *counts;
	double min, max;
	if(PyArg_ParseTuple(args, "OOOdd", &block, &weights, &counts, &min, &max)) {
		long long block_length = -1;
		long long counts_length = -1;
		double *block_ptr = NULL;
		int block_stride = -1;
		double *counts_ptr = NULL;
		
		double *weights_ptr = NULL;
		int weights_stride = -1;
		try {
			object_to_numpy1d_nocopy(block_ptr, block, block_length, block_stride);
			object_to_numpy1d_nocopy(counts_ptr, counts, counts_length, weights_stride);
			if(weights != Py_None) {
				object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
			}
			Py_BEGIN_ALLOW_THREADS
			histogram1d(block_ptr, block_stride, weights_ptr, weights_stride, block_length, counts_ptr, counts_length, min, max);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}



void histogram2d(const double* const __restrict__ blockx, const double* const __restrict__ blocky, const double* const weights, const long long block_length, double* const __restrict__ counts, const int counts_length_x, const int counts_length_y, const double xmin, const double xmax, const double ymin, const double ymax, const long long offset_x, const long long offset_y) {
	long long i_x = offset_x;
	long long i_y = offset_y;
	const double scale_x = counts_length_x/ (xmax-xmin);
	const double scale_y = counts_length_y/ (ymax-ymin);
	if((weights == NULL) & (offset_x == 0) & (offset_y == 0)) { // default: fasted algo
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx[i];
			double value_y = blocky[i];
			
			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) ) {
			//if( (index_x >= 0) & (index_x < counts_length_x) &  (index_y >= 0) & (index_y < counts_length_y) ) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				counts[index_x + counts_length_x*index_y] += 1;
			}
		}
	} else {
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx[i];
			double value_y = blocky[i];
			
			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) ) {
			//if( (index_x >= 0) & (index_x < counts_length_x) &  (index_y >= 0) & (index_y < counts_length_y) ) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				counts[index_x + counts_length_x*index_y] += weights == NULL ? 1 : weights[i];
			}
			i_x = i_x >= block_length-1 ? 0 : i_x+1;
			i_y = i_y >= block_length-1 ? 0 : i_y+1;
		}
	}
	/**/
}

PyObject* histogram2d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* blockx, *blocky, *weights, *counts;
	double xmin, xmax, ymin, ymax;
	long long offset_x = 0;
	long long offset_y = 0;
	if(PyArg_ParseTuple(args, "OOOOdddd|LL", &blockx, &blocky, &weights, &counts, &xmin, &xmax, &ymin, &ymax, &offset_x, &offset_y)) {
		long long block_length = -1;
		int counts_length_x = -1;
		int counts_length_y = -1;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		double *weights_ptr = NULL;
		double *counts_ptr = NULL;
		try {
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
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, double* counts, const int counts_length_x, const int counts_length_y, const int counts_length_z, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, long long const offset_x, long long const offset_y, long long const offset_z) {
	/*for(long long i = 0; i < block_length; i++) {
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
	}*/
	long long i_x = offset_x;
	long long i_y = offset_y;
	long long i_z = offset_z;
	const double scale_x = counts_length_x/ (xmax-xmin);
	const double scale_y = counts_length_y/ (ymax-ymin);
	const double scale_z = counts_length_z/ (zmax-zmin);
	if((weights == NULL) & (offset_x == 0) & (offset_y == 0) & (offset_z == 0)) { // default: fasted algo
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx[i];
			double value_y = blocky[i];
			double value_z = blockz[i];
			
			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) & (value_z >= zmin) & (value_z < zmax)) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				int index_z = (int)((value_z - zmin) * scale_z);
				counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += 1;
			}
		}
	} else {
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx[i];
			double value_y = blocky[i];
			double value_z = blockz[i];
			
			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) & (value_z >= zmin) & (value_z < zmax)) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				int index_z = (int)((value_z - zmin) * scale_z);
				counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += weights[i];
			}
		}
		/*for(long long i = 0; i < block_length; i++) {
			double value_x = blockx[i_x];
			int index_x = (int)((value_x - xmin) * scale_x);
			
			if( (index_x >= 0) & (index_x < counts_length_x)) {
				double value_y = blocky[i_y];
				int index_y = (int)((value_y - ymin) * scale_y);
				
				if ( (index_y >= 0) & (index_y < counts_length_y) ) {
					double value_z = blockz[i_z];
					int index_z = (int)((value_z - zmin) * scale_z);
					if ( (index_z >= 0) & (index_z < counts_length_z) ) {
						counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += weights == NULL ? 1 : weights[i];
						if(!((weights[i] == 0) | (weights[i] == 1)))
							printf("%d %d %d %f\n", i_x, i_y, i_z, weights[i]);
					}
				}
			}
			
			i_x = i_x >= block_length-1 ? 0 : i_x+1;
			i_y = i_y >= block_length-1 ? 0 : i_y+1;
			i_z = i_z >= block_length-1 ? 0 : i_z+1;
		}*/
	}	
}

PyObject* histogram3d_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject* blockx, *blocky, *blockz, *weights, *counts;
	double xmin, xmax, ymin, ymax, zmin, zmax;
	long long offset_x = 0, offset_y = 0, offset_z = 0;
	if(PyArg_ParseTuple(args, "OOOOOdddddd|LLL", &blockx, &blocky, &blockz, &weights, &counts, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &offset_x, &offset_y, &offset_z)) {
		long long block_length = -1;
		int counts_length_x = -1;
		int counts_length_y = -1;
		int counts_length_z = -1;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		double *blockz_ptr = NULL;
		double *weights_ptr = NULL;
		double *counts_ptr = NULL;
		try {
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
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
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
		
		long long projection_length = -1;
		double *projection_ptr = NULL;

		long long offset_length = -1;
		double *offset_ptr = NULL;

		try {
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
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
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
		long long polygon_length = -1, length = -1;
		double *x_ptr = NULL;
		double *y_ptr = NULL;
		double *blockx_ptr = NULL;
		double *blocky_ptr = NULL;
		try {
			object_to_numpy1d_nocopy(x_ptr, x, polygon_length);
			object_to_numpy1d_nocopy(y_ptr, y, polygon_length);
			object_to_numpy1d_nocopy(blockx_ptr, blockx, length);
			object_to_numpy1d_nocopy(blocky_ptr, blocky, length);
			object_to_numpy1d_nocopy(mask_ptr, mask, length, stride_default, NPY_BOOL);
			Py_BEGIN_ALLOW_THREADS
			pnpoly(x_ptr, y_ptr, polygon_length, blockx_ptr, blocky_ptr, mask_ptr, length, meanx, meany, radius);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	
	return result;
}


// from http://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}

void soneira_peebles(double* coordinates, double center, double width, double lambda, int eta, int level, int max_level) {
	int level_left = max_level - level;
	long long seperation = ipow(eta, level_left);
	//printf("level: %d of %d seperation: %d\n", level, max_level, seperation);

	for(int i = 0; i < eta; i++) {
		double pos =  ((double) rand() / (RAND_MAX)) * width - width/2+ center;
		if(level == max_level) {
			coordinates[i] = pos;
		} else {
			soneira_peebles(coordinates+seperation*i, pos, width / lambda, lambda, eta, level+1, max_level);
		}
	}
}



static PyObject* soneira_peebles_(PyObject* self, PyObject *args) {
//object x, object y, object blockx, object blocky, object mask, double meanx, double meany, double radius)
	PyObject* result = NULL;
	PyObject *coordinates;
	double lambda, center, width;
	int eta, max_level;
	if(PyArg_ParseTuple(args, "Odddii", &coordinates, &center, &width, &lambda, &eta, &max_level)) {
		long long length = -1;
		double *coordinates_ptr = NULL;
		try {
			object_to_numpy1d_nocopy(coordinates_ptr, coordinates, length);
			if(length != pow(eta, max_level))
				throw Error("length of coordinates != eta**max_level (%d != %d)", length, pow(eta, max_level));
			Py_BEGIN_ALLOW_THREADS
			soneira_peebles(coordinates_ptr, center, width, lambda, eta, 1, max_level);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
			//PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}
	
	return result;
}

#ifdef  __APPLE__
#else

#include <iostream>
//#include <chrono>
#include <random>
#endif

/*
Implements:
http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle "The "inside-out" algorithm"

Fill array of length 'length ' with a shuffled sequence of numbers between 0 and length-1.
important to use a 64 bit rng, mt19937_64 seems to be the only one
*/
void shuffled_sequence_(long long * array, long long length) {
#ifdef  __APPLE__
	for(long long i=0; i < length; i++) {
		uint_fast64_t r = rand(); 
		uint_fast64_t j =  r * i / RAND_MAX;
		array[i] = array[j];
		array[j] = i;
	}
#else
	auto rnd = std::mt19937_64(std::random_device{}());
	std::uniform_int_distribution<long long> dist(0, length-1); 
	for(long long i=0; i < length; i++) {
		uint_fast64_t r = dist(rnd);
		uint_fast64_t j = r * i / (length-1);
		array[i] = array[j];
		array[j] = i;
		if( ((i% 10000000) == 0) ){
			printf("%lld out of %lld (%.2f%%)\n", i, length, (i*100./length));
			fflush(stdout);
		}
		//printf("r=%d\n", r);
		//for(long long k=0; k < i+1; k++)
		//	printf(" %d", array[k]);
		//printf("\n");
	}
#endif
	
}

static PyObject* shuffled_sequence_(PyObject* self, PyObject *args) {
	PyObject* result = NULL;
	PyObject *array;
	if(PyArg_ParseTuple(args, "O", &array)) {
		long long length = -1;
		long long *array_ptr = NULL;
		try {
			object_to_numpy1d_nocopy(array_ptr, array, length, stride_default, NPY_INT64);
			Py_BEGIN_ALLOW_THREADS
			shuffled_sequence_(array_ptr, length);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
			//PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}
	
	return result;
}


void resize(double* source, int size, int dimension, double* target, int new_size)
{
	/*
	 * Reshape a N dimensional grid to a smaller size
	 * in input grid should have all dimensions of equals size, and size should be a power of two
	 * the output grid should have similar properties, but the new size should be equal or smaller 
	 * (and again a power of two)
	 * 
	 * algo works by using a 1d index for the output grid, each value of the output grid is the sum
	 * of the hybercube of the input grid. The 1d index together with a 1d index in the hypercube (blockindex)
	 * index the input grid.
	 * Example:
	 *  input grid is 16x16, output grid is 8x8, so each pixel in the 8x8 block is replaced by the sum
	 *   of 2x2 blocks in the input grid.
	 */
	int block_index = 0;
	int target_index = 0;
	int block_end_index = 1;
	int target_end_index = 1;
	// TODO: check if both are a power of two, and new_size <= size
	if(new_size > size)
		throw std::runtime_error("target size should be smaller than source size");
	int block_size_1d = size/new_size;
	
	for(int i = 0; i < dimension; i++) {
		block_end_index *= block_size_1d;
		target_end_index *= new_size;
	}
	int block_length = block_end_index; // alias, same value but different concept
	//printf("resize: %i %i\n", size, new_size);
	//int blocksize = size/new_size
	//int blocklength = 1;
	//for(int i = 0; i < dimension; i++) {
	//	blocklength *= block_size_1d;
	bool done = false;
	while(!done) {

		double value = 0;
		bool done_block = false;
		block_index = 0;
		// sum up all values from the subblock of the source

		int source_index = 0; //target_index * block_length;
		int target_index_subspace = target_index;
		int scale = block_size_1d;
		for(int i = 0; i < dimension; i++) {
			source_index += (target_index_subspace % new_size) * scale;
			scale *= new_size*block_size_1d;
			target_index_subspace /= new_size;
		}
		while(!done_block) {
			/* now convert the 1d block_index to of offset for the source */
			int block_index_subspace = block_index;
			int offset = 0;
			scale = 1;
			//printf("\tstart offset = %i\n", offset);
			for(int i = 0; i < dimension; i++) {
				offset += (block_index_subspace % block_size_1d) * scale;
				//printf("\tsource_index = %i dim=%i scale=%i block_index_subspace=%i %i\n", offset, i, scale, block_index_subspace, (block_index_subspace % block_size_1d));
				scale *= size;
				block_index_subspace /= block_size_1d;
			}
			//printf(" source_index = %i\n", source_index+offset);
			value += source[source_index+offset];
			block_index++;
			if(block_index == block_end_index)
				done_block = true;
		}
		//printf("target_index = %i\n", target_index);
		target[target_index] = value;
		target_index++;
		if(target_index == target_end_index)
			done = true;
	}
}

static PyObject* resize_(PyObject* self, PyObject *args) {
	PyObject* result = NULL;
	PyObject *array = NULL;
	int new_size;
	if(PyArg_ParseTuple(args, "Oi", &array, &new_size)) {
		double *array_ptr, *new_array_ptr;
		int sizes[3];
		npy_intp new_sizes[3];
		int strides[3];
		int dimension = 0;
		try {
			object_to_numpyNd_nocopy(array_ptr, array, 3, dimension, sizes, strides, NPY_DOUBLE);
			for(int i = 0; i < dimension; i++) {
				new_sizes[i] = new_size;
			}
			int size1 = sizes[0];
			for(int i = 1; i < dimension; i++) {
				if(sizes[i] != size1)
					throw std::runtime_error("array sizes aren't equal in all dimensions");
			}
			// check the array is 'normally' shaped, continuous, not transposed etc
			int stride = 8;
			for(int i = 0; i < dimension; i++) {
				//printf("strides[dimension-1-i] = %i\n", strides[dimension-1-i]);
				if(strides[dimension-1-i] != stride)
					throw std::runtime_error("array strides don't match that of a continuous array");
				stride *= size1;
			}
			PyObject* new_array = PyArray_SimpleNew(dimension, new_sizes, NPY_DOUBLE);
			new_array_ptr = (double*)PyArray_DATA(new_array);
			
			//Py_BEGIN_ALLOW_THREADS
			resize(array_ptr, size1, dimension, new_array_ptr, new_size);
			//Py_END_ALLOW_THREADS
			//Py_INCREF(Py_None);
			result = new_array;
		} catch(std::runtime_error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
			//PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}
	
	return result;
}


static PyMethodDef pygavi_functions[] = {
        {"range_check", (PyCFunction)range_check_, METH_VARARGS, ""},
        {"find_nan_min_max", (PyCFunction)find_nan_min_max_, METH_VARARGS, ""},
        {"histogram1d", (PyCFunction)histogram1d_, METH_VARARGS, ""},
        {"histogram2d", (PyCFunction)histogram2d_, METH_VARARGS, ""},
        {"histogram3d", (PyCFunction)histogram3d_, METH_VARARGS, ""},
        {"project", (PyCFunction)project_, METH_VARARGS, ""},
        {"pnpoly", (PyCFunction)pnpoly_, METH_VARARGS, ""},
        {"soneira_peebles", (PyCFunction)soneira_peebles_, METH_VARARGS, ""},
        {"shuffled_sequence", (PyCFunction)shuffled_sequence_, METH_VARARGS, ""},
        {"resize", (PyCFunction)resize_, METH_VARARGS, ""},
    { NULL, NULL, 0 }
};


PyMODINIT_FUNC
initgavifast(void)
{
	import_array();

	Py_InitModule("gavifast", pygavi_functions);
}







