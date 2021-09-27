#include <Python.h>
#include <math.h>
#include <stdexcept>
#include <cstdio>
//#include <omp.h>

// from http://stackoverflow.com/questions/126279/c99-stdint-h-header-and-ms-visual-studio
#ifdef _MSC_VER
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#define __attribute__(x)
#define __builtin_bswap64 _byteswap_uint64

// from http://stackoverflow.com/questions/2538339/infinity-in-msvc
#include <limits>
#define INFINITY std::numeric_limits<float>::infinity()

// for isfinite
#define isfinite _finite
#include <float.h>

#define __restrict__ __restrict
#else
#include <stdint.h>
#endif



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
			throw Error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 1)
			throw Error("array is not 1d");
		long long size = PyArray_DIMS(obj)[0];
		if((count >= 0) && (size != count))
			throw Error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw Error("is not of proper type");
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
void object_to_numpy1d_nocopy_endian(T* &ptr, PyObject* obj, long long  &count, bool &native, int& stride=stride_default, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw Error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 1)
			throw Error("array is not 1d");
		long long size = PyArray_DIMS(obj)[0];
		if((count >= 0) && (size != count))
			throw Error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw Error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		if(stride == -1) {
			stride = strides[0];
		} else {
			if(strides[0] != stride*PyArray_ITEMSIZE(obj)) {
				throw Error("stride is not equal to %d", stride);
			}
		}
		native = PyArray_ISNOTSWAPPED(obj);
		ptr = (T*)PyArray_DATA(obj);
		count = size;
}

template<typename T>
void object_to_numpy2d_nocopy(T* &ptr, PyObject* obj, int &count_x, int &count_y, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw Error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 2)
			throw Error("array is not 2d");
		int size_x = PyArray_DIMS(obj)[1];
		if((count_x >= 0) && (size_x != count_x))
			throw Error("arrays not of equal size");
		int size_y = PyArray_DIMS(obj)[0];
		if((count_y >= 0) && (size_y != count_y))
			throw Error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw Error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		//printf("strides: %d %d (%d %d)\n", strides[0],strides[1], size_x, size_y);
		if(strides[1] != PyArray_ITEMSIZE(obj)) {
			throw Error("stride[0] is not 1");
		}
		if(strides[0] != PyArray_ITEMSIZE(obj)*size_x) {
			throw Error("stride[1] is not 1");
		}

		ptr = (T*)PyArray_DATA(obj);
		count_x = size_x;
		count_y = size_y;
}

template<typename T>
void object_to_numpy3d_nocopy(T* &ptr, PyObject* obj, int &count_x, int &count_y, int &count_z, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw Error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 3)
			throw Error("array is not 3d");
		int size_x = PyArray_DIMS(obj)[2];
		if((count_x >= 0) && (size_x != count_x))
			throw Error("arrays not of equal size");
		int size_y = PyArray_DIMS(obj)[1];
		if((count_y >= 0) && (size_y != count_y))
			throw Error("arrays not of equal size");
		int size_z = PyArray_DIMS(obj)[0];
		if((count_z >= 0) && (size_z != count_z))
			throw Error("arrays not of equal size");
		if(PyArray_TYPE(obj) != type)
			throw Error("is not of proper type");
		npy_intp* strides =  PyArray_STRIDES(obj);
		//printf("strides: %d %d %d(%d %d %d)\n", strides[0], strides[1], strides[2], size_x, size_y, size_z);
		if(strides[2] != PyArray_ITEMSIZE(obj)) {
			throw Error("stride[0] is not 1");
		}
		if(strides[1] != PyArray_ITEMSIZE(obj)*size_x) {
			throw Error("stride[1] is not 1");
		}
		if(strides[0] != PyArray_ITEMSIZE(obj)*size_y*size_x) {
			throw Error("stride[2] is not 1");
		}

		ptr = (T*)PyArray_DATA(obj);
		count_x = size_x;
		count_y = size_y;
		count_z = size_z;
}

template<typename T>
void object_to_numpyNd_nocopy(T* &ptr, PyObject* obj, int max_dimension, int& dimension, int* sizes, long long int* strides, int type=NPY_DOUBLE) {
		if(obj == NULL)
			throw Error("cannot convert to numpy array");
		//printf("dim = %i maxdim = %i %i\n", dimension, max_dimension,  (int)PyArray_NDIM(obj));
		dimension = (int)PyArray_NDIM(obj);
		if(dimension > max_dimension) {
			printf("dim = %i maxdim = %i\n", dimension, max_dimension);
			throw Error("array dimension is bigger than allowed");
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
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		} catch(...) {
			PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}
	return result;
}


double double_to_native1(double value_non_native) {
	unsigned char* bytes = (unsigned char*)&value_non_native;
	double result;
	unsigned char* result_bytes = (unsigned char*)&result;
	for(int i = 0; i < 8; i++)
		result_bytes[7-i] = bytes[i];
	return result;
}

inline double double_to_native(double value)
{
	uint64_t* val = (uint64_t*)&value;
	double result_value;
    uint64_t* result = (uint64_t*)&result_value;
    /**val = (((*val) << 8) & 0xFF00FF00FF00FF00ULL )  | (((*val) >> 8) & 0x00FF00FF00FF00FFULL );
    *val = (((*val) << 16) & 0xFFFF0000FFFF0000ULL ) | (((*val) >> 16) & 0x0000FFFF0000FFFFULL );
    *result = ((*val) << 32) | (((*val) >> 32) & 0xFFFFFFFFULL);*/
	*result = __builtin_bswap64(*val);
	return result_value;
}
// no-op for template
inline double double_to_double(double value) {
    return value;
}

#define custom_isfinite(value) ((value-value)==0)
#define custom_isnan(value) (!(value==value))

void find_nan_min_max(const double* const block_ptr, const long long length, bool native, double &min_, double &max_) {
	double min = INFINITY, max = -INFINITY;// not using the reference but a local var seems easier for the compiler to optimize
	if(native) {
		for(long long i = 0; i < length; i++) {
	    	const double value = block_ptr[i];
	    	if(custom_isfinite(value)) {
                min = value < min ? value : min;
                max = value > max ? value : max;
            }
		}
	} else {
		for(long long i = 0; i < length; i++) {
			const double value = double_to_native(block_ptr[i]);
	    	if(custom_isfinite(value)) {
                min = value < min ? value : min;
                max = value > max ? value : max;
            }
		}
	}
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
		bool native = true;
		try {
			object_to_numpy1d_nocopy_endian(block_ptr, block, length, native);
			Py_BEGIN_ALLOW_THREADS
			find_nan_min_max(block_ptr, length, native, min, max);
			Py_END_ALLOW_THREADS
			result = Py_BuildValue("dd", min, max);
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

void nansum(const double* const __restrict__ block_ptr, const long long length, bool native, double &sum_) {
	double sum = sum_;

	if(native) {
		for(long long i = 0; i < length; i++) {
			double value = block_ptr[i];
			sum += value != value ? 0 : value;
		}
	} else {
		for(long long i = 0; i < length; i++) {
			double value = double_to_native(block_ptr[i]);
			sum += value != value ? 0 : value;
		}
	}
	sum_ = sum;
}


PyObject* nansum_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject* block;
	if(PyArg_ParseTuple(args, "O", &block)) {

		long long length = -1;
		double *block_ptr = NULL;
		double sum=0.;
		bool native = true;
		try {
			object_to_numpy1d_nocopy_endian(block_ptr, block, length, native);
			Py_BEGIN_ALLOW_THREADS
			nansum(block_ptr, length, native, sum);
			Py_END_ALLOW_THREADS
			result = Py_BuildValue("d", sum);
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

void sum(const double* const __restrict__ block_ptr, const long long length, bool native, double &sum_) {
	double sum = sum_;

	if(native) {
	    /*
	    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
        ymm0 = __builtin_ia32_loadupd256(block_ptr+4*0);
        ymm1 = __builtin_ia32_loadupd256(block_ptr+4*1);
        ymm2 = __builtin_ia32_loadupd256(block_ptr+4*2);
        ymm3 = __builtin_ia32_loadupd256(block_ptr+4*3);
        double normal[4];
		for(long long i = 4; i < length; i+=4*4) {
			//double value = block_ptr[i];
			//sum += block_ptr[i];
			ymm4 = __builtin_ia32_loadupd256(block_ptr+i+4*0);
			ymm5 = __builtin_ia32_loadupd256(block_ptr+i+4*1);
			ymm6 = __builtin_ia32_loadupd256(block_ptr+i+4*2);
			ymm7 = __builtin_ia32_loadupd256(block_ptr+i+4*3);

			ymm0 = __builtin_ia32_addpd256(ymm0, ymm4);
			ymm1 = __builtin_ia32_addpd256(ymm1, ymm5);
			ymm2 = __builtin_ia32_addpd256(ymm2, ymm6);
			ymm3 = __builtin_ia32_addpd256(ymm3, ymm7);

		}
        ymm0 = __builtin_ia32_addpd256(ymm0, ymm1);
        ymm1 = __builtin_ia32_addpd256(ymm2, ymm3);

        ymm0 = __builtin_ia32_addpd256(ymm0, ymm1);

        __builtin_ia32_storeupd256(normal, ymm0);
        for (int k=0; k<4; k++) {
          sum += normal[k];
        }/*/
		for(long long i = 0; i < length; i++) {
			double value = block_ptr[i];
			sum += value;
		}
        /**/

	} else {
		for(long long i = 0; i < length; i++) {
			double value = double_to_native(block_ptr[i]);
			sum += value;
		}
	}
	sum_ = sum;
}


PyObject* sum_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject* block;
	if(PyArg_ParseTuple(args, "O", &block)) {

		long long length = -1;
		double *block_ptr = NULL;
		double sum__=0.;
		bool native = true;
		try {
			object_to_numpy1d_nocopy_endian(block_ptr, block, length, native);
			Py_BEGIN_ALLOW_THREADS
			sum(block_ptr, length, native, sum__);
			Py_END_ALLOW_THREADS
			result = Py_BuildValue("d", sum__);
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}




void histogram1d(const double* const __restrict__ block, const long long block_stride, const bool block_native, const double* const weights, const int weights_stride, bool weights_native, long long block_length, double* __restrict__ counts, const int counts_length, const double min, const double max) {
	//const double* __restrict__ block_ptr = block;


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


	const double scale = 1 / (max-min);;
	if(block_native && weights_native) {
		for(long long i = 0; i < block_length; i++) {
			const double value = block[i]; //block[i*block_stride];
			const double scaled = (value - min) * scale;
			if( (scaled >= 0) & (scaled < 1) ) {
				int index = (int)(scaled * counts_length);
				counts[index] +=  weights == NULL ? 1 : (isfinite(weights[i]) ? weights[i] : 0);
			}
		}
	} else {
		/*if(!block_native && weights == NULL) {
			for(long long i = 0; i < block_length; i++) {
				const double value = block_native ? block[i] : double_to_native(block[i]);
				if((value > min) & (value < max)) {
					const double scaled = (value - min) * scale;
					const long long index = (long long)(scaled);
					counts[index] += 1;
				}
			}
		} else*/ {
			for(long long i = 0; i < block_length; i++) {
				const double value = block_native ? block[i] : double_to_native(block[i]);
    			const double scaled = (value - min) * scale;
				//block_ptr++;
				//const double value = *block_ptr;
				//block_ptr += block_stride;
				//__builtin_prefetch(block_ptr, 1, 1); // read, and no temporal locality
				//__builtin_prefetch(block_ptr+block_stride*10, 1, 1); // read, and no temporal locality
				//if( (index >= 0) & (index < counts_length) )
				//if((value >= min) & (value < max)) {
    			if( (scaled >= 0) & (scaled < 1) ) {
					//const double scaled = (value - min) * scale;
					//const long long index = (long long)(scaled);
    				int index = (int)(scaled * counts_length);
					if(weights != NULL) {
    					double weight = weights_native ? weights[i] : double_to_native(weights[i]);
	    				counts[index] +=  isfinite(weight) ? weight : 0;
                    } else {
                        counts[index] += 1;
                    }
				}
			}
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
		bool block_native = true;
		bool weights_native = true;
		bool counts_native = true;

		try {
			object_to_numpy1d_nocopy_endian(block_ptr, block, block_length, block_native, block_stride);
			object_to_numpy1d_nocopy_endian(counts_ptr, counts, counts_length, counts_native, weights_stride);
			if(weights != Py_None) {
				object_to_numpy1d_nocopy_endian(weights_ptr, weights, block_length, weights_native);
			}
			if(!counts_native)
				throw Error("counts is not in native byteorder");
			Py_BEGIN_ALLOW_THREADS
			histogram1d(block_ptr, block_stride, block_native, weights_ptr, weights_stride, weights_native, block_length, counts_ptr, counts_length, min, max);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}



void histogram2d(const double* const __restrict__ blockx, const double* const __restrict__ blocky, const double* const weights, const long long block_length, bool blockx_native, bool blocky_native, bool weights_native, double* const __restrict__ counts, const int counts_length_x, const int counts_length_y, const double xmin, const double xmax, const double ymin, const double ymax, const long long offset_x, const long long offset_y) {
	long long i_x = offset_x;
	long long i_y = offset_y;
	const double scale_x = 1./ (xmax-xmin);
	const double scale_y = 1./ (ymax-ymin);
	// do nothing if any of the bounds are inf
	if(!(isfinite(xmin) && isfinite(xmax) && isfinite(ymin) && isfinite(ymax)))
		return;
	if(blockx_native && blocky_native & weights_native) {
		if((weights == NULL) & (offset_x == 0) & (offset_y == 0)) { // default: fasted algo
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx[i];
				double value_y = blocky[i];
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += 1;
				}
			}
		} else {
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx[i];
				double value_y = blocky[i];
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += weights == NULL ? 1 : (isfinite(weights[i]) ? weights[i] : 0);
				}
				i_x = i_x >= block_length-1 ? 0 : i_x+1;
				i_y = i_y >= block_length-1 ? 0 : i_y+1;
			}
		}
	} else {
		if((weights == NULL) & (offset_x == 0) & (offset_y == 0)) { // default: fasted algo
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
				double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += 1;
				}
			}
		} else {
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
				double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					if(weights != NULL) {
    					double weight = weights_native ? weights[i] : double_to_native(weights[i]);
	    				counts[index_x + counts_length_x*index_y] += isfinite(weight) ? weight : 0;
                    } else {
                        counts[index_x + counts_length_x*index_y] += 1;
                    }
				}
				i_x = i_x >= block_length-1 ? 0 : i_x+1;
				i_y = i_y >= block_length-1 ? 0 : i_y+1;
			}
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
		bool blockx_native = true;
		bool blocky_native = true;
		bool weights_native = true;

		try {
			object_to_numpy1d_nocopy_endian(blockx_ptr, blockx, block_length, blockx_native);
			object_to_numpy1d_nocopy_endian(blocky_ptr, blocky, block_length, blocky_native);
			object_to_numpy2d_nocopy(counts_ptr, counts, counts_length_x, counts_length_y);
			if(weights != Py_None) {
				object_to_numpy1d_nocopy_endian(weights_ptr, weights, block_length, weights_native);
			}
			/*if(!counts_native)
				throw Error("counts is not in native byteorder");*/
			Py_BEGIN_ALLOW_THREADS
			histogram2d(blockx_ptr, blocky_ptr, weights_ptr, block_length, blockx_native, blocky_native, weights_native, counts_ptr, counts_length_x, counts_length_y, xmin, xmax, ymin, ymax, offset_x, offset_y);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}


template<typename Tout>
void histogram2d_f4(const float* const __restrict__ blockx, const float* const __restrict__ blocky, const float* const weights, const long long block_length, bool blockx_native, bool blocky_native, bool weights_native, Tout* const __restrict__ counts, const int counts_length_x, const int counts_length_y, const double xmin_, const double xmax_, const double ymin_, const double ymax_, const long long offset_x, const long long offset_y) {
	long long i_x = offset_x;
	long long i_y = offset_y;
	const float xmin = xmin_;
	const float xmax = xmax_;
	const float ymin = ymin_;
	const float ymax = ymax_;
	const float scale_x = 1./ (xmax-xmin);
	const float scale_y = 1./ (ymax-ymin);
	// do nothing if any of the bounds are inf
	if(!(isfinite(xmin) && isfinite(xmax) && isfinite(ymin) && isfinite(ymax)))
		return;
	//if(blockx_native && blocky_native & weights_native) {
		//printf(".");
		if((weights == NULL) & (offset_x == 0) & (offset_y == 0)) { // default: fasted algo
			for(long long i = 0; i < block_length; i++) {
				float value_x = blockx[i];
				float value_y = blocky[i];
				float scaled_x = (value_x - xmin) * scale_x;
				float scaled_y = (value_y - ymin) * scale_y;

				{//if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += 1;
				}
			}
		}/* else {
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx[i];
				double value_y = blocky[i];
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += weights == NULL ? 1 : (isfinite(weights[i]) ? weights[i] : 0);
				}
				i_x = i_x >= block_length-1 ? 0 : i_x+1;
				i_y = i_y >= block_length-1 ? 0 : i_y+1;
			}
		}
	} else {
		if((weights == NULL) & (offset_x == 0) & (offset_y == 0)) { // default: fasted algo
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
				double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					counts[index_x + counts_length_x*index_y] += 1;
				}
			}
		} else {
			for(long long i = 0; i < block_length; i++) {
				double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
				double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
				double scaled_x = (value_x - xmin) * scale_x;
				double scaled_y = (value_y - ymin) * scale_y;

				if( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) {
					int index_x = (int)(scaled_x * counts_length_x);
					int index_y = (int)(scaled_y * counts_length_y);
					if(weights != NULL) {
    					double weight = weights_native ? weights[i] : double_to_native(weights[i]);
	    				counts[index_x + counts_length_x*index_y] += isfinite(weight) ? weight : 0;
                    } else {
                        counts[index_x + counts_length_x*index_y] += 1;
                    }
				}
				i_x = i_x >= block_length-1 ? 0 : i_x+1;
				i_y = i_y >= block_length-1 ? 0 : i_y+1;
			}
		}
	}
	/**/
}

PyObject* histogram2d_f4_(PyObject* self, PyObject* args) {
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
		float *blockx_ptr = NULL;
		float *blocky_ptr = NULL;
		float *weights_ptr = NULL;
		long long int *counts_ptr = NULL;
		bool blockx_native = true;
		bool blocky_native = true;
		bool weights_native = true;

		try {
			object_to_numpy1d_nocopy_endian(blockx_ptr, blockx, block_length, blockx_native, stride_default, NPY_FLOAT);
			object_to_numpy1d_nocopy_endian(blocky_ptr, blocky, block_length, blocky_native, stride_default, NPY_FLOAT);
			object_to_numpy2d_nocopy(counts_ptr, counts, counts_length_x, counts_length_y, NPY_INT64);
			if(weights != Py_None) {
				object_to_numpy1d_nocopy_endian(weights_ptr, weights, block_length, weights_native, stride_default, NPY_FLOAT);
			}
			/*if(!counts_native)
				throw Error("counts is not in native byteorder");*/
			Py_BEGIN_ALLOW_THREADS
			histogram2d_f4<long long int>(blockx_ptr, blocky_ptr, weights_ptr, block_length, blockx_native, blocky_native, weights_native, counts_ptr, counts_length_x, counts_length_y, xmin, xmax, ymin, ymax, offset_x, offset_y);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, bool blockx_native, bool blocky_native, bool blockz_native, bool weights_native, double* counts, const int counts_length_x, const int counts_length_y, const int counts_length_z, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, long long const offset_x, long long const offset_y, long long const offset_z) {
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
	//long long i_x = offset_x;
	//long long i_y = offset_y;
	//long long i_z = offset_z;
	const double scale_x = counts_length_x/ (xmax-xmin);
	const double scale_y = counts_length_y/ (ymax-ymin);
	const double scale_z = counts_length_z/ (zmax-zmin);
	if((weights == NULL) & (offset_x == 0) & (offset_y == 0) & (offset_z == 0)) { // default: fasted algo
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
			double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
			double value_z = blockz_native ? blockz[i] : double_to_native(blockz[i]);

			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) & (value_z >= zmin) & (value_z < zmax)) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				int index_z = (int)((value_z - zmin) * scale_z);
				counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += 1;
			}
		}
	} else {
		for(long long i = 0; i < block_length; i++) {
			double value_x = blockx_native ? blockx[i] : double_to_native(blockx[i]);
			double value_y = blocky_native ? blocky[i] : double_to_native(blocky[i]);
			double value_z = blockz_native ? blockz[i] : double_to_native(blockz[i]);

			if( (value_x >= xmin) & (value_x < xmax) &  (value_y >= ymin) & (value_y < ymax) & (value_z >= zmin) & (value_z < zmax)) {
				int index_x = (int)((value_x - xmin) * scale_x);
				int index_y = (int)((value_y - ymin) * scale_y);
				int index_z = (int)((value_z - zmin) * scale_z);
				double weight = weights_native ? weights[i] : double_to_native(weights[i]);
				counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += weight;
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
		bool blockx_native = true;
		bool blocky_native = true;
		bool blockz_native = true;
		bool weights_native = true;
		try {
			object_to_numpy1d_nocopy_endian(blockx_ptr, blockx, block_length, blockx_native);
			object_to_numpy1d_nocopy_endian(blocky_ptr, blocky, block_length, blocky_native);
			object_to_numpy1d_nocopy_endian(blockz_ptr, blockz, block_length, blockz_native);
			object_to_numpy3d_nocopy(counts_ptr, counts, counts_length_x, counts_length_y, counts_length_z);
			if(weights != Py_None) {
				object_to_numpy1d_nocopy_endian(weights_ptr, weights, block_length, weights_native);
			}
			Py_BEGIN_ALLOW_THREADS
			histogram3d(blockx_ptr, blocky_ptr, blockz_ptr, weights_ptr, block_length, blockx_native, blocky_native, blockz_native, weights_native, counts_ptr, counts_length_x, counts_length_y, counts_length_z, xmin, xmax, ymin, ymax, zmin, zmax, offset_x, offset_y, offset_z);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

const int MAX_DIMENSIONS = 50;
const int MAX_WEIGHTS = 2000;

// __restrict__
void histogramNd(const double* const blocks[], const double* const weights, long long block_length, int dimensions, double* counts, long long count_strides[], int count_sizes[], double minima[], double maxima[]) {
//void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, double* counts, const int counts_length_x, const int counts_length_y, const int counts_length_z, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, long long const offset_x, long long const offset_y, long long const offset_z){
	double scales[MAX_DIMENSIONS];
	for(int d = 0; d < dimensions; d++) {
		scales[d] = count_sizes[d] / (maxima[d] - minima[d]);
	}
	if((weights == NULL)) { // default: fasted algo
		for(long long i = 0; i < block_length; i++) {
			long long index = 0;
			bool inside = true;
			for(int d = 0; d < dimensions; d++) {
				double value = blocks[dimensions-1-d][i];
				if( (value >= minima[dimensions-1-d]) & (value < maxima[dimensions-1-d]) ) {
					int sub_index = (int)((value - minima[dimensions-1-d]) * scales[dimensions-1-d]);
					index += count_strides[d] * sub_index;
				} else {
					inside = false;
					break;
				}
			}
			if(inside)
				counts[index] += 1;
		}
	} else {
		//counts[index_x + counts_length_x*index_y + counts_length_x*counts_length_y*index_z] += weights == NULL ? 1 : weights[i];
	}
}



PyObject* histogramNd_(PyObject* self, PyObject* args) {
	//object block, object weights, object counts, double min, double max) {
	PyObject* result = NULL;
	PyObject *blocklist, *weights, *counts_object, *minimalist, *maximalist;
	//double xmin, xmax, ymin, ymax, zmin, zmax;
	double minima[MAX_DIMENSIONS];
	double maxima[MAX_DIMENSIONS];
	if(PyArg_ParseTuple(args, "OOOOO", &blocklist, &weights, &counts_object, &minimalist, &maximalist)) {
		long long block_length = -1;
		int count_sizes[MAX_DIMENSIONS];
		long long count_strides[MAX_DIMENSIONS];
		int dimensions = -1;
		double *block_ptrs[MAX_DIMENSIONS];

		double *weights_ptr = NULL;
		double *counts_ptr = NULL;
		try {
			if(!PyList_Check(blocklist))
				throw Error("blocks is not a list of blocks");
			dimensions = PyList_Size(blocklist);

			if(!PyList_Check(minimalist))
				throw Error("minima is not a list of blocks");
			if(PyList_Size(minimalist) != dimensions)
				throw Error("minima is of length %ld, expected %d", PyList_Size(minimalist), dimensions);

			if(!PyList_Check(maximalist))
				throw Error("maxima is not a list of blocks");
			if(PyList_Size(maximalist) != dimensions)
				throw Error("maxima is of length %ld, expected %d", PyList_Size(maximalist), dimensions);

			for(int d = 0; d < dimensions; d++) {
				object_to_numpy1d_nocopy(block_ptrs[d], PyList_GetItem(blocklist, d), block_length);
				PyObject *min = PyList_GetItem(minimalist, d);
				PyObject *max = PyList_GetItem(maximalist, d);
				if(!PyFloat_Check(min))
					throw Error("element %d of minima is not of type float", d);
				if(!PyFloat_Check(max))
					throw Error("element %d of maxima is not of type float", d);
				minima[d] =  PyFloat_AsDouble(min);
				maxima[d] =  PyFloat_AsDouble(max);
				//printf("min/max[%d] = %f/%f\n", d, minima[d], maxima[d]);
			}
			if(weights != Py_None) {
				object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
			}
			object_to_numpyNd_nocopy(counts_ptr, counts_object, MAX_DIMENSIONS, dimensions, &count_sizes[0], &count_strides[0]);
			for(int d = 0; d < dimensions; d++) {
				count_strides[d] /= 8; // convert from byte stride to element stride
			}
			if(weights != Py_None) {
				object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
			}
			Py_BEGIN_ALLOW_THREADS
			histogramNd(block_ptrs, weights_ptr, block_length, dimensions, counts_ptr, count_strides, count_sizes, minima, maxima);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

template<typename T>
struct functor_double_to_double {
    inline T operator()(T value) {
        return value;
    }
};

struct functor_double_to_native {
    inline double operator()(double value) {
        return double_to_native(value);
    }
};

template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_add1 {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        outputs[0] += 1;
    }
};
template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_count {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        double value = endian(inputs[0][input_offsets]);
        outputs[0] += (custom_isnan(value) ? 0 : 1);
    }
};

template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_add_weight_moment_01 {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        double value = endian(inputs[0][input_offsets]);
        if(!custom_isnan(value))
        {
            outputs[0] += 1;
            outputs[1] += value;
        }
    }
};

template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_min_max {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        double value = endian(inputs[0][input_offsets]);
        if(value < outputs[0]) {
            outputs[0] = value;
        }
        if (value > outputs[1]) {
            outputs[1] = value;
        }
    }
};

template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_add_weight_moment_012 {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        double value = endian(inputs[0][input_offsets]);
        if(!custom_isnan(value)) {
            outputs[0] += 1;
            outputs[1] += value;
            outputs[2] += value*value;
        }
    }
};

template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_cov {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        // 
        //int N_lower_triangle = ((inputs_count+1)*inputs_count)/2;
        // we keep counts, sum and lower_triangle counts and sums a in 1 1d array
        // N x N matrix
        int N = inputs_count;
        int Nsquare = N * N;
        for(int col = 0; col < N; col++) {
	        double value = endian(inputs[col][input_offsets]);
	        if(!custom_isnan(value)) {
	            outputs[col] += 1;
	            outputs[col+N] += value;
	            // diagonal elements
	            outputs[N*2 + col + col * N] += 1;
	            outputs[N*2 + Nsquare + col + col * N] += value*value;
		        double value_x = value;
		        // off diagonal
		        for(int row = col+1; row < N; row++) {
			        double value_y = endian(inputs[row][input_offsets]);
			        if(!custom_isnan(value_y)) {
			    	    int index_1d_a = row + col * N;
			    	    int index_1d_b = col + row * N;
			            outputs[N*2 + index_1d_a] += 1;
			            outputs[N*2 + index_1d_b] = outputs[N*2 + index_1d_a];
			            outputs[N*2 + Nsquare + index_1d_a] += value_x * value_y;
			            outputs[N*2 + Nsquare + index_1d_b] = outputs[N*2 + Nsquare + index_1d_a];
			        }
			    }
	        }
        }
    }
};


template<typename T=double, typename Tout=double, typename ENDIAN=functor_double_to_double<T> >
struct op_first {
    void operator()(Tout* const __restrict__ outputs, const T* const __restrict__ inputs[], const long long int input_offsets, const int inputs_count) {
        ENDIAN endian;
        double value = endian(inputs[0][input_offsets]);
		double order_value = endian(inputs[1][input_offsets]);
        if(order_value < outputs[1]) {
			outputs[0] = value;
            outputs[1] = order_value;
        }
    }
};


template<typename T, typename OP, typename ENDIAN>
void statisticNd(
    const T* const __restrict__ blocks[],
    const T* const __restrict__ weights[],
    long long block_length,
    const int weights_count,
    const int dimensions,
    double* const __restrict__ counts,
    const long long * const __restrict__ count_strides,
    const int * const __restrict__ count_sizes,
    const T* const __restrict__ minima,
    const T* const __restrict__ maxima,
    int use_edges
    ) {
//void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, double* counts, const int counts_length_x, const int counts_length_y, const int counts_length_z, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, long long const offset_x, long long const offset_y, long long const offset_z){
    OP op;
    ENDIAN endian;

	T scales[MAX_DIMENSIONS];
    for(int d = 0; d < dimensions; d++) {
        scales[d] = 1 / (maxima[d] - minima[d]);
    }
	if(use_edges) {
	    // this is the unoptimized path (so no if(dimension == X)), since it won't be used as much and less
	    // burden to maintain
       for(long long i = 0; i < block_length; i++) {
            long long index = 0;
            for(int d = 0; d < dimensions; d++) {
                T value = endian(blocks[d][i]);
                double scaled = (value - minima[d]) * scales[d];
                if(scaled != scaled) { // nan check
                    index += count_strides[d] * 0; // nans are put at offset 0
                } else
                if(scaled < 0) { // smaller values are put at offset 1
                    index += count_strides[d] * 1;
                } else
                if(scaled >= 1) { // bigger values are put at offset -1 (last)
                    index += count_strides[d] * (count_sizes[d]-1);
                } else {
                    int sub_index = (int)(scaled * (count_sizes[d]-3)) + 2; // we start
                    index += count_strides[d] * sub_index;
                }
            }
            op(&counts[index], weights, i, weights_count);
        }
	} else {
        // the (gcc) compiler does better when we put the same statements in if's...
        if(dimensions == 0) { // optimization.. can we generalize it?
            for(long long i = 0; i < block_length; i++) {
            	op(&counts[0], weights, i, weights_count);
            }
        } else
        if(dimensions == 1) { // optimization.. can we generalize it?
            for(long long i = 0; i < block_length; i++) {
                long long index = 0;
                bool inside = true;
                for(int d = 0; d < 1; d++) {
                    T value = endian(blocks[d][i]);
                    double scaled = (value - minima[d]) * scales[d];
                    if( (scaled >= 0) & (scaled < 1) ) {
                        int sub_index = (int)(scaled * count_sizes[d]);
                        index += count_strides[d] * sub_index;
                    } else {
                        inside = false;
                        break;
                    }
                }
                if(inside)
		            op(&counts[index], weights, i, weights_count);
            }
        } else

        if(dimensions == 2) { // optimization.. can we generalize it?
            for(long long i = 0; i < block_length; i++) {
                long long index = 0;
                bool inside = true;
                for(int d = 0; d < dimensions; d++) {
                    T value = endian(blocks[d][i]);
                    T scaled = (value - minima[d]) * scales[d];
                    if( (scaled >= 0) & (scaled < 1) ) {
                        int sub_index = (int)(scaled * count_sizes[d]);
                        index += count_strides[d] * sub_index;
                    } else {
                        inside = false;
                        break;
                    }
                }
                if(inside)
          		  op(&counts[index], weights, i, weights_count);
            }
        } else {
            for(long long i = 0; i < block_length; i++) {
                long long index = 0;
                bool inside = true;
                for(int d = 0; d < dimensions; d++) {
                    T value = endian(blocks[d][i]);
                    double scaled = (value - minima[d]) * scales[d];
                    if( (scaled >= 0) & (scaled < 1) ) {
                        int sub_index = (int)(scaled * count_sizes[d]);
                        index += count_strides[d] * sub_index;
                    } else {
                        inside = false;
                        break;
                    }
                }
                if(inside)
           			op(&counts[index], weights, i, weights_count);
            }
        }
    }
}

#define ENUM2STR(k) #k
enum {
    OP_ADD1,
    OP_COUNT,
    OP_MIN_MAX,
    OP_ADD_WEIGHT_MOMENTS_01,
    OP_ADD_WEIGHT_MOMENTS_012,
    OP_COV,
    OP_FIRST
};// STATISTIC_OPS;

/*

static const char *statistic_op_names[] = {
    ENUM2STR(OP_ADD1),
    ENUM2STR(OP_COUNT),
    ENUM2STR(OP_MIN_MAX),
    ENUM2STR(OP_ADD_WEIGHT_MOMENTS_01),
    ENUM2STR(OP_ADD_WEIGHT_MOMENTS_012)
    ENUM2STR(OP_COV)
};
*/

template<typename T, typename endian>
void statisticNd_wrap_template_endian(
        const T* const blocks[],
        const T* const weights[],
        long long block_length,
        int weights_count,
        int dimensions,
        double* counts,
        long long count_strides[],
        int count_sizes[],
        T minima[],
        T maxima[],
        int op_code,
        int use_edges) {
        if(op_code == OP_ADD1) {
            statisticNd<T, op_add1<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_COUNT) {
            statisticNd<T, op_count<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_MIN_MAX) {
            statisticNd<T, op_min_max<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_ADD_WEIGHT_MOMENTS_01) {
            statisticNd<T, op_add_weight_moment_01<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_ADD_WEIGHT_MOMENTS_012) {
            statisticNd<T, op_add_weight_moment_012<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_COV) {
            statisticNd<T, op_cov<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else if(op_code == OP_FIRST) {
            statisticNd<T, op_first<T, double, endian>, endian >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, use_edges);
        } else {
            printf("unknown op code for statistic: %i", op_code);
        }

}

template<typename T>
void statisticNd_wrap_template(
        const T* const blocks[],
        const T* const weights[],
        long long block_length,
        int weights_count,
        int dimensions,
        double* counts,
        long long count_strides[],
        int count_sizes[],
        T minima[],
        T maxima[],
        bool native,
        int op_code,
        int use_edges) {
        if(native)
            statisticNd_wrap_template_endian<T, functor_double_to_double<T> >(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, op_code, use_edges);
        else
            statisticNd_wrap_template_endian<T, functor_double_to_native>(blocks, weights, block_length, weights_count, dimensions, counts, count_strides, count_sizes, minima, maxima, op_code, use_edges);
}

template<typename T ,int NP_TYPE>
PyObject* statisticNd_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject *blocklist, *weightslist, *counts_object, *minimalist, *maximalist;
	//double xmin, xmax, ymin, ymax, zmin, zmax;
	T minima[MAX_DIMENSIONS];
	T maxima[MAX_DIMENSIONS];
	int op_code;
	int use_edges = 0;
	if(PyArg_ParseTuple(args, "OOOOOi|i", &blocklist, &weightslist, &counts_object, &minimalist, &maximalist, &op_code, &use_edges)) {
		long long block_length = -1;
		int count_sizes[MAX_DIMENSIONS];
		long long count_strides[MAX_DIMENSIONS];
		int dimensions = -1;
		int dimensions_grid = -1;
		int weights_count = -1;
		T *block_ptrs[MAX_DIMENSIONS];
		bool native;

		T *weights_ptrs[MAX_WEIGHTS];
		double *counts_ptr = NULL;
		try {
			if(!PyList_Check(blocklist))
				throw Error("blocks is not a list of blocks");
			dimensions = PyList_Size(blocklist);
			dimensions_grid = dimensions + 1; // one dimension higher for multiple output values

			weights_count = PyList_Size(weightslist);

			if(!PyList_Check(minimalist))
				throw Error("minima is not a list of blocks");
			if(PyList_Size(minimalist) != dimensions)
				throw Error("minima is of length %ld, expected %d", PyList_Size(minimalist), dimensions);

			if(!PyList_Check(maximalist))
				throw Error("maxima is not a list of blocks");
			if(PyList_Size(maximalist) != dimensions)
				throw Error("maxima is of length %ld, expected %d", PyList_Size(maximalist), dimensions);

			for(int d = 0; d < dimensions; d++) {
				//object_to_numpy1d_nocopy       (block_ptrs[d], PyList_GetItem(blocklist, d), block_length);
				bool native_current;
    			object_to_numpy1d_nocopy_endian(block_ptrs[d], PyList_GetItem(blocklist, d), block_length, native_current, stride_default, NP_TYPE);
				PyObject *min = PyList_GetItem(minimalist, d);
				PyObject *max = PyList_GetItem(maximalist, d);
				if(!PyFloat_Check(min))
					throw Error("element %d of minima is not of type float", d);
				if(!PyFloat_Check(max))
					throw Error("element %d of maxima is not of type float", d);

				if(d == 0) {
					native = native_current;
				}
				else if(native != native_current) {
					throw Error("mixed native and non-native arrays not supported");
				}
				minima[d] =  (T)PyFloat_AsDouble(min);
				maxima[d] =  (T)PyFloat_AsDouble(max);
				//printf("min/max[%d] = %f/%f\n", d, minima[d], maxima[d]);
			}
			/*if(weights != Py_None) {
    			bool weight_native = true;
				object_to_numpy1d_nocopy_endian(weights_ptr, weights, block_length, weight_native);
				if(dimensions == 0) {
				    block_native[0] = weight_native; // ugly, but works to support 0 dim
				}
				if(weight_native != block_native[0])
					throw Error("mixed native and non-native arrays not supported");
			}*/
			for(int d = 0; d < weights_count; d++) {
				bool native_current;
    			object_to_numpy1d_nocopy_endian(weights_ptrs[d], PyList_GetItem(weightslist, d), block_length, native_current, stride_default, NP_TYPE);
				if((d == 0) & (dimensions == 0)) {
					native = native_current;
				}
				else if(native != native_current) {
					throw Error("mixed native and non-native arrays not supported");
				}
			}

			object_to_numpyNd_nocopy(counts_ptr, counts_object, MAX_DIMENSIONS, dimensions_grid, &count_sizes[0], &count_strides[0]);
			for(int d = 0; d < dimensions_grid; d++) {
				count_strides[d] /= 8; // convert from byte stride to element stride
			}
			if(count_strides[dimensions_grid-1] != 1) {
			    throw Error("last dimension in grid should have stride of 1, not %lli", count_strides[dimensions_grid-1]);
			}
			/*if(weights != Py_None) {
				object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
			}*/
			Py_BEGIN_ALLOW_THREADS
			statisticNd_wrap_template(block_ptrs, weights_ptrs, block_length, weights_count, dimensions, counts_ptr, count_strides, count_sizes, minima, maxima, native, op_code, use_edges);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}

double interpolate_1d(double* const __restrict__ grid, const int grid_size, double value) {
    /**
     * Say we have grid of size 3, with 4 edges:
     *bin:    0   1   2
     *      |   |   |   |
     *edge: 0   1   2   3
     * we have explicitly the cumulative count on the right of the bins, so at 1,2,3, and an implicit 0 on the left
     *
     */
    int index_left = 0; // index of left edge, left holds it's value
    int index_right = 1;// index of right edge, right holds it's value
    int i = 1; // points to the value of the current right edge, so +1 gives the edge index
    double left_y  = 0;
    double right_y = grid[i-1];
    //double next_y = grid[i-1];
    //*
    // y is the value of the right edge
    // we will find the edge such that
    int current_edge = 0;
    double current_value = 0;
    double next_value = grid[0];


    while(1) {
        //printf("left) %i %f %f\n", current_edge, left_y, value);
        if(current_edge == grid_size) // at end
            break;
        if(next_value <= value) { // is next edge still smaller or equal?
            bool exact_match = next_value == value;
            current_edge += 1;
            current_value = left_y = grid[current_edge-1];
            next_value = grid[current_edge]; // TODO: is this ok... out of bounds check?
            index_left = current_edge;
            if(exact_match) { // for the left edge we stop at the first match
                //printf("left exact) %i %f %f\n", current_edge, left_y, value);
                break;
            }
        } else { // we found an index, such that the next edge has a larger value than 'value'
            break;
        }
    }

    right_y = left_y;
    index_right = current_edge;

    while(1) {
        //printf("right) %i %f %f\n", current_edge, right_y, value);
        if(current_edge == grid_size) // at end
            break;
        if(current_value <= value) { // the current satisfies this, so take the next
            if((current_value == value) && (next_value > value)) {
                break;
            }
            current_edge += 1;
            current_value = right_y = grid[current_edge-1];
            next_value = grid[current_edge]; // TODO: is this ok... out of bounds check?
            index_right = current_edge;
        } else {
            break;
        }
    }

    //printf("left/right) %f %f %f\n", left_y, right_y, value);
    double x1 = ((double)index_left / grid_size);
    double x2 = ((double)index_right / grid_size);
    //printf("%f - %f\n", x1, x2);
    //printf("%d - %d\n", index_left, index_right);
    double x;
    if(left_y == right_y)
        x = (x1 + x2) / 2;
    else if(index_left == index_right)
        x = x1;
    else
        x = (value - left_y) / (right_y - left_y) * (x2 - x1) + x1;
    //printf("x = %f\n", x);
    return x;
    //double sum = (index_left) + (index_right);
    //return (sum/2) / (grid_size); // TODO: interpolate instead of taking center
}

void grid_interpolate(const int dimensions,
    double* const __restrict__ grid, const long long * const __restrict__ grid_strides, const int * const __restrict__ grid_sizes,
    double* const __restrict__ output, const long long * const __restrict__ output_strides, const int * const __restrict__ output_sizes,
    double value
    ) {
    long long length_1d = 1;//grid_strides[0];
    for(int d = 0; d < dimensions; d++) {
        length_1d *= grid_sizes[d];
    }
    //length_1d *= grid_strides[0];
    /*for(int d = 0; d < dimensions+1; d++) {
        printf("strides[%d] = %d size[%d] = %d\n", d, grid_strides[d], d, grid_sizes[d]);
    }
    for(int d = 0; d < dimensions; d++) {
        printf("strides[%d] = %d\n", d, output_strides[d]);
    }*/
    //printf("length_1d = %d\n", length_1d);
    //printf("output = %p\n", output);
    for(long long int i = 0; i < length_1d; i++) {
        //printf("i = %d\n", i);
        double v = interpolate_1d(&grid[i*grid_strides[0]], grid_sizes[dimensions], value);
        //printf("v = %f\n", v);
        output[i] = v;
    }

}
//void histogram3d(const double* const blockx, const double* const blocky, const double* const blockz, const double* const weights, long long block_length, double* counts, const int counts_length_x, const int counts_length_y, const int counts_length_z, const double xmin, const double xmax, const double ymin, const double ymax, const double zmin, const double zmax, long long const offset_x, long long const offset_y, long long const offset_z){

PyObject* grid_interpolate_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject *grid_object, *output_object;
	//double xmin, xmax, ymin, ymax, zmin, zmax;
	//double minima[MAX_DIMENSIONS];
	//double maxima[MAX_DIMENSIONS];
	//int op_code;
	double value;
	if(PyArg_ParseTuple(args, "OOd",  &grid_object, &output_object, &value)) {

		int grid_sizes[MAX_DIMENSIONS];
		long long grid_strides[MAX_DIMENSIONS];
		int output_sizes[MAX_DIMENSIONS];
		long long output_strides[MAX_DIMENSIONS];

		double *grid_ptr = NULL;
		double *output_ptr = NULL;
		try {
			int dimensions_grid = -1;
			object_to_numpyNd_nocopy(grid_ptr, grid_object, MAX_DIMENSIONS, dimensions_grid, &grid_sizes[0], &grid_strides[0]);

			int dimensions_output = dimensions_grid-1;
			object_to_numpyNd_nocopy(output_ptr, output_object, MAX_DIMENSIONS, dimensions_output, &output_sizes[0], &output_strides[0]);

			for(int d = 0; d < dimensions_grid; d++) {
				grid_strides[d] /= 8; // convert from byte stride to element stride
			}
			for(int d = 0; d < dimensions_output; d++) {
				output_strides[d] /= 8; // convert from byte stride to element stride
			}
			if(!grid_ptr)
				throw Error("grid is null");
			if(!output_ptr)
				throw Error("output is null");
			//if(count_strides[dimensions_grid-1] != 1) {
			//    throw Error("last dimension in grid should have stride of 1, not %i", count_strides[dimensions_grid-1]);
			//}
			/*if(weights != Py_None) {
				object_to_numpy1d_nocopy(weights_ptr, weights, block_length);
			}*/
			Py_BEGIN_ALLOW_THREADS
	        grid_interpolate(dimensions_output, grid_ptr, grid_strides, grid_sizes, output_ptr, output_strides, output_sizes, value);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}


void grid_find_edges(
    const int dimensions,
    const int * const grid_sizes,
    double* const __restrict__ cumulative_grid,
    const long long * const __restrict__ cumulative_grid_strides,
    double* const __restrict__ values_grid,
    const long long * const __restrict__ values_strides,
    long long * const __restrict__ edges_grid,
    const long long * const __restrict__ edges_strides
) {
    long long length_1d = 1;//grid_strides[0];
    //printf("dimension: %d\n", dimensions);
    for(int d = 0; d < dimensions-1; d++) {
        //printf("length_1d = %d, mul by %d\n", length_1d, grid_sizes[d]);
        length_1d *= grid_sizes[d];
    }
    //length_1d *= grid_strides[0];
    /*for(int d = 0; d < dimensions+1; d++) {
        printf("strides[%d] = %d size[%d] = %d\n", d, grid_strides[d], d, grid_sizes[d]);
    }
    for(int d = 0; d < dimensions; d++) {
        printf("strides[%d] = %d\n", d, output_strides[d]);
    }*/
    //printf("length_1d = %d\n", length_1d);
    //printf("output = %p\n", output);
    //printf("grid sizes: %d %d\n", grid_sizes[0], grid_sizes[1]);
    //printf("edges_strides: %d %d\n", edges_strides[0], edges_strides[1]);
    //printf("values_strides: %d %d\n", values_strides[0], values_strides[1]);
    //printf("cumulative_grid_strides: %d %d %d\n", cumulative_grid_strides[0], cumulative_grid_strides[1], cumulative_grid_strides[2]);
    //printf("dimensions: %d\n", dimensions);


    for(long long int i = 0; i < length_1d; i++) {
        double* const __restrict__ cumulative_grid1d = &cumulative_grid[i*cumulative_grid_strides[dimensions-2]];
        double value = values_grid[i*values_strides[dimensions-2]];
        long long left = 0; //  'edge' index if left of the value we are looking for
        //printf("first/last value: %f / %f, looking for %f %dn", cumulative_grid1d[0], cumulative_grid1d[cumulative_grid_strides[dimensions-2]-1], value, left);
        while(cumulative_grid1d[left+1] < value && left < cumulative_grid_strides[dimensions-2]-1) {
            left+= 1;
        }
        long long right = left; // 'edge' index that is right of the value
        while (cumulative_grid1d[right] < value && right < cumulative_grid_strides[dimensions-2]-1) {
            right+= 1;
        }
        //while (cumulative_grid[right+1] == cumulative_grid[right]) { // make sure we get the right most bin
        //    right+= 1;
        //}
        //printf("setting: %d(%d) %d(%d)\n", i*edges_strides[0]+edges_strides[dimensions-1]*0, left, i*edges_strides[0]+edges_strides[dimensions-1]*1, right);
        edges_grid[i*edges_strides[dimensions-2]+edges_strides[dimensions-1]*0] = left;
        edges_grid[i*edges_strides[dimensions-2]+edges_strides[dimensions-1]*1] = right;
    }



}


PyObject* grid_find_edges_(PyObject* self, PyObject* args) {
	PyObject* result = NULL;
	PyObject *cumulative_grid, *values_grid, *edges_grid;
	//double xmin, xmax, ymin, ymax, zmin, zmax;
	//double minima[MAX_DIMENSIONS];
	//double maxima[MAX_DIMENSIONS];
	//int op_code;
	if(PyArg_ParseTuple(args, "OOO",  &cumulative_grid, &values_grid, &edges_grid)) {

		int cumulative_grid_sizes[MAX_DIMENSIONS];
		long long cumulative_grid_strides[MAX_DIMENSIONS];

		int values_sizes[MAX_DIMENSIONS];
		long long values_strides[MAX_DIMENSIONS];

		int edges_sizes[MAX_DIMENSIONS];
		long long edges_strides[MAX_DIMENSIONS];

		double *cumulative_grid_ptr = NULL;
		double *values_ptr = NULL;
		long long *edges_ptr = NULL;
		try {
			int dimensions_cumulative_grid= -1;
			object_to_numpyNd_nocopy(cumulative_grid_ptr, cumulative_grid, MAX_DIMENSIONS, dimensions_cumulative_grid, &cumulative_grid_sizes[0], &cumulative_grid_strides[0]);


			int dimensions_values = dimensions_cumulative_grid-1;
			object_to_numpyNd_nocopy(values_ptr, values_grid, MAX_DIMENSIONS, dimensions_values, &values_sizes[0], &values_strides[0]);

			int dimensions_edges = dimensions_cumulative_grid;
			object_to_numpyNd_nocopy(edges_ptr, edges_grid, MAX_DIMENSIONS, dimensions_edges, &edges_sizes[0], &edges_strides[0], NPY_INT64);

			for(int d = 0; d < dimensions_cumulative_grid; d++) {
				cumulative_grid_strides[d] /= 8; // convert from byte stride to element stride
				edges_strides[d] /= 8; // convert from byte stride to element stride
			}
			for(int d = 0; d < dimensions_values; d++) {
				values_strides[d] /= 8; // convert from byte stride to element stride
                if(cumulative_grid_sizes[d] != values_sizes[d])
                    throw Error("cumulative_grid and values_grid dont match shape in dimension: %d", d);
                if(cumulative_grid_sizes[d] != edges_sizes[d])
                    throw Error("cumulative_grid and edges_grid dont match shape in dimension: %d (%d vs %d)", d, cumulative_grid_sizes[d], edges_sizes[d]);
			}
			if(!cumulative_grid_ptr)
				throw Error("cumulative_grid is null");
			if(!values_ptr)
				throw Error("values is null");
			if(!edges_ptr)
				throw Error("edges is null");
			Py_BEGIN_ALLOW_THREADS
	        grid_find_edges(dimensions_cumulative_grid, cumulative_grid_sizes, cumulative_grid_ptr, cumulative_grid_strides, values_ptr, values_strides, edges_ptr, edges_strides);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
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
				throw Error("projection array should be of length 8");
			if(offset_length != 3)
				throw Error("center array should be of length 3");
			Py_BEGIN_ALLOW_THREADS
			project(cube_ptr, cube_length_x, cube_length_y, cube_length_z, surface_ptr, surface_length_x, surface_length_y, projection_ptr, offset_ptr);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
	return result;
}


void pnpoly(double *vertx, double *verty, int nvert, const double* const blockx, const double* const blocky, bool blockx_native, bool blocky_native, unsigned char* const mask, int length, double meanx, double meany, double radius) {
	double radius_squared = radius*radius;
	for(int k= 0; k < length; k++){
		double testx = blockx_native ? blockx[k] : double_to_native(blockx[k]);
		double testy = blocky_native ? blocky[k] : double_to_native(blocky[k]);
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
		bool blockx_native = true;
		bool blocky_native = true;
		try {
			object_to_numpy1d_nocopy(x_ptr, x, polygon_length);
			object_to_numpy1d_nocopy(y_ptr, y, polygon_length);
			object_to_numpy1d_nocopy_endian(blockx_ptr, blockx, length, blockx_native);
			object_to_numpy1d_nocopy_endian(blocky_ptr, blocky, length, blocky_native);
			object_to_numpy1d_nocopy(mask_ptr, mask, length, stride_default, NPY_BOOL);
			Py_BEGIN_ALLOW_THREADS
			pnpoly(x_ptr, y_ptr, polygon_length, blockx_ptr, blocky_ptr, blockx_native, blocky_native, mask_ptr, length, meanx, meany, radius);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
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
			if(length != pow((double)eta, (int)max_level))
				throw Error("length of coordinates != eta**max_level (%lld != %f)", length, pow((double)eta, (int)max_level));
			Py_BEGIN_ALLOW_THREADS
			soneira_peebles(coordinates_ptr, center, width, lambda, eta, 1, max_level);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
			//PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}

	return result;
}

#if  defined(__APPLE__) || defined(_MSC_VER)
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
void shuffled_sequence_(long long * array, long long length, bool native) {
#if  defined(__APPLE__) || defined(_MSC_VER)
	for(long long i=0; i < length; i++) {
		uint64_t r = rand();
		uint64_t j =  r * i / RAND_MAX;
		array[i] = array[j];
		array[j] = 	native ? i : __builtin_bswap64(i);
	}
#else
	auto rnd = std::mt19937_64(std::random_device{}());
	std::uniform_int_distribution<long long> dist(0, length-1);
	for(long long i=0; i < length; i++) {
		uint_fast64_t r = dist(rnd);
		uint_fast64_t j = r * i / (length-1);
		array[i] = array[j];
		//array[j] = i;
    array[j] = 	native ? i : __builtin_bswap64(i);
		if( ((i% 10000000) == 0) ){
			//printf("\r%lld out of %lld (%.2f%%)", i, length, (i*100./length));
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
      bool native;
			object_to_numpy1d_nocopy_endian(array_ptr, array, length, native, stride_default, NPY_INT64);
			Py_BEGIN_ALLOW_THREADS
			shuffled_sequence_(array_ptr, length, native);
			Py_END_ALLOW_THREADS
			Py_INCREF(Py_None);
			result = Py_None;
		} catch(Error e) {
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
		throw Error("target size should be smaller than source size");
	int block_size_1d = size/new_size;

	for(int i = 0; i < dimension; i++) {
		block_end_index *= block_size_1d;
		target_end_index *= new_size;
	}
	//int block_length = block_end_index; // alias, same value but different concept
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
		long long int strides[3];
		int dimension = 0;
		try {
			object_to_numpyNd_nocopy(array_ptr, array, 3, dimension, sizes, strides, NPY_DOUBLE);
			for(int i = 0; i < dimension; i++) {
				new_sizes[i] = new_size;
			}
			int size1 = sizes[0];
			for(int i = 1; i < dimension; i++) {
				if(sizes[i] != size1)
					throw Error("array sizes aren't equal in all dimensions");
			}
			// check the array is 'normally' shaped, continuous, not transposed etc
			int stride = 8;
			for(int i = 0; i < dimension; i++) {
				//printf("strides[dimension-1-i] = %i\n", strides[dimension-1-i]);
				if(strides[dimension-1-i] != stride)
					throw Error("array strides don't match that of a continuous array");
				stride *= size1;
			}
			PyObject* new_array = PyArray_SimpleNew(dimension, new_sizes, NPY_DOUBLE);
			new_array_ptr = (double*)PyArray_DATA(new_array);

			//Py_BEGIN_ALLOW_THREADS
			resize(array_ptr, size1, dimension, new_array_ptr, new_size);
			//Py_END_ALLOW_THREADS
			//Py_INCREF(Py_None);
			result = new_array;
		} catch(Error e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
			//PyErr_SetString(PyExc_RuntimeError, "unknown exception");
		}
	}

	return result;
}


static PyMethodDef pyvaex_functions[] = {
        {"range_check", (PyCFunction)range_check_, METH_VARARGS, ""},
        {"nansum", (PyCFunction)nansum_, METH_VARARGS, ""},
        {"sum", (PyCFunction)sum_, METH_VARARGS, ""},
        {"find_nan_min_max", (PyCFunction)find_nan_min_max_, METH_VARARGS, ""},
        {"histogram1d", (PyCFunction)histogram1d_, METH_VARARGS, ""},
        {"histogram2d", (PyCFunction)histogram2d_, METH_VARARGS, ""},
        {"histogram2d_f4", (PyCFunction)histogram2d_f4_, METH_VARARGS, ""},
        {"histogram3d", (PyCFunction)histogram3d_, METH_VARARGS, ""},
        {"histogramNd", (PyCFunction)histogramNd_, METH_VARARGS, ""},
        {"statisticNd_f8", (PyCFunction)statisticNd_<double, NPY_DOUBLE>, METH_VARARGS, ""},
        {"statisticNd_f4", (PyCFunction)statisticNd_<float, NPY_FLOAT>, METH_VARARGS, ""},
        {"grid_find_edges", (PyCFunction)grid_find_edges_, METH_VARARGS, ""},
        {"grid_interpolate", (PyCFunction)grid_interpolate_, METH_VARARGS, ""},
        {"project", (PyCFunction)project_, METH_VARARGS, ""},
        {"pnpoly", (PyCFunction)pnpoly_, METH_VARARGS, ""},
        {"soneira_peebles", (PyCFunction)soneira_peebles_, METH_VARARGS, ""},
        {"shuffled_sequence", (PyCFunction)shuffled_sequence_, METH_VARARGS, ""},
        {"resize", (PyCFunction)resize_, METH_VARARGS, ""},
    { NULL, NULL, 0 }
};

struct module_state {

    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3

static int vaexfast_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int vaexfast_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "vaex.vaexfast",
        NULL,
        sizeof(struct module_state),
        pyvaex_functions,
        NULL,
        vaexfast_traverse,
        vaexfast_clear,
        NULL
};
#define INITERROR return NULL
#else
#define INITERROR return
#endif


#if PY_MAJOR_VERSION >= 3
extern "C" PyObject *
PyInit_vaexfast(void)
#else
PyMODINIT_FUNC
initvaexfast(void)
#endif
{
	import_array();
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
	PyObject *module = Py_InitModule("vaex.vaexfast", pyvaex_functions);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("vaex.vaexfast.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
