#include <boost/python.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdexcept>
#include <cstdio>

extern "C" {
#include "kerneldensity.h"
int ImagePGMBinWrite(map2d *map, char *fname);
}
//#include <boost/numpy.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;

// TODO: DRY, merge two functions below

template<typename T>
void object_to_numpy1d(T* &ptr, object &obj_boost, int &count, int type=NPY_DOUBLE, int flags=NPY_CONTIGUOUS | NPY_ALIGNED) {
		PyObject* obj = PyArray_FROM_OTF(obj_boost.ptr(), type, flags);
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 1)
			throw std::runtime_error("array is not 1d");
		int size = PyArray_DIMS(obj)[0];
		if((count >= 0) && (size != count))
			throw std::runtime_error("arrays not of equal size");
		ptr = (T*)PyArray_DATA(obj);
		count = size;
}

template<typename T>
void object_to_numpy2d(T* &ptr, object &obj_boost, int &count_x, int &count_y, int type=NPY_DOUBLE, int flags=NPY_CONTIGUOUS | NPY_ALIGNED) {
		PyObject* obj = PyArray_FROM_OTF(obj_boost.ptr(), type, flags);
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 2)
			throw std::runtime_error("array is not 2d");
		int sizex = PyArray_DIMS(obj)[0];
		int sizey = PyArray_DIMS(obj)[1];
		if((count_x >= 0) && (sizex != count_x)) {
			fprintf(stderr, "size of first dimension is %d, expected %d\n", sizex, count_x);
			throw std::runtime_error("first dimension not of equal size");
		}
		if((count_y >= 0) && (sizey != count_y)) {
			fprintf(stderr, "size of second dimension is %d, expected %d\n", sizey, count_y);
			throw std::runtime_error("second dimension not of equal size");
		}
		ptr = (T*)PyArray_DATA(obj);
		count_x = sizex;
		count_y = sizey;
}

class DensityMap2d {
public:
	DensityMap2d(double xmin, double xmax, int xbins, double ymin, double ymax, int ybins) {
		init_map2d(&map, xmin, xmax, xbins, ymin, ymax, ybins);
	}
	
	void pgm_write(char* filename) {
		ImagePGMBinWrite(&map, filename);
	}
	
	void test(boost::python::object a) {
		PyObject *obj = a.ptr();
		printf("obj: %p\n", obj);
		obj = PyArray_FROM_OTF(obj,NPY_DOUBLE, NPY_CONTIGUOUS | NPY_ALIGNED);
		printf("obj: %p\n", obj);
		printf("array: %d\n", (int)PyArray_NDIM(obj));
		int size = PyArray_DIMS(obj)[0];
		printf("size: %d\n", size);
	}
	
	void _fill(object image_py) {
		double *image_ptr = NULL;
		int sizex = -1, sizey = -1;
		object_to_numpy2d(image_ptr, image_py, sizex, sizey);
		if(sizex != map.x_bins) {
			fprintf(stderr, "size of first dimension is %d, should be %d\n", sizex, map.x_bins);
			throw std::runtime_error(std::string("first dimension size not correct"));
		}
		if(sizey != map.y_bins) {
			fprintf(stderr, "size of econd dimension is %d, should be %d\n", sizey, map.y_bins);
			throw std::runtime_error("second dimension size not correct");
		}
		fill(image_ptr);
	}
	
	void fill(double *image) {
		for(int x=0; x < map.x_bins; x++) {
			for(int y=0; y < map.y_bins; y++) {
				//printf("x,y = %d,%d\n", x, y);
				image[x+y*map.x_bins] = map.map[y][x];
				//printf("%f ", map.map[y][x]);
			}
		}
	}

	
	double _comp_data_probs_2d(double xwidth, double ywidth, object xdata, object ydata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d<double>(xdata_ptr, xdata, num_data);
		object_to_numpy1d(ydata_ptr, ydata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		comp_data_probs_2d(xwidth, ywidth, num_data, xdata_ptr, ydata_ptr, prob_ptr);
	}
	double comp_data_probs_2d(double xwidth,
                            double ywidth,
                            int    num_data,
			    double *xdata,
			    double *ydata,
			    double *prob)
	{
		map2d *density = &map; // alias
		int i,k,l;
		double normalization = 1/( (double)num_data*xwidth*ywidth*M_PI/2.0 ), 
		gmean = 0.0;
		double xstep = (density->x_max-density->x_min)/(double)(density->x_bins - 1);
		double ystep = (density->y_max-density->y_min)/(double)(density->y_bins - 1);


		for (i=0; i<num_data; i++)
			add_one_point_epan(xdata[i],ydata[i],density,xwidth,ywidth);

		density->data_max  =density->map[0][0]*normalization;
		density->data_min = density->map[0][0]*normalization;
		for (k=0; k<density->y_bins; k++) {
			for (l=0; l<density->x_bins; l++) {
				density->map[k][l] = density->map[k][l]*normalization;
				if (density->map[k][l]>density->data_max) 
					density->data_max=density->map[k][l];
				else
					if (density->map[k][l]< density->data_min)
				density->data_min = density->map[k][l];
			}
		}

		for (i=0; i<num_data; i++) {
			prob[i] = this->prob_from_map2d(xdata[i],ydata[i],xstep,ystep);
			gmean += log(prob[i]);
		}

		for (k=0; k<density->y_bins; k++)
			for (l=0; l<density->x_bins; l++)
				density->map[k][l]=0;
			
			return exp(gmean/num_data);
	}
	
	void _comp_density_2d (double xwidth, double ywidth, double gmean, object xdata, object ydata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d<double>(xdata_ptr, xdata, num_data);
		object_to_numpy1d(ydata_ptr, ydata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		comp_density_2d(xwidth, ywidth, gmean, num_data, xdata_ptr, ydata_ptr, prob_ptr);
	}
	void comp_density_2d (double xwidth,
				double ywidth,
				double gmean,
				int    num_data,
				double *xdata,
				double *ydata,
				double *prob) {
	int i,k,l;
	map2d *density = &map; // alias
	double normalization = 1/( (double)num_data*M_PI/2.0 );

	for (i=0; i<num_data; i++) {
		add_one_point_epan2(xdata[i],ydata[i],
				density,
				xwidth/sqrt(prob[i]/gmean),
				ywidth/sqrt(prob[i]/gmean));

	}

	density->data_max  =density->map[0][0]*normalization;
	density->data_min = density->map[0][0]*normalization;
	for (k=0; k<density->y_bins; k++)
		for (l=0; l<density->x_bins; l++)
		{
		density->map[k][l] = density->map[k][l]*normalization;
			
		if (density->map[k][l]>density->data_max) 
		density->data_max=density->map[k][l];
		else
		if (density->map[k][l]< density->data_min)
			density->data_min = density->map[k][l];
		}

	}
	
	
	double prob_from_map2d(double x,
							double y,
							double xstep,
							double ystep )       
	{
		int xbin=(int)((x - map.x_min)/xstep);
		double xratio=(x-map.x_min-(xstep*xbin))/xstep;
		int ybin=(int)((y-map.y_min)/ystep);
		double yratio=(y-map.y_min-(ystep*ybin))/ystep;
		double dens1,dens2;

		dens1 = map.map[ybin][xbin]+( map.map[ybin][xbin+1]
						-map.map[ybin][xbin]  )*xratio; 
		dens2 = map.map[ybin+1][xbin]+( map.map[ybin+1][xbin+1]
						-map.map[ybin+1][xbin]  )*xratio;
		return dens1 + ( dens2 - dens1 ) * yratio;
	}	

	
private:
	map2d map;
	
};

void hello() {
	printf("hello\n");
}

BOOST_PYTHON_MODULE(subspacefind)
{
	namespace bp = boost::python;
	using namespace boost::python;
	//Py_Initialize();
	//bp::object main = bp::import("__main__");
	//bp::object global(main.attr("__dict__"));
	//global["np"] = bp::import("numpy");
	import_array();
	
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	def("hello", hello);
	class_<DensityMap2d >("DensityMap2d", init<double, double, int, double, double, int>())
		.def("pgm_write", &DensityMap2d::pgm_write)
		.def("test", &DensityMap2d::test)
		.def("fill", &DensityMap2d::_fill)
		.def("comp_data_probs_2d", &DensityMap2d::_comp_data_probs_2d)
		.def("comp_density_2d", &DensityMap2d::_comp_density_2d)
	;
}

