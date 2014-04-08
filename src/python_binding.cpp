#include <boost/python.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdexcept>
#include <cstdio>
#include <omp.h>
#include "ShortVolume.h"
#include "MaxTree3d.h"

extern "C" {
#include "maxtreeForSpaces.h"
#include "kerneldensity.h"
#include "pgm.h"
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
template<typename T>
void object_to_numpy3d(T* &ptr, object &obj_boost, int &count_x, int &count_y, int &count_z, int type=NPY_DOUBLE, int flags=NPY_CONTIGUOUS | NPY_ALIGNED) {
		PyObject* obj = PyArray_FROM_OTF(obj_boost.ptr(), type, flags);
		if(obj == NULL)
			throw std::runtime_error("cannot convert to numpy array");
		if((int)PyArray_NDIM(obj) != 3)
			throw std::runtime_error("array is not 3d");
		int sizex = PyArray_DIMS(obj)[0];
		int sizey = PyArray_DIMS(obj)[1];
		int sizez = PyArray_DIMS(obj)[2];
		if((count_x >= 0) && (sizex != count_x)) {
			fprintf(stderr, "size of first dimension is %d, expected %d\n", sizex, count_x);
			throw std::runtime_error("first dimension not of equal size");
		}
		if((count_y >= 0) && (sizey != count_y)) {
			fprintf(stderr, "size of second dimension is %d, expected %d\n", sizey, count_y);
			throw std::runtime_error("second dimension not of equal size");
		}
		if((count_z >= 0) && (sizez != count_z)) {
			fprintf(stderr, "size of third dimension is %d, expected %d\n", sizez, count_z);
			throw std::runtime_error("third dimension not of equal size");
		}
		ptr = (T*)PyArray_DATA(obj);
		count_x = sizex;
		count_y = sizey;
		count_z = sizez;
}

class DensityMap1d {
public:
	DensityMap1d(double xmin, double xmax, int xbins) {
		init_map1d(&map, xmin, xmax, xbins);
	}
	void pgm_write(char *filename) {
		ImagePGMBinWrite1D_map(&map, filename);
	}
	double _comp_data_probs(double xwidth, object xdata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *prob_ptr = NULL;;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		return comp_data_probs_1d(&map, xwidth, num_data, xdata_ptr, prob_ptr);
	}
	void _adaptive_density(double xwidth, double gmean, object xdata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		adaptive_dens_1d(&map, xwidth, gmean, num_data, xdata_ptr, prob_ptr);
	}
	void _fill(object image_py) {
		double *image_ptr = NULL;
		int sizex = -1;
		object_to_numpy1d(image_ptr, image_py, sizex);
		if(sizex != map.bins) {
			fprintf(stderr, "size of first dimension is %d, should be %d\n", sizex, map.bins);
			throw std::runtime_error(std::string("first dimension size not correct"));
		}
		fill(image_ptr);
	}
	
	void fill(double *image) {
		for(int x=0; x < map.bins; x++) {
			image[x] = map.map[x];
		}
	}
private:
	map1d map;
};

class DensityMap2d {
public:
	DensityMap2d(double xmin, double xmax, int xbins, double ymin, double ymax, int ybins) {
		init_map2d(&map, xmin, xmax, xbins, ymin, ymax, ybins);
	}
	
	void pgm_write(char* filename) {
		ImagePGMBinWrite_map(&map, filename);
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
		object_to_numpy1d(xdata_ptr, xdata, num_data);
		object_to_numpy1d(ydata_ptr, ydata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		return comp_data_probs_2d(xwidth, ywidth, num_data, xdata_ptr, ydata_ptr, prob_ptr);
		
	}
	double comp_data_probs_2d(double xwidth,
                            double ywidth,
                            int    num_data,
			    double *xdata,
			    double *ydata,
			    double *prob)
	{
		map2d *density = &map; // alias
		/*//return ::comp_data_probs_2d(density, xwidth, ywidth, num_data, xdata, ydata, prob);
  int i,k,l;
  double normalization = 1/( (double)num_data*xwidth*ywidth*M_PI/2.0 ), 
         gmean = 0.0;
  double xstep = (density->x_max-density->x_min)/(double)(density->x_bins - 1);
  double ystep = (density->y_max-density->y_min)/(double)(density->y_bins - 1);


  for (i=0; i<num_data; i++)
    add_one_point_epan(xdata[i],ydata[i],density,xwidth,ywidth);
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

  for (i=0; i<num_data; i++)
    {
      prob[i]=::prob_from_map2d(density,xdata[i],ydata[i],xstep,ystep);
      gmean += log(prob[i]);
    }

  for (k=0; k<density->y_bins; k++)
    for (l=0; l<density->x_bins; l++)
      density->map[k][l]=0;

  return exp(gmean/num_data);		*/
		//return ::comp_data_probs_2d(density, xwidth, ywidth, num_data, xdata, ydata, prob);
		//?int i,k,l;
		double normalization = 1/( (double)num_data*xwidth*ywidth*M_PI/2.0 ), 
		gmean = 0.0;
		double xstep = (density->x_max-density->x_min)/(double)(density->x_bins - 1);
		double ystep = (density->y_max-density->y_min)/(double)(density->y_bins - 1);

		int tid = 0;
		map2d *maps = NULL;
		#pragma omp parallel private(tid)
		{
			tid = omp_get_thread_num();
			int tmax = omp_get_num_threads();

			#pragma omp single
			{
				maps = (map2d*)calloc(sizeof(map2d), tmax);
				for(int q=0; q < tmax; q++) {
					init_map2d(&maps[q], map.x_min, map.x_max, map.x_bins, map.y_min, map.y_max, map.y_bins);
				}
			}
			#pragma omp barrier // wait till above finishes (although omp single is auto barrier'ed)
			#pragma omp for
			for (int q=0; q<num_data; q++) {
				add_one_point_epan(xdata[q],ydata[q],&maps[tid],xwidth,ywidth);
			}
			#pragma omp barrier
			// now merge all data
			#pragma omp for
			for(int y=0; y < map.y_bins; y++) {
				for (int w=0; w<tmax; w++)
					for(int x=0; x < map.x_bins; x++)
					{
						map.map[y][x] += maps[w].map[y][x];
					}
			}
			#pragma omp single
			{
				for(int i=0; i < tmax; i++)
					exit_map2d(&maps[i]);
				free(maps);
			}
			
		}
		/*/
		for (i=0; i<num_data; i++)
			add_one_point_epan(xdata[i],ydata[i],density,xwidth,ywidth);
		//*/

		density->data_max  =density->map[0][0]*normalization;
		density->data_min = density->map[0][0]*normalization;
		for (int k=0; k<density->y_bins; k++) {
			for (int l=0; l<density->x_bins; l++) {
				density->map[k][l] = density->map[k][l]*normalization;
				if (density->map[k][l]>density->data_max) 
					density->data_max=density->map[k][l];
				else
					if (density->map[k][l]< density->data_min)
				density->data_min = density->map[k][l];
			}
		}

		//#pragma omp parallel for reduction(+:gmean) schedule(auto)
		for (int i=0; i<num_data; i++) {
			prob[i] = this->prob_from_map2d(xdata[i],ydata[i],xstep,ystep);
			gmean += log(prob[i]);
		}

		//#pragma omp parallel for schedule(auto)
		for (int k=0; k<density->y_bins; k++)
			for (int l=0; l<density->x_bins; l++)
				density->map[k][l]=0;
			
			return exp(gmean/num_data);
	}
	
	void _comp_density_2d (double xwidth, double ywidth, double gmean, object xdata, object ydata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
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
		int k,l;
		map2d *density = &map; // alias
		double normalization = 1/( (double)num_data*M_PI/2.0 );

		int tid = 0;
		map2d *maps = NULL;
		#pragma omp parallel private(tid)
		{
			tid = omp_get_thread_num();
			int tmax = omp_get_num_threads();

			#pragma omp single
			{
				maps = (map2d*)calloc(sizeof(map2d), tmax);
				for(int q=0; q < tmax; q++) {
					init_map2d(&maps[q], map.x_min, map.x_max, map.x_bins, map.y_min, map.y_max, map.y_bins);
				}
			}
			#pragma omp barrier // wait till above finishes (although omp single is auto barrier'ed)
			#pragma omp for
			for (int q=0; q<num_data; q++) {
				add_one_point_epan2(xdata[q],ydata[q],
						&maps[tid],
						xwidth/sqrt(prob[q]/gmean),
						ywidth/sqrt(prob[q]/gmean));
			}
			#pragma omp barrier
			// now merge all data
			#pragma omp for
			for(int y=0; y < map.y_bins; y++) {
				for (int w=0; w<tmax; w++)
					for(int x=0; x < map.x_bins; x++)
					{
						map.map[y][x] += maps[w].map[y][x];
					}
			}
			#pragma omp single
			{
				for(int i=0; i < tmax; i++)
					exit_map2d(&maps[i]);
				free(maps);
			}
			
		}
		density->data_max  =density->map[0][0]*normalization;
		density->data_min = density->map[0][0]*normalization;
		for (k=0; k<density->y_bins; k++) {
			for (l=0; l<density->x_bins; l++)
			{
				density->map[k][l] = density->map[k][l]*normalization;
				if (density->map[k][l]>density->data_max) 
					density->data_max=density->map[k][l];
				else if (density->map[k][l]< density->data_min)
					density->data_min = density->map[k][l];
			}
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


class DensityMap3d {
public:
	DensityMap3d(double xmin, double xmax, int xbins, double ymin, double ymax, int ybins, double zmin, double zmax, int zbins) {
		init_map3d(&map, xmin, xmax, xbins, ymin, ymax, ybins, zmin, zmax, zbins);
	}
	void _avs_write(int scaling, char *fname) {
		AVSdens3dwrite(scaling, &map, fname);
	}
	double _comp_data_probs_3d(double xwidth, double ywidth, double zwidth, object xdata, object ydata, object zdata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *zdata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
		object_to_numpy1d(ydata_ptr, ydata, num_data);
		object_to_numpy1d(zdata_ptr, zdata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		return comp_data_probs_3d(&map, xwidth, ywidth, zwidth, num_data, xdata_ptr, ydata_ptr, zdata_ptr, prob_ptr);
	}
	void _comp_density_3d (double xwidth, double ywidth, double zwidth, double gmean, object xdata, object ydata, object zdata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *zdata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
		object_to_numpy1d(ydata_ptr, ydata, num_data);
		object_to_numpy1d(zdata_ptr, zdata, num_data);
		object_to_numpy1d(prob_ptr, prob, num_data);
		comp_density_3d(&map, xwidth, ywidth, zwidth, gmean, num_data, xdata_ptr, ydata_ptr, zdata_ptr, prob_ptr);
	}
	void _fill(object image_py) {
		double *image_ptr = NULL;
		int sizex = -1, sizey = -1, sizez = -1;
		object_to_numpy3d(image_ptr, image_py, sizex, sizey, sizez);
		if(sizex != map.x_bins) {
			fprintf(stderr, "size of first dimension is %d, should be %d\n", sizex, map.x_bins);
			throw std::runtime_error(std::string("first dimension size not correct"));
		}
		if(sizey != map.y_bins) {
			fprintf(stderr, "size of second dimension is %d, should be %d\n", sizey, map.y_bins);
			throw std::runtime_error("second dimension size not correct");
		}
		if(sizez != map.z_bins) {
			fprintf(stderr, "size of third dimension is %d, should be %d\n", sizez, map.z_bins);
			throw std::runtime_error("third dimension size not correct");
		}
		fill(image_ptr);
	}
	
	void fill(double *image) {
		for(int x=0; x < map.x_bins; x++) {
			for(int y=0; y < map.y_bins; y++) {
				for(int z=0; z < map.z_bins; z++) {
					//printf("x,y = %d,%d\n", x, y);
					image[x+y*map.x_bins + z*map.x_bins*map.y_bins] = map.map[z][y][x];
					//printf("%f ", map.map[y][x]);
				}
			}
		}
	}
	
		/*
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
		object_to_numpy1d(xdata_ptr, xdata, num_data);
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

		int tid = 0;
		//*
		map2d *maps = NULL;
		#pragma omp parallel private(tid)
		{
			int tid = omp_get_thread_num();
			int tmax = omp_get_num_threads();

			#pragma omp single
			{
				maps = (map2d*)calloc(sizeof(map2d), tmax);
				for(int q=0; q < tmax; q++) {
					init_map2d(&maps[q], map.x_min, map.x_max, map.x_bins, map.y_min, map.y_max, map.y_bins);
				}
			}
			#pragma omp barrier // wait till above finishes (although omp single is auto barrier'ed)
			#pragma omp for schedule(auto)
			for (int q=0; q<num_data; q++) {
				add_one_point_epan(xdata[q],ydata[q],&maps[tid],xwidth,ywidth);
			}
			#pragma omp barrier
			// now merge all data
			#pragma omp for
			for(int y=0; y < map.y_bins; y++) {
				for (int w=0; w<tmax; w++)
					for(int x=0; x < map.x_bins; x++)
					{
						map.map[y][x] += maps[w].map[y][x];
					}
			}
			#pragma omp single
			{
				for(int i=0; i < tmax; i++)
					exit_map2d(&maps[i]);
				free(maps);
			}
			
		}

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

		#pragma omp parallel for reduction(+:gmean) schedule(auto)
		for (int i=0; i<num_data; i++) {
			prob[i] = this->prob_from_map2d(xdata[i],ydata[i],xstep,ystep);
			gmean += log(prob[i]);
		}

		//#pragma omp parallel for schedule(auto)
		for (int k=0; k<density->y_bins; k++)
			for (int l=0; l<density->x_bins; l++)
				density->map[k][l]=0;
			
			return exp(gmean/num_data);
	}
	
	void _comp_density_2d (double xwidth, double ywidth, double gmean, object xdata, object ydata, object prob) {
		int num_data = -1;
		double *xdata_ptr = NULL, *ydata_ptr = NULL, *prob_ptr = NULL;
		object_to_numpy1d(xdata_ptr, xdata, num_data);
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

		int tid = 0;
		map2d *maps = NULL;
		#pragma omp parallel private(tid)
		{
			int tid = omp_get_thread_num();
			int tmax = omp_get_num_threads();

			#pragma omp single
			{
				maps = (map2d*)calloc(sizeof(map2d), tmax);
				for(int q=0; q < tmax; q++) {
					init_map2d(&maps[q], map.x_min, map.x_max, map.x_bins, map.y_min, map.y_max, map.y_bins);
				}
			}
			#pragma omp barrier // wait till above finishes (although omp single is auto barrier'ed)
			#pragma omp for schedule(auto)
			for (int q=0; q<num_data; q++) {
				add_one_point_epan2(xdata[q],ydata[q],
						&maps[tid],
						xwidth/sqrt(prob[q]/gmean),
						ywidth/sqrt(prob[q]/gmean));
			}
			#pragma omp barrier
			// now merge all data
			#pragma omp for
			for(int y=0; y < map.y_bins; y++) {
				for (int w=0; w<tmax; w++)
					for(int x=0; x < map.x_bins; x++)
					{
						map.map[y][x] += maps[w].map[y][x];
					}
			}
			#pragma omp single
			{
				for(int i=0; i < tmax; i++)
					exit_map2d(&maps[i]);
				free(maps);
			}
			
		}
		density->data_max  =density->map[0][0]*normalization;
		density->data_min = density->map[0][0]*normalization;
		for (k=0; k<density->y_bins; k++) {
			for (l=0; l<density->x_bins; l++)
			{
				density->map[k][l] = density->map[k][l]*normalization;
				if (density->map[k][l]>density->data_max) 
					density->data_max=density->map[k][l];
				else if (density->map[k][l]< density->data_min)
					density->data_min = density->map[k][l];
			}
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
	}*/

	
private:
	map3d map;
	
};


void hello() {
	printf("hello\n");
}

double maxtree1d_quality(object xdata)
{
	int num_data = -1;
	double *xdata_ptr = NULL;
	object_to_numpy1d(xdata_ptr, xdata, num_data);
	int width = num_data, height = 1;
	double datamin = xdata_ptr[0];
	double datamax = xdata_ptr[0];
	for(int i = 0; i < num_data; i++) {
		if(xdata_ptr[i] < datamin)
			datamin = xdata_ptr[i];
		if(xdata_ptr[i] > datamax)
			datamax = xdata_ptr[i];
	}
	ImageGray *img = ImageGrayCreate(width, height);
	//printf("datamax: %f\n", datamax);
	for(int i = 0; i < num_data; i++) {
		//img->Pixmap[i] = (unsigned char)((xdata_ptr[i] - datamin)/(datamax - datamin) * 255.);
		img->Pixmap[i] = (unsigned char)(sqrt(xdata_ptr[i])/sqrt(datamax) * 255.);
		//printf("%d %d\n", i, img->Pixmap[i]);
	}
	int nLeaves = 0;
	return MaxTreeforSpacesNoFile(img, 2, &nLeaves);
}

double maxtree2d_quality(object image)
{
	int size1 = -1, size2 = -1;
	double *image_ptr = NULL;
	object_to_numpy2d(image_ptr, image, size1, size2);
	double datamin = image_ptr[0];
	double datamax = image_ptr[0];
	for(int i = 0; i < (size1*size2); i++) {
		if(image_ptr[i] < datamin)
			datamin = image_ptr[i];
		if(image_ptr[i] > datamax)
			datamax = image_ptr[i];
	}
	ImageGray *img = ImageGrayCreate(size1, size2); // TODO: free it, memleak
	//*
	for(int i = 0; i < (size1*size2); i++) {
		//img->Pixmap[i] = (unsigned char)((image_ptr[i] - datamin)/(datamax - datamin) * 255.);
		img->Pixmap[i] = (unsigned char)(sqrt(image_ptr[i])/sqrt(datamax) * 255.);
		//printf("%d %d\n", i, img->Pixmap[i]);
	}
	/*/
	for(int x = 0; x < (size1); x++) {
	for(int y = 0; y < (size2); y++) {
		//img->Pixmap[i] = (unsigned char)((image_ptr[i] - datamin)/(datamax - datamin) * 255.);
		int i = x + y * size2;
		int j = x + (size2-1-y) * size2;
		//int j = i;
		img->Pixmap[i] = (unsigned char)(sqrt(image_ptr[j])/sqrt(datamax) * 255.);
		//printf("%d %d\n", i, img->Pixmap[i]);
	}}
	printf("%d %d\n", size1, size2);
	ImagePGMBinWrite(img, "test.pgm");*/
	int nLeaves = 0;
	return MaxTreeforSpacesNoFile(img, 4, &nLeaves);
}

double maxtree3d_quality(object volume, double scaling)
{
	int size1 = -1, size2 = -1, size3 = -1;
	double *volume_ptr = NULL;
	object_to_numpy3d(volume_ptr, volume, size1, size2, size3);
	double datamin = volume_ptr[0];
	double datamax = volume_ptr[0];
	for(int i = 0; i < (size1*size2*size3); i++) {
		if(volume_ptr[i] < datamin)
			datamin = volume_ptr[i];
		if(volume_ptr[i] > datamax)
			datamax = volume_ptr[i];
	}
	ShortVolume * shortVolume = new ShortVolume(size1, size2, size3);
	for(int x = 0; x < size1; x++) {
	for(int y = 0; y < size2; y++) {
	for(int z = 0; z < size3; z++) {
		unsigned short value = (unsigned short) (volume_ptr[x + y * size2 + z * size2*size3] / datamax * scaling);
		//if( (x == 10) && (y == 10) )
		//printf("%f ", (double)value);
		shortVolume->putVoxel(x, y, z, value);
	}
	}
	}
	
	shortVolume->createHistogram();
	MaxTree3d *maxTree = new MaxTree3d(*shortVolume, 26);

	int nrOfLeaveNodes = maxTree->getNumberOfLeaves();
	//printf("NumberofLeavesNodes:%d\n",nrOfLeaveNodes);
	double quality=maxTree->getLeaves();
	return quality/nrOfLeaveNodes;
	
}

PyObject* maxtree1d_and_2d_dynamics(ImageGray *img);

PyObject* maxtree1d_dynamics(object xdata)
{
	int num_data = -1;
	double *xdata_ptr = NULL;
	object_to_numpy1d(xdata_ptr, xdata, num_data);
	int width = num_data, height = 1;
	double datamin = xdata_ptr[0];
	double datamax = xdata_ptr[0];
	for(int i = 0; i < num_data; i++) {
		if(xdata_ptr[i] < datamin)
			datamin = xdata_ptr[i];
		if(xdata_ptr[i] > datamax)
			datamax = xdata_ptr[i];
	}
	ImageGray *img = ImageGrayCreate(width, height);
	//printf("datamax: %f\n", datamax);
	for(int i = 0; i < num_data; i++) {
		//img->Pixmap[i] = (unsigned char)((xdata_ptr[i] - datamin)/(datamax - datamin) * 255.);
		img->Pixmap[i] = (unsigned char)(sqrt(xdata_ptr[i])/sqrt(datamax) * 255.);
		//printf("%d %d\n", i, img->Pixmap[i]);
	}
	return maxtree1d_and_2d_dynamics(img);
}

PyObject* maxtree2d_dynamics(object image)
{
	int size1 = -1, size2 = -1;
	double *image_ptr = NULL;
	object_to_numpy2d(image_ptr, image, size1, size2);
	double datamin = image_ptr[0];
	double datamax = image_ptr[0];
	for(int i = 0; i < (size1*size2); i++) {
		if(image_ptr[i] < datamin)
			datamin = image_ptr[i];
		if(image_ptr[i] > datamax)
			datamax = image_ptr[i];
	}
	ImageGray *img = ImageGrayCreate(size1, size2); // TODO: free it, memleak
	for(int i = 0; i < (size1*size2); i++) {
		img->Pixmap[i] = (unsigned char)(sqrt(image_ptr[i])/sqrt(datamax) * 255.);
	}
	return maxtree1d_and_2d_dynamics(img);
}

PyObject* maxtree1d_and_2d_dynamics(ImageGray *img)
{
	int CONNECTIVITY = 4;
	
	ImageGray *template_, *out;
	MaxTree *mt;
	char *imgfname, *template_fname = NULL, *outfname = "out.pgm";
	double lambda=2;
	int attrib=2, decision=3, r,NumLeaves,i;
	double *dyn,*ElongationOfInflunceZone,*Xextent,*Yextent,quality;
	ulong *AreaOfInfluence;


	template_ = GetTemplate(template_fname, img);
	if (template_==NULL)
	{
		fprintf(stderr, "Can't create template_\n");
		ImageGrayDelete(img);
		return NULL;
	}

	out = ImageGrayCreate(img->Width, img->Height);
	if (out==NULL)
	{
		fprintf(stderr, "Can't create output image\n");
		ImageGrayDelete(template_);
		ImageGrayDelete(img);
		return NULL;
	}
	mt = MaxTreeCreate(img, template_, Attribs[attrib].NewAuxData, Attribs[attrib].AddToAuxData, Attribs[attrib].MergeAuxData, Attribs[attrib].PostAuxData, Attribs[attrib].DeleteAuxData,CONNECTIVITY);
	if (mt==NULL)
	{
		fprintf(stderr, "Can't create Max-tree\n");
		ImageGrayDelete(out);
		ImageGrayDelete(template_);
		ImageGrayDelete(img);
		return NULL;
	}
	NumLeaves=NumberOfLeaves(mt);
	//nLeaves[0]=NumLeaves;
	dyn = (double *)calloc(NumLeaves,sizeof(double));
	AreaOfInfluence= (ulong *)calloc(NumLeaves,sizeof(ulong));
	ElongationOfInflunceZone= (double *)calloc(NumLeaves,sizeof(double));
	Xextent=(double *)calloc(NumLeaves,sizeof(double));
	Yextent=(double *)calloc(NumLeaves,sizeof(double));
	quality=CalcDynamic(mt,NumLeaves,dyn,AreaOfInfluence,ElongationOfInflunceZone,Xextent,Yextent,Attribs[attrib].Attribute);

	//FILE *outfile = fopen("InterestingSubspacesNew.txt","a");

	// pixel_qsort (dyn,NumLeaves);
	//fprintf(outfile,"NumberOfLocalMaxima:%d\nQuality:%lf\n",NumLeaves,quality);
	//fprintf(outfile,"%s\n","Dynamics,AreaOfInfluence,ElongatonOfInfluenceArea,X-extent,Y-extent");
	PyObject* dynamics = PyList_New(NumLeaves);
	PyObject* x = PyList_New(NumLeaves);
	PyObject* y = PyList_New(NumLeaves);
	for (i=NumLeaves-1;i>=0;i--) {
		//printf("%lf,%u,%lf,%lf,%lf\n",dyn[i],AreaOfInfluence[i],ElongationOfInflunceZone[i],Xextent[i],Yextent[i]);
		PyList_SetItem(dynamics, i, PyFloat_FromDouble(dyn[i]));
		PyList_SetItem(x, i, PyFloat_FromDouble(Xextent[i]));
		PyList_SetItem(y, i, PyFloat_FromDouble(Yextent[i]));
	}

	Decisions[decision].Filter(mt, img, template_, out, Attribs[attrib].Attribute, lambda);
	//PrintFilterStatistics(mt, Attribs[attrib].Attribute, img, out);
	MaxTreeDelete(mt);
	r = ImagePGMBinWrite(out, outfname);
	// if (r)  fprintf(stderr, "Error writing image '%s'\n", outfname);
	//else  printf("Filtered image written to '%s'\n", outfname);
	//fclose(outfile);
	ImageGrayDelete(out);
	ImageGrayDelete(template_);
	ImageGrayDelete(img);
	free(AreaOfInfluence);
	free(ElongationOfInflunceZone);
	free(Xextent);
	free(Yextent);
	free(dyn);

	return Py_BuildValue("NNN", dynamics, x, y);
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
	
	def("maxtree1d_quality", maxtree1d_quality);
	def("maxtree2d_quality", maxtree2d_quality);
	def("maxtree3d_quality", maxtree3d_quality);
	
	def("maxtree1d_dynamics", maxtree1d_dynamics);
	def("maxtree2d_dynamics", maxtree2d_dynamics);

	class_<DensityMap1d >("DensityMap1d", init<double, double, int>())
		//.def("pgm_write", &DensityMap2d::pgm_write)
		//.def("test", &DensityMap2d::test)
		//.def("fill", &DensityMap2d::_fill)
		.def("pgm_write", &DensityMap1d::pgm_write)
		.def("comp_data_probs", &DensityMap1d::_comp_data_probs)
		.def("adaptive_density", &DensityMap1d::_adaptive_density)
		//.def("avs_write", &DensityMap3d::_avs_write)
		.def("fill", &DensityMap1d::_fill)
		
		//.def("comp_density_2d", &DensityMap2d::_comp_density_2d)
	;
	
	class_<DensityMap2d >("DensityMap2d", init<double, double, int, double, double, int>())
		.def("pgm_write", &DensityMap2d::pgm_write)
		.def("test", &DensityMap2d::test)
		.def("fill", &DensityMap2d::_fill)
		.def("comp_data_probs_2d", &DensityMap2d::_comp_data_probs_2d)
		.def("comp_density_2d", &DensityMap2d::_comp_density_2d)
	;
	class_<DensityMap3d >("DensityMap3d", init<double, double, int, double, double, int, double, double, int>())
		//.def("pgm_write", &DensityMap2d::pgm_write)
		//.def("test", &DensityMap2d::test)
		//.def("fill", &DensityMap2d::_fill)
		.def("comp_data_probs_3d", &DensityMap3d::_comp_data_probs_3d)
		.def("comp_density_3d", &DensityMap3d::_comp_density_3d)
		.def("avs_write", &DensityMap3d::_avs_write)
		.def("fill", &DensityMap3d::_fill)
		
		//.def("comp_density_2d", &DensityMap2d::_comp_density_2d)
	;
}


