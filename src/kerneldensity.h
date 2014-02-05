

/* 
   3D density map type, essentially a volume data type with double 
   elements
 
*/



typedef struct { double 
		   x_min,      /* minima and maxima in x, y, and z */
		   x_max, 
		   y_min, 
		   y_max, 
		   z_min, 
		   z_max;
                 int x_bins,   /* numbers of bins */
                     y_bins,
		     z_bins;
                 double 
		   ***map;     /* the data */
                 double 
		   data_min,
		   data_max;   /* min and max of data */
               } map3d;

/* 
   2D density map type, essentially an image data type with double 
   elements
 
*/
typedef struct { double 
		   x_min,      /* minima and maxima in x, and y*/
		   x_max, 
		   y_min, 
		   y_max;
                 int x_bins,   /* numbers of bins */
                     y_bins;
                 double        /* the data */
		   **map;
                 double        /* min and max of data */
		   data_min,
		   data_max;
               } map2d;


/* 
   2 1/2 D regression map type, currently unused
*/
typedef struct { double 
		   x_min, 
		   x_max, 
		   y_min, 
		   y_max;
                 int x_bins,
                     y_bins;
                 double **dens_map,
                       **z_map;
                 double data_min,
                       data_max;
               } regress_map2d;


/* 
   1D density map type, essentially a signal data type with double 
   elements
 
*/
typedef struct { double 
		   x_min, 
		   x_max;
                 int bins;
                 double *map;
                 double data_min,
                       data_max;
               } map1d;


typedef struct { double 
		   x_min, 
		   x_max;
                 int bins,
                     validbins;
                 double *x_map,
                       *y_map,
                       *sig_map;
                 double data_min,
                       data_max;
               } regress_map1d;

typedef struct { int     num_contours;
                 double   *levels;
                 int     *colours;
               } contour_rec;


void init_map2d ( map2d *map,
                  double x_min,
		  double x_max,
                  int   x_bins,
                  double y_min,
		  double y_max,
                  int   y_bins);

void exit_map2d ( map2d *map );

void init_regress_map2d ( regress_map2d *map,
			  double x_min,
			  double x_max,
                          int   x_bins,
			  double y_min,
			  double y_max,
                          int   y_bins );

void exit_regress_map2d ( regress_map2d *map );

void init_map1d ( map1d *map,
                  double x_min,
		  double x_max,
                  int   bins);


void exit_map1d ( map1d *map );

void add_one_point_1d ( double       x_val,
                        map1d       *d,
                        double       h    );

void init_regress_map1d ( regress_map1d *map,
			  double x_min,
			  double x_max,
                          int           bins);


void exit_regress_map1d ( regress_map1d *map );

void add_one_point_regress_1d ( double         x_val,
                                double         y_val,
                                regress_map1d *d,
                                double         h    );

void clean_regress_map1d ( regress_map1d *d,
                           double min_dens   );

void clean_regress_map2d ( regress_map2d *d,
                           double min_dens   );


void add_one_point_regress_2d ( double         x_val,
                                double         y_val,
                                double         z_val,
                                regress_map2d *d,
                                double         hx,
                                double         hy    );


void init_contours ( contour_rec *contours,
                     int         num_contours,
                     double      min,
                     double      max,
                     double      log_radix,
                     int         min_col,
                     int         max_col       );





// added below: MB
void add_one_point_epan ( double       x_val,
                          double       y_val,
                          map2d       *d,
                          double       hx,
                          double       hy    );

void add_one_point_epan2 ( double       x_val,
                           double       y_val,
                           map2d       *d,
                           double       hx,
                           double       hy    );
