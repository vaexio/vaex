/*
compile as: gcc -Wall -std=c99 -o ascii_to_hdf5 ascii_to_hdf5.c -lhdf5
run as: ./ascii_to_hdf5 example.hdf5 ../../data/helmi2000-header.asc 3300000 3
	arguments are: output filename, input filename, rows, columns

*/
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h> 
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#define MAX_COLUMNS 512

char column_names[MAX_COLUMNS][512];


static void
check (int test, const char * message, ...)
{
	if (test) {
		va_list args;
		va_start (args, message);
		vfprintf (stderr, message, args);
		va_end (args);
		fprintf (stderr, "\n");
		exit (EXIT_FAILURE);
	}
}

int main(int argc, char *argv[])
{
	hid_t		file;    /* Handles */
	herr_t		status;
	haddr_t		offsets[MAX_COLUMNS];
	hsize_t		dims[1];
	
	char* filename_output = argv[1];
	char* filename_input = argv[2];
	FILE* file_input = fopen(filename_input, "r");

	int no_rows = atoi(argv[3]);
	int no_columns = atoi(argv[4]);
	dims[0] = no_rows;
				

	
	// create the file and the group 'columns', which vaex will expect
	file = H5Fcreate(filename_output, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hid_t group = H5Gcreate1(file, "data", 0);
	
	// find the column names in the first line
	for(int i=0; i<no_columns; i++) {
		fscanf(file_input," %s", column_names[i]);
		printf("column[%d]: %s\n", i, column_names[i]);
	}
	fscanf(file_input," \n");

	// just create the dataspace using the HDF5 library, and ask for the offset from the beginning of the file
	for(int i = 0; i < no_columns; i++)  {
		hid_t space = H5Screate_simple(1, dims, NULL);

		
		hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
		H5Pset_layout (dcpl, H5D_CONTIGUOUS); // compact allows us the memory map the file
		H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY); // need this to allocate the space so offset exists
		hid_t dset = H5Dcreate(group, column_names[i], H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		
		offsets[i] = H5Dget_offset(dset);
		
		H5D_space_status_t space_status;
		H5Dget_space_status(dset, &space_status);
		printf("offset[%d] = %x allocated: %s\n", i, (unsigned int)offsets[i], (space_status == H5D_SPACE_STATUS_ALLOCATED ? "yes" : "no"));

		status = H5Dclose (dset);
		status = H5Pclose (dcpl);
		status = H5Sclose (space);
	}
	//close the group and file
	H5Gclose(group);
	status = H5Fclose (file);
	
	
	// now we can simpy memory map the file (meaning we tread the file as one big 'array'
	// the offsets will tell us where we can write the columns
	
	struct stat s;
	status = stat(filename_output,  &s);
	check (status < 0, "stat %s failed: %s", filename_output, strerror (errno));
	printf("file size: %lld\n", (unsigned long long)s.st_size);
	
	int fd = open(filename_output, O_RDWR);
	check (fd < 0, "open %s failed: %s", filename_output, strerror (errno));
	
    
	// the mapped pointer points to the beginning of the file
	char* mapped = mmap (0, s.st_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	check (mapped == MAP_FAILED, "mmap %s failed: %s",
           filename_output, strerror (errno));

	// read in the rows, and directly write them to the file
	for(int j=0; j<no_rows; j++) {
		for(int i=0; i<no_columns; i++) {
			double* column_ptr = (double*)(mapped+offsets[i]);
			fscanf(file_input," %lf", &column_ptr[j]);
		}
		if( ((j % 100000) == 0) & (j > 0) )
			printf("%d of %d\n", j, no_rows);
	}
	printf("done!\n");
	close(fd);
}
