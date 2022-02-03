#include "hdf5.h"
#include "writeh5.h"
#include <string>
#include <iostream>

void writeh5(std::string FILE, std::string DATASET, const int DIM0, const int DIM1, float (*wdata)[8][2500]) {
    hid_t file, space, dset;
    herr_t status;
    hsize_t dims[2] = {(long long unsigned int)DIM0, (long long unsigned int)DIM1};

    /*
     * Create a new file using the default properties.
     */
    file = H5Fcreate(FILE.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /*
     * Create dataspace.  Setting maximum size to NULL sets the maximum
     * size to be the current size.
     */
    space = H5Screate_simple(2, dims, NULL);

    /*
     * Create the dataset and write the floating point data to it.  In
     * this example we will save the data as 64 bit little endian IEEE
     * floating point numbers, regardless of the native type.  The HDF5
     * library automatically converts between different floating point
     * types.
     */
    dset = H5Dcreate(file, DATASET.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT,
                     H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      (*wdata));

    /*
     * Close and release resources.
     */
    status = H5Dclose(dset);
    status = H5Sclose(space);
    status = H5Fclose(file);
}

