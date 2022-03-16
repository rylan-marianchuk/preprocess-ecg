#include "hdf5.h"
#include "../include/writeh5.h"
#include "H5Cpp.h"
#include <string>
#include <iostream>

using namespace H5;

void writeh5(std::string FILE, std::string write_path, const int LEADS, const int SAMPLES, float * wdata) {
    // Writing the h5 file
    std::string filename = FILE;
    filename.erase(filename.size() - 4);
    filename.append(".h5");
    std::string write_to = write_path;
    write_to.append(filename);


    const H5std_string FILE_NAME(write_to);
    const H5std_string DATASET_NAME("ECG");


    // Try block to detect exceptions raised by any of the calls inside it
    try {
        // Turn off the auto-printing when failure occurs so that we can
        // handle the errors appropriately
        Exception::dontPrint();

        // Create a new file using the default property lists.
        H5File file(FILE_NAME, H5F_ACC_TRUNC);

        // Create the data space for the dataset.
        hsize_t dims[1]; // dataset dimensions
        dims[0] = LEADS*SAMPLES;
        //dims[1] = SAMPLES;
        DataSpace dataspace(1, dims);

        // Create the dataset.
        DataSet dataset = file.createDataSet(DATASET_NAME, PredType::NATIVE_FLOAT, dataspace);
        dataset.write(wdata, PredType::NATIVE_FLOAT);

    } // end of try block

        // catch failure caused by the H5File operations
    catch (FileIException error) {
        error.printErrorStack();
        return;
    }

        // catch failure caused by the DataSet operations
    catch (DataSetIException error) {
        error.printErrorStack();
        return;
    }

        // catch failure caused by the DataSpace operations
    catch (DataSpaceIException error) {
        error.printErrorStack();
        return;
    }
}

