#include <iostream>
#include "pugixml.hpp"
#include "base64.h"
#include <string>
#include <filesystem>
#include "writeh5.h"

namespace fs = std::filesystem;

#define LEADS 8

int main(int argc, char *argv[]) {
    /*
     * Invoke with parameters
     * READ_PATH (str)
     * WRITE_PATH (str)
     * BATCH_SIZE (int) : number of xmls to read before sending the signals to the gpu for statistic computing
     * MAX_XMLS (int)   : stop the program after reading and processing MAX_XMLS
     */

    //std::string read_path = "/home/rylan/May_2019_XML/";
    std::string read_path = argv[1];
    //std::string write_path = "/home/rylan/xmls_AS_h5/";
    std::string write_path = argv[2];
    const int BATCH_SIZE = std::atoi(argv[3]);
    const int MAX_XMLS = std::atoi(argv[4]);

    int completed = 0;
    for (const auto & entry : fs::directory_iterator(read_path)){
        //std::cout << "Working on:\t\t" << entry.path().filename();
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(entry.path().string().c_str());


        if (result){
            // Parsed without error
            pugi::xml_node wvfm = doc.child("RestingECG").child("Waveform").next_sibling();
            pugi::xml_node leadData = wvfm.child("LeadData");
            int fs;

            try{
               fs = std::stoi(wvfm.child("SampleBase").child_value());
            }
            catch  (const std::exception& e) {
                std::cout << "\t\t Error in parsing :(" << std::endl;
                continue;
            }
            if (fs != 500){
                continue;
            }

            const int SAMPLES = fs*10;
            //std::cout << "\t\tfs:  " << fs;

            //float ecg[LEADS][SAMPLES];
            float ecg[LEADS*SAMPLES];
            //double * ecg;

            bool goodToWrite = true;

            for (int lead = 0; lead < 8; lead++){
                std::string b64encoded = leadData.child("WaveFormData").child_value();

                if (b64encoded.empty()){
                    goodToWrite = false;
                    break;
                }

                size_t cutOff = b64encoded.find('=');
                std::string DE = base64_decode(b64encoded.substr(0, cutOff), true);

                const char * c = DE.c_str();

                for (size_t i = 0; i < DE.size(); i += 2) {
                    size_t ind = i / 2;
                    //ecg[lead][ind] = (float) *(int16_t *) &c[i];
                    ecg[lead*SAMPLES + ind] = (float) *(int16_t *) &c[i];
                }

                leadData = leadData.next_sibling();
            }

            if (!goodToWrite){
                std::cout << "\t\t Error in parsing :(" << std::endl;
                continue;
            }

            writeh5(entry.path().filename(), write_path, LEADS, SAMPLES, ecg);
            /*
            // Writing the h5 file
            std::string filename = entry.path().filename();
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
                hsize_t dims[2]; // dataset dimensions
                dims[0] = LEADS;
                dims[1] = SAMPLES;
                DataSpace dataspace(2, dims);

                // Create the dataset.
                DataSet dataset = file.createDataSet(DATASET_NAME, PredType::NATIVE_FLOAT, dataspace);
                dataset.write(ecg, PredType::NATIVE_FLOAT);

            } // end of try block

                // catch failure caused by the H5File operations
            catch (FileIException error) {
                error.printErrorStack();
                return -1;
            }

                // catch failure caused by the DataSet operations
            catch (DataSetIException error) {
                error.printErrorStack();
                return -1;
            }

                // catch failure caused by the DataSpace operations
            catch (DataSpaceIException error) {
                error.printErrorStack();
                return -1;
            }
             */
            /*
            hid_t file, space, dset;
            herr_t status;
            hsize_t dims[2] = {LEADS, (long long unsigned int)SAMPLES};

            file = H5Fcreate(write_to.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            space = H5Screate_simple(2, dims, NULL);


            dset = H5Dcreate(file, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                              ecg[0]);

            status = H5Dclose(dset);
            status = H5Sclose(space);
            status = H5Fclose(file);
             */
            //std::cout << "\t\t FINISHED! \t\t Saved:" << 10000 - total << std::endl;
        }
        else{
            // Error in parsing
            std::cout << "\t\t Error in parsing :(" << std::endl;
            continue;
        }
        completed++;
        if (completed == MAX_XMLS)
            break;
    }

    return 0;
}
