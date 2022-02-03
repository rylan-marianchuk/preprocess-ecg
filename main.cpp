#include <iostream>
#include "pugixml.hpp"
#include "base64.h"
#include <string>
#include <filesystem>
#include "hdf5.h"

namespace fs = std::filesystem;

#define LEADS 8

int main() {

    std::string path = "/home/rylan/May_2019_XML/";

    int total = 10;
    for (const auto & entry : fs::directory_iterator(path)){

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(entry.path().string().c_str());


        if (result){
            // Parsed without error
            pugi::xml_node wvfm = doc.child("RestingECG").child("Waveform").next_sibling();
            pugi::xml_node leadData = wvfm.child("LeadData");
            int fs = std::stoi(wvfm.child("SampleBase").child_value());
            const int SAMPLES = fs*10;
            //std::cout << fs << std::endl;
            //std::cout << SAMPLES << std::endl;

            double ecg[LEADS][SAMPLES];

            for (int lead = 0; lead < 8; lead++){
                std::string b64encoded = leadData.child("WaveFormData").child_value();

                if (b64encoded.empty())
                    continue;
                b64encoded.erase(b64encoded.size() - 11);
                std::string DE = base64_decode(b64encoded, true);
                const char * c = DE.c_str();

                for (size_t i = 0; i < DE.size(); i += 2) {
                    size_t ind = i / 2;
                    ecg[lead][ind] = (double) *(int16_t *) &c[i];
                }

                //std::cout << "\n" << leadData.child("LeadID").child_value() << std::endl;
                //for (size_t j = 0; j < fs*10; j++){
                //    std::cout << ecg[lead][j] << "  ";
                //}
                leadData = leadData.next_sibling();
            }

            // Writing the h5 file
            std::string filename = entry.path().filename();
            filename.erase(filename.size() - 4);
            filename.append(".h5");
            std::cout << filename << std::endl;

            std::string dataset = "ECG";

            hid_t file, space, dset;
            herr_t status;
            hsize_t dims[2] = {LEADS, (long long unsigned int)SAMPLES};

            file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            space = H5Screate_simple(2, dims, NULL);


            dset = H5Dcreate(file, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                              ecg[0]);

            status = H5Dclose(dset);
            status = H5Sclose(space);
            status = H5Fclose(file);
        }
        else{
            // Error in parsing
            std::cout << "Error in parsing\n";
            continue;
        }
        total--;
        if (total < 0)
            break;
    }

    return 0;
}
