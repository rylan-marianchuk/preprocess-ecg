#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>

#include "pugixml.hpp"
#include "base64.h"
#include "writeh5.h"
#include "cudaExtract.cuh"
#include "writeSQL.h"

#define LEADS 8
#define SAMPLES 5000

namespace fs = std::filesystem;


int main(int argc, char *argv[]) {
    /*
     * Invoke with parameters
     * READ_PATH (str)
     * WRITE_PATH (str)
     * MAX_XMLS (int)   : stop the program after reading and processing MAX_XMLS
     *                    - set to -1 if no max desired
     */
    std::string read_path = argv[1];
    std::string write_path = argv[2];
    const int MAX_XMLS = std::atoi(argv[3]);

    const int BATCH_SIZE = 128;

    const int ECG_BUFFER_LEN = LEADS*SAMPLES;

    float * ecgs = new float[ECG_BUFFER_LEN*BATCH_SIZE];

    int completed_total = 0;
    int completed_batch = 0;

    std::vector<std::string> filename_vector;

    std::ofstream unparsable ("unparsable.txt");

    int init = createArtifactDB();

    for (const auto & entry : fs::directory_iterator(read_path)){

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
                std::cout << "\t\t Error in parsing :( \t\t";
                unparsable << entry.path().filename() << std::endl;
                std::cout << entry.path().filename() << std::endl;
                continue;
            }
            if (fs != 500){
                continue;
            }

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
                    ecgs[(completed_batch*ECG_BUFFER_LEN) + (lead*SAMPLES) + ind] = (float) *(int16_t *) &c[i];
                }

                leadData = leadData.next_sibling();
            }

            if (!goodToWrite){
                std::cout << "\t\t Error in parsing :( \t\t";
                unparsable << entry.path().filename() << std::endl;
                std::cout << entry.path().filename() << std::endl;
                continue;
            }

        }
        else{
            // Error in parsing
            std::cout << "\t\t Error in parsing :( \t\t";
            unparsable << entry.path().filename() << std::endl;
            std::cout << entry.path().filename() << std::endl;
            continue;
        }

        //writeh5(entry.path().filename(), write_path, LEADS, SAMPLES, ecgs + (completed_batch*ECG_BUFFER_LEN));
        filename_vector.push_back(entry.path().filename());
        completed_batch++;
        completed_total++;
        if (completed_batch == BATCH_SIZE){
            // Get & write the parameters from CUDA
            cudaResults res = getArtifactParams(ecgs, BATCH_SIZE*LEADS, SAMPLES);

            int sqlWrite = writeCudaOutput(res, filename_vector, BATCH_SIZE, LEADS);

            if (sqlWrite == -1){
                std::cout << "Error writing SQL D:" << std::endl;
                return -1;
            }

            filename_vector.clear();
            completed_batch = 0;
        }
        if (completed_total == MAX_XMLS) break;
    }

    if (!filename_vector.empty()){
        // Get & write the parameters from CUDA
        cudaResults res = getArtifactParams(ecgs, filename_vector.size()*LEADS, SAMPLES);

        int sqlWrite = writeCudaOutput(res, filename_vector, filename_vector.size(), LEADS);
    }

    delete[] ecgs;
    unparsable.close();
    return 0;
}

