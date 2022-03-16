#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <chrono>
#include <variant>

#include "pugixml.hpp"
#include "base64.h"
#include "include/writeh5.h"
#include "include/sqlite_wrapper.h"
#include "include/cudaExtract.cuh"

#define LEADS 8
#define SAMPLES 5000

namespace fs = std::filesystem;


void writeBatch(std::vector<std::string> filename_vector, float * ecgs, SqliteWrapper& wvfmDB){
    /*
     *
     *
     */
    // Get & write the parameters from CUDA
    const int B_SIZE = filename_vector.size();

    std::string * euids = new std::string[B_SIZE*LEADS];
    int * lead_ids = new int[B_SIZE*LEADS];

    for (int i = 0; i < B_SIZE; i++){
        std::string filename = filename_vector[i].substr(0, filename_vector[i].size() - 3);
        for (int lead = 0; lead < LEADS; lead++){
            euids[i*LEADS + lead] = filename + std::to_string(lead);
            lead_ids[i*LEADS + lead] = lead;
        }
    }

    cudaResults res = getArtifactParams(ecgs, B_SIZE*LEADS, SAMPLES);
    std::vector<std::variant<int*, double*, std::string*>> insert_arrays {
            euids,
            lead_ids,
            res.res20flat,
            res.resCL,
            res.resHE
    };

    wvfmDB.BatchInsert("wvfm_params", insert_arrays, B_SIZE*LEADS);

    delete[] euids;
    delete[] lead_ids;
}


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

    //const int BATCH_SIZE = 256;
    const int BATCH_SIZE = 2048;

    const int ECG_BUFFER_LEN = LEADS*SAMPLES;

    float * ecgs = new float[ECG_BUFFER_LEN*BATCH_SIZE];

    int completed_total = 0;
    int completed_batch = 0;

    std::vector<std::string> filename_vector;

    std::ofstream unparsable ("unparsable.txt");

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //int init = createArtifactDB();
    SqliteWrapper wvfmDB = SqliteWrapper("database/wvfmWrapped.db");
    std::vector<std::pair<std::string, std::string>> column_desc {
            {"EUID", "TEXT PRIMARY KEY"},
            {"LEAD", "INT"},
            {"NOCHANGE20", "INT"},
            {"CURVELENGTH", "REAL"},
            {"HISTENTROPY", "REAL"}
    };
    wvfmDB.CreateTable("wvfm_params", column_desc);

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
            writeBatch(filename_vector, ecgs, wvfmDB);
            filename_vector.clear();
            completed_batch = 0;
        }
        if (completed_total == MAX_XMLS) break;
    }

    if (!filename_vector.empty()){
        writeBatch(filename_vector, ecgs, wvfmDB);
    }

    delete[] ecgs;
    unparsable.close();
    wvfmDB.CloseDB();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "TIME: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;
    return 0;
}



