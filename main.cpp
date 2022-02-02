#include <iostream>
#include "pugixml.hpp"
#include "base64.h"
#include <string>
#include <filesystem>
#include "writeh5.h"

namespace fs = std::filesystem;

#define LEADS 8
#define SAMPLES 5000

int main() {
    writeh5();
    std::cout << "Wrote h5\n";

    std::string path = "/home/rylan/May_2019_XML/";
    double S = 0;
    int total = 0;
    for (const auto & entry : fs::directory_iterator(path)){

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(entry.path().string().c_str());

        if (result){
            // Parsed without error
            pugi::xml_node wvfm = doc.child("RestingECG").child("Waveform").next_sibling();
            std::string b64encoded = wvfm.child("LeadData").child("WaveFormData").child_value();
            if (b64encoded.empty())
                continue;
            b64encoded.erase(b64encoded.size() - 11);
            std::string DE = base64_decode(b64encoded, true);
            const char * c = DE.c_str();

            float * a = (float *) malloc((DE.size() / 2)*sizeof(float));

            for (size_t i = 0; i < DE.size(); i += 2) {
                size_t ind = i / 2;
                a[ind] = (float) *(int16_t *) &c[i];
            }

            for (size_t i = 0; i < DE.size() / 2; i++){
                std::cout << a[i] << "  ";
            }

        }
        else{
            // Error in parsing
            std::cout << "Error in parsing\n";
            continue;
        }
        break;
    }

    return 0;
}
