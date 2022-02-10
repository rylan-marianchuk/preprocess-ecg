#include "writeSQL.h"
#include <iostream>
#include <sqlite3.h>

int createArtifactDB()
{
    sqlite3* DB;
    int exit = 0;
    exit = sqlite3_open("wvfm_params.db", &DB);

    if (exit) {
        std::cerr << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
        return -1;
    }

    std::string create_table = "CREATE TABLE wvfm_params( "
                               "ID          TEXT PRIMARY KEY, "
                               "LEAD        INT, "
                               "NOCHANGE20  INT, "
                               "CURVELENGTH REAL,"
                               "HISTENTROPY REAL,"
                               "SEGAUTOCORR REAL );";

    char* messageError;
    exit = sqlite3_exec(DB, create_table.c_str(), NULL, 0, &messageError);

    if (exit != SQLITE_OK) {
        std::cerr << "Error Create Table" << std::endl;
        sqlite3_free(messageError);
    }


    sqlite3_close(DB);
    return 0;
}


int writeCudaOutput(cudaResults results, std::vector<std::string> filenames, int BATCH_SIZE, const int LEADS){
    sqlite3* DB;
    int exit = 0;
    exit = sqlite3_open("wvfm_params.db", &DB);

    if (exit) {
        std::cerr << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
        return -1;
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int lead = 0; lead < LEADS; lead++){
            std::string K = filenames[i].substr(0, filenames[i].size() - 3) + std::to_string(lead);
            std::string row = "INSERT INTO 'wvfm_params' VALUES('"
                    + K                                                   + "', '"
                    + std::to_string(lead)                             + "', '"
                    + std::to_string(results.res20flat[(i*LEADS) + lead])      + "', '"
                    + std::to_string(results.resCL[(i*LEADS) + lead])          + "', '"
                    + std::to_string(results.resHE[(i*LEADS) + lead])          + "', 'NULL');";

            //std::cout << "Attempting to write " << row << std::endl;

            char* messageError;
            exit = sqlite3_exec(DB, row.c_str(), NULL, 0, &messageError);
            if (exit) {
                std::cerr << "Error inserting into DB " << sqlite3_errmsg(DB) << std::endl;
                return -1;
            }
            //std::cout << "Successfully wrote " << filenames[i] << std::endl;
        }
    }

    std::cout << "Successfully wrote batch to DB " << sqlite3_errmsg(DB) << std::endl;
    return 0;
}

