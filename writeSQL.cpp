#include "writeSQL.h"
#include <iostream>
#include <sqlite3.h>

int createArtifactDB()
{
    sqlite3* DB;
    int exit = 0;
    exit = sqlite3_open("ecgArtifact.db", &DB);

    if (exit) {
        std::cerr << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
        return -1;
    }

    std::string create_table = "CREATE TABLE FEATURES( "
                               "ID          TEXT PRIMARY KEY, "
                               "LEAD        INT, "
                               "CURVELENGTH REAL,"
                               "HISTENTROPY REAL );";

    char* messageError;
    exit = sqlite3_exec(DB, create_table.c_str(), NULL, 0, &messageError);

    if (exit != SQLITE_OK) {
        std::cerr << "Error Create Table" << std::endl;
        sqlite3_free(messageError);
    }


    sqlite3_close(DB);
    return 0;
}


int writeCudaOutput(cudaResults results, std::vector<std::string> filenames, int BATCH_SIZE){
    sqlite3* DB;
    int exit = 0;
    exit = sqlite3_open("ecgArtifact.db", &DB);

    if (exit) {
        std::cerr << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
        return -1;
    }

    for (int i = 0; i < BATCH_SIZE; i++){
        for (int lead = 0; lead < 8; lead++){
            std::string row = "INSERT INTO FEATURES VALUES("
                    + filenames[i]                                       + " "
                    + std::to_string(lead)                            + " "
                    + std::to_string(results.resCL[i + lead])          + " "
                    + std::to_string(results.resHE[i + lead])          + ");";

            char* messageError;
            exit = sqlite3_exec(DB, row.c_str(), NULL, 0, &messageError);
            if (exit) {
                std::cerr << "Error inserting into DB " << sqlite3_errmsg(DB) << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Successfully wrote to DB " << sqlite3_errmsg(DB) << std::endl;
    return 0;
}

