#ifndef WRITESQL_H
#define WRITESQL_H

#include <vector>
#include <string>

#include "cudaResults.h"

int createArtifactDB();
int writeCudaOutput(cudaResults results, std::vector<std::string> filenames, int BATCH_SIZE,  const int LEADS);

#endif /* WRITESQL_H */
