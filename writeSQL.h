#include "cudaResults.h"
#include <vector>
#include <string>

int createArtifactDB();
int writeCudaOutput(cudaResults results, std::vector<std::string> filenames, int BATCH_SIZE);
