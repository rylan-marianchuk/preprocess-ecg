#ifndef CUDARESULTS_H
#define CUDARESULTS_H

struct cudaResults {
    float * resCL;  // Curve Length
    float * resHE;  // Histogram entropy
    bool * res20flat;  // 0 or 1 whether there is a flat line of length 20 (at least)
};

#endif /* CUDARESULTS_H */
