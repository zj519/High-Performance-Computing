
#include <iostream>
#include <cmath>
using namespace std;
// Directly calling Fortran API equally valid
#include "cblas.h"
double ComputeNorm(const int n, const double* x) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += x[i]*x[i];
    }
    return sqrt(result);
}
double ComputeSum(const int n, const double* x) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += x[i];
    }
    return result;
}
int ComputeMaxIndex(const int n, const double* x) {
    double max = 0.0;    // Assume only positive numbers in x
    int max_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max) {
            max = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}
int main() {
    // Declare the array
    const int n = 100;
    double* x = new double[n];
    // Seed the random number generate with the current epoch time
    srand(time(0));
    // Generate random numbers and print them to the screen
    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand()/RAND_MAX;
        cout << x[i] << endl;
    }
    // Use BLAS to generate L2-norm
    double norm = cblas_dnrm2(n, x, 1);
    // Since x[i] >= 0, we can use DASUM to compute the sum
    double sum  = cblas_dasum(n, x, 1);
    // Use the BLAS routine IDAMAX to find the index of entry with maximum value
    int    max  = cblas_idamax(n, x, 1);
    // Print out the values
    cout << "Norm:      " << norm << endl;
    cout << "Sum:       " << sum << endl;
    cout << "Max index: " << max << endl;
    cout << "Max value: " << x[max] << endl;
    // Manual verification
    cout << "Verify norm:      " << ComputeNorm(n, x) << endl;
    cout << "Verify sum:       " << ComputeSum(n, x) << endl;
    int man_max = ComputeMaxIndex(n, x);
    cout << "Verify max idx:   " << man_max << endl;
    cout << "Verify max value: " << x[man_max] << endl;
    return 0;
}

