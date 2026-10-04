#include <iostream>
using namespace std;
// Need this to read and write JPEG files
#include <boost/gil/extension/io/jpeg.hpp>
using namespace boost::gil;
#define F77NAME(x) x##_
extern "C" {
    // Compute singular value decomposition of general matrix
    void F77NAME(dgesvd) (const char& jobu, const char& jobvt, const int& m,
                        const int& n, double* A, const int& lda, double* s,
                        double* U, const int& ldu, double* vt, const int& ldvt,
                        double* work, const int& lwork, int* info);
}
int main() {
    rgb8_image_t img;
    read_image("plane.jpg", img, jpeg_tag());
    const int n = img.width();
    const int m = img.height();
    cout << "Image dimensions: " << n << " x " << m << endl;
    // Three matrices for R,G,B components of image.
    double* r = new double[n*m];
    double* g = new double[n*m];
    double* b = new double[n*m];
    auto imgv = const_view(img);
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < m; ++y) {
            r[x*m+y] = (int)at_c<0>(imgv(x,y));
            g[x*m+y] = (int)at_c<1>(imgv(x,y));
            b[x*m+y] = (int)at_c<2>(imgv(x,y));
        }
    }
    int nsvd = min(m,n);
    double* s = new double[3*nsvd];
    double* u = new double[3*m*m];
    double* v = new double[3*n*n];
    // Perform initial workspace estimate
    double wktmp;
    int lwork = -1;
    int info;
    F77NAME(dgesvd)('A', 'A', m, n, r, m, s, u, m, v, n, &wktmp, lwork, &info);
    // Perform SVD on three components separately
    lwork = (int)(wktmp);
    double* wk = new double[lwork*3];
    F77NAME(dgesvd)('A', 'A', m, n, r, m, s, u, m, v, n, wk, lwork, &info);
    F77NAME(dgesvd)('A', 'A', m, n, g, m, s+nsvd, u+m*m, m, v+n*n, n,
                    wk+lwork, lwork, &info);
    F77NAME(dgesvd)('A', 'A', m, n, b, m, s+2*nsvd, u+2*m*m, m, v+2*n*n, n,
                    wk+2*lwork, lwork, &info);
    // Create compressed image
    rgb8_image_t out(n, m);
    auto outv = view(out);
    const int nvec = 250;
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < m; ++y) {
            double rr = 0;
            double gg = 0;
            double bb = 0;
            for (int k = 0; k < nvec; ++k) {
                rr += u[k*m+y]*v[x*n+k]*s[k];
                gg += u[m*m+k*m+y]*v[n*n+x*n+k]*s[nsvd+k];
                bb += u[2*m*m+k*m+y]*v[2*n*n+x*n+k]*s[2*nsvd+k];
            }
            outv(x,y) = rgb8_pixel_t(rr, gg, bb);
        }
    }
    // Write out compressed image
    write_view("output.jpg", outv, jpeg_tag());
    delete[] r;
    delete[] g;
    delete[] b;
    delete[] s;
    delete[] u;
    delete[] v;
    delete[] wk;
}

