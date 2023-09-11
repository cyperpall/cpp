#include <iostream>
#include <vector>



void reset(int*& image, int*& imgage_original, int& width, int& height);



void gogray(int*& color, int*& gray, int& width, int& height);





void rotateImage_90L(int*& image, int& width, int& height);




void rotateImage_90R(int*& image, int& width, int& height);




void Thresh(int* image, int& width, int& height, int threshold);




void Thres_IVT(int* image, int& width, int& height, int threshold);




void bgr_hsv(int*& img_color,double*& image_hsv , int width, int height);




void blur_aver(double* image, int width, int height, int kernelSize);




void blur_gausian(double* image, int width, int height);





void sobel(double* image, int& width, int& height);




void canny(double* img, int& width, int& height);


void sharpenImage(double* image,int width,int height);

void computeGradient(double* inputImage, double* gradient, int* direction, int width, int height);


void nonMaximumSuppression(double* gradient, int* direction, double* suppressed, int width, int height);


void doubleThresholding(double* suppressed, double* edges, int width, int height, int lowThreshold, int highThreshold);




void edgeTracking(double* edges, double* output, int width, int height);



void erosion(int* image, int width, int height);


void delitation(int* im, int width, int height); 



void erosion_double(double* image, int width, int height);



struct Point
{
    int x;
    int y;
};

struct Stat
{
    int x;
    int y;
    int width;
    int height;
    int area;
};

void rectangle(int* image, int x, int y, int width, int height, int imageWidth);

void dfs(int x, int y, int label, int* image, int width, int height, Point* points, int& pointIndex);

void connectedComponentsWithStats(int* image, int width, int height, std::vector<std::vector<Point>>& components, std::vector<Stat>& stats);