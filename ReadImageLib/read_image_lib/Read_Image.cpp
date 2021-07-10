#include <iostream>
#include "Read_Image.h"


using namespace std;
using namespace cv;

void ReadImage::Read_Image() {
        /* Read the image file */
    Mat image = imread("../index.jpg",
                       IMREAD_GRAYSCALE);
    /* Error Handling */
    if (image.empty()) {
        cout << "index.jpg "
             << "Not Found" << endl;

        /* wait for any key press */
        cin.get();
        return;
    }

    // Show Image inside a window
    imshow("Cat", image);
    waitKey();
    return;
}