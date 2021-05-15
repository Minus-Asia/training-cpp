#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    /* Read the image file */
    Mat image = imread("../index.jpg",
                       IMREAD_GRAYSCALE);
  
    /* Error Handling */
    if (image.empty()) {
        cout << "index.jpg "
             << "Not Found" << endl;

        /* wait for any key press */
        cin.get();
        return -1;
    }
  
    // Show Image inside a window
    imshow("Cat", image);
  
    waitKey(0);
    return 0;
}