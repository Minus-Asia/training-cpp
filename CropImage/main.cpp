#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    Mat img = imread("../1.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "error: image not read from file\n\n";
        return(0);
    }

    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    Mat threshold_output;
    // Detect edges using Threshold
    threshold( img, threshold_output, 0, 255, THRESH_BINARY + THRESH_OTSU);
    imshow( "threshold_output", threshold_output );

    vector<vector<Point> > contours;
    /// Find contours
    cv::findContours( threshold_output, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    Rotate the image
    RotatedRect rect = minAreaRect(contours[0]);

    /* some code to check */
    Mat draw;
    cvtColor(img, draw, COLOR_GRAY2BGR);
    drawContours(draw, contours, 0, (0, 255, 0), 5);
    imshow("contours draw", draw);

    /* End check */

    Mat M, rotated, rotatedImg;
    M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
    warpAffine(img, rotated, M, img.size(), cv::INTER_AREA);
    imshow("rotated", rotated);

    getRectSubPix(rotated, rect.size, rect.center, rotatedImg);
    imshow("rotatedImg", rotatedImg);

    waitKey(0);
    return(0);
}