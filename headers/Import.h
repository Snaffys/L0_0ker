#ifndef IMPORT_H
#define IMPORT_H

#include <opencv2/highgui.hpp>

cv::Mat importImage(const char filePath[]);
cv::VideoCapture importVideo(const char filePath[]);
cv::VideoCapture importWebcam(int inputIndex);

#endif
