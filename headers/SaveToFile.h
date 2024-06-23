#ifndef SAVETOFILE_H
#define SAVETOFILE_H

#include <opencv2/highgui.hpp>

void saveImage(cv::Mat image);
void saveImage(cv::Mat image, const char filePath[]);

#endif
