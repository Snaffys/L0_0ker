#ifndef INPUTTYPE_H
#define INPUTTYPE_H
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageOperations.h"
#include "SaveToFile.h"
#include "Import.h"


void createTrackbarWindow();

void importAndSaveImage(const char* filePat, int funcNumberh);

void importAndSaveVideo(const char* filePath, int funcNumber);

void importAndSaveWebcam(const int webcamNumber, int funcNumber);



cv::VideoWriter createVidWriter(cv::VideoCapture vCap);
void videoOperation(cv::VideoCapture vCap, cv::VideoWriter vidWriter, int funcNumber);
#endif