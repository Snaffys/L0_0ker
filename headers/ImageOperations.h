#pragma once
#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facemarkLBF.hpp>
#include <opencv2/xphoto/inpainting.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include <time.h>

void selectiveSearch(cv::Mat image);

void createTrackbar(const char trackbarName[], const char windowName[], int selectedValue, int maxValue, void(*f)(int, void*));

// SINGLETONS________________________________________________________________________________________________________

cv::Mat* originalImageSingleton();
cv::Mat* imageSingleton();
char* windowSingleton();
int* brightnessSingleton();
int* contrastSingleton();
int* angleSingleton();
int* scaleSingleton();
int* borderModeSingleton();
int* LowHSingleton();
int* LowSSingleton();
int* LowVSingleton();
int* HighHSingleton();
int* HighSSingleton();
int* HighVSingleton();

// DETECTION__________________________________________________________________________________________________________

void drawPolyline(cv::Mat& im, const std::vector<cv::Point2f>& landmarks, const int start, const int end, bool isClosed);
void drawLandmarks(cv::Mat& im, std::vector<cv::Point2f>& landmarks);
cv::Mat detectFace(cv::Mat detectableImage, cv::Ptr<cv::face::Facemark> facemark);

cv::Mat detectPose(cv::Mat detectableImage);

cv::Mat detectHand(cv::Mat detectableImage);

cv::Mat plateDetection(cv::Mat changeableImage);

std::vector<cv::Point> getContours(cv::Mat tempImage, cv::Mat changeableImage, int contourId, cv::Scalar color, int thickness);
cv::Mat getWarp(cv::Mat image, std::vector<cv::Point> points, float w, float h);
std::vector<cv::Point> reorderPoints(std::vector<cv::Point> points);
cv::Mat docScanner(cv::Mat changeableImage);

// CALLBACKS__________________________________________________________________________________________________________

void contrastCallback(int valueForContrast, void* userData);
void brightnessCallback(int valueForBrightness, void* userData);
void borderModeCallback(int borderModeValue, void* userData);
void scaleCallback(int scaleValue, void* userData);
void angleCallback(int angleValue, void* userData);
cv::Mat rotateImage(cv::Mat changeableImage);
void changeImage();

void lowHCallback(int lowH, void* userData);
void lowSCallback(int lowH, void* userData);
void lowVCallback(int lowH, void* userData);
void HighHCallback(int lowH, void* userData);
void HighSCallback(int lowH, void* userData);
void HighVCallback(int lowH, void* userData);
void colorDetection();

// ADDITION___________________________________________________________________________________________________________

void onMouse(int event, int x, int y, int flags, void*);
cv::Mat imageInpaint(cv::Mat changeableImage);
cv::Mat* inpaintMaskSingleton();
cv::Mat shapeDetection(cv::Mat changeableImage);
cv::Mat getContoursShapes(cv::Mat tempImage, cv::Mat changeableImage, int contourId, cv::Scalar color, int thickness);

#endif