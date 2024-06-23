#include "Import.h"

cv::Mat importImage(const char filePath[]) {
    // reads image and returns it as a matrix
    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);

    // checks for errors
    if (image.empty()) {
        printf("Could not open or find the image\n");
        exit(-1);
    }
    return image;
}

cv::VideoCapture importVideo(const char filePath[]) {
    cv::VideoCapture vCap(filePath);

    // checks for errors
    if (vCap.isOpened() == false) {
        printf("Could not open or find the video\n");
        exit(-1);
    }
    return vCap;
}

cv::VideoCapture importWebcam(int inputIndex) {
    // 0 - deffalut camera
    cv::VideoCapture vCap(inputIndex);

    // checks for errors
    if (vCap.isOpened() == false) {
        printf("Could not open or find the video camera\n");
        exit(-1);
    }
    return vCap;
}
