#include "SaveToFile.h"

void saveImage(cv::Mat image) {
    // writes the image to a file
    char imageName[200];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(imageName, "Content\\Saved\\SavedImage%d-%02d-%02d %02d.%02d.%02d.jpg", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    bool writeSuccess = cv::imwrite(imageName, image);

    // checks for errors
    if (writeSuccess == false) {
        printf("Failed to save the image\n");
        exit(-1);
    }
    printf("Image is succesfully saved to a file\n");
}

void saveImage(cv::Mat image, const char filePath[]) {
    // writes the image to a file
    bool writeSuccess = cv::imwrite(filePath, image);

    // checks for errors
    if (writeSuccess == false) {
        printf("Failed to save the image\n");
        exit(-1);
    }
    printf("Image is succesfully saved to a file\n");
}
