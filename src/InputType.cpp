#include "InputType.h"

void createTrackbarWindow() {
    char trackbarsWindow[] = "Trackbars";
    cv::namedWindow(trackbarsWindow, cv::WINDOW_NORMAL);

    // creates trackbar to change brightness
    int* brightnessValue = brightnessSingleton();
    const char trackbarName1[] = "Brightness";
    createTrackbar(trackbarName1, trackbarsWindow, *brightnessValue, 100, brightnessCallback);

    // creates trackbar to change contrast 
    int* contrastValue = contrastSingleton();
    const char trackbarName2[] = "Contrast";
    createTrackbar(trackbarName2, trackbarsWindow, *contrastValue, 100, contrastCallback);

    // creates trackbar to change angle
    int* angleValue = angleSingleton();
    const char trackbarName3[] = "Angle";
    createTrackbar(trackbarName3, trackbarsWindow, *angleValue, 360, angleCallback);

    int* scaleValue = scaleSingleton();
    const char trackbarName4[] = "Scale";
    createTrackbar(trackbarName4, trackbarsWindow, *scaleValue, 100, scaleCallback);

    int* borderModeValue = borderModeSingleton();
    const char trackbarName5[] = "BorderMode";
    createTrackbar(trackbarName5, trackbarsWindow, *borderModeValue, 5, borderModeCallback);

    // creates trackbars to change color finding result
    int* lowH = LowHSingleton();
    const char trackbarName6[] = "LowH";
    createTrackbar(trackbarName6, trackbarsWindow, *lowH, 179, lowHCallback);

    int* highH = HighHSingleton();
    const char trackbarName7[] = "HighH";
    createTrackbar(trackbarName7, trackbarsWindow, *highH, 179, HighHCallback);

    int* lowS = LowSSingleton();
    const char trackbarName8[] = "LowS";
    createTrackbar(trackbarName8, trackbarsWindow, *lowS, 255, lowSCallback);

    int* highS = HighSSingleton();
    const char trackbarName9[] = "HighS";
    createTrackbar(trackbarName9, trackbarsWindow, *highS, 255, HighSCallback);

    int* lowV = LowVSingleton();
    const char trackbarName10[] = "LowV";
    createTrackbar(trackbarName10, trackbarsWindow, *lowV, 255, lowVCallback);

    int* highV = HighVSingleton();
    const char trackbarName11[] = "HighV";
    createTrackbar(trackbarName11, trackbarsWindow, *highV, 255, HighVCallback);
}

//IMAGE______________________________________________________________________
void importAndSaveImage(const char* filePath, int funcNumber) {

    char* windowName = windowSingleton();
    // creates window
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat* image = imageSingleton();
    cv::Mat* originalImage = originalImageSingleton();

    *originalImage = importImage(filePath);
    // imports image
    *image = importImage(filePath);

    createTrackbarWindow();

    cv::Mat changedImage;

    cv::Ptr<cv::face::Facemark> facemark;

    if (funcNumber == 3) {
        // Create an instance of Facemark
        facemark = cv::face::FacemarkLBF::create();
        // Load landmark detector
        facemark->loadModel("Content\\models\\lbfmodel.yaml");
    }

    cv::Mat imageResult;

    while (true) {

        if (funcNumber == 1)
            imageResult = docScanner(*image);
        else if (funcNumber == 2)
            imageResult = plateDetection(*image);
        else if (funcNumber == 3)
            imageResult = detectFace(*image, facemark);
        else if (funcNumber == 4)
            imageResult = detectPose(*image);
        else if (funcNumber == 5)
            imageResult = detectHand(*image);
        else if (funcNumber == 6) {
            imageResult = image->clone();
            selectiveSearch(*image);
        }

        cv::imshow(windowName, imageResult);
        colorDetection();
        int k = cv::waitKey(1);
        if (k == 115)
            // saves image
            saveImage(*image);
        if (k == 27)
            break;
    }

    // destroys window
    cv::destroyWindow(windowName);

    image->release();
}

// VIDEO_____________________________________________________________________
void importAndSaveVideo(const char* filePath, int funcNumber) {
    // imports video
    cv::VideoCapture vCap = importVideo(filePath);

    // creates writer
    cv::VideoWriter vidWriter = createVidWriter(vCap);
    videoOperation(vCap, vidWriter, funcNumber);
}

// WEBCAM____________________________________________________________________
void importAndSaveWebcam(const int webcamNumber, int funcNumber) {
    cv::VideoCapture vCap = importWebcam(webcamNumber);

    cv::VideoWriter vidWriter = createVidWriter(vCap);

    videoOperation(vCap, vidWriter, funcNumber);
}

cv::VideoWriter createVidWriter(cv::VideoCapture vCap) {
    // gets the width and height of frames of the video
    int frameWidth = static_cast<int>(vCap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(vCap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // sets size of frame
    cv::Size frameSize(frameWidth, frameHeight);
    // gets fps
    double fps = vCap.get(cv::CAP_PROP_FPS);

    char videoName[200];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(videoName, "Content\\Saved\\SavedVideo%d-%02d-%02d %02d.%02d.%02d.avi", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    cv::VideoWriter vidWriter(videoName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize, true);

    // checks for errors
    if (vidWriter.isOpened() == false) {
        printf("Cannot save the video to a file");
        exit(-1);
    }
    return vidWriter;
}

void videoOperation(cv::VideoCapture vCap, cv::VideoWriter vidWriter, int funcNumber) {
    cv::Mat frame;
    double fps;
    bool frameSuccess;
    cv::Mat imageResult;

    // reads frame
    frameSuccess = vCap.read(frame);

    cv::Mat* image = imageSingleton();
    cv::Mat* originalImage = originalImageSingleton();
    *originalImage = frame.clone();
    // imports image
    *image = frame.clone();

    createTrackbarWindow();

    cv::Ptr<cv::face::Facemark> facemark;

    if (funcNumber == 3) {
        // Create an instance of Facemark
        facemark = cv::face::FacemarkLBF::create();
        // Load landmark detector
        facemark->loadModel("Content\\models\\lbfmodel.yaml");
    }

    char* windowName = windowSingleton();
    // creates window
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // loop through all frames
    while (true) {


        // reads frame
        frameSuccess = vCap.read(frame);

        // checks whether video ended
        if (frameSuccess == false) {
            printf("Video ended\n");
            break;
        }

        *originalImage = frame.clone();
        // imports image
        *image = frame.clone();

        // FPS
        fps = vCap.get(cv::CAP_PROP_FPS);
        printf("FPS: %f\n", fps);

        // writes frame
        vidWriter.write(frame);

        if (funcNumber == 1)
            imageResult = docScanner(*image);
        else if (funcNumber == 2)
            imageResult = plateDetection(*image);
        else if (funcNumber == 3)
            imageResult = detectFace(*image, facemark);
        else if (funcNumber == 4)
            imageResult = detectPose(*image);
        else if (funcNumber == 5)
            imageResult = detectHand(*image);
        else if (funcNumber == 6) {
            imageResult = image->clone();
            selectiveSearch(*image);
        }

        // shows frame
        cv::imshow(windowName, imageResult);

        // checks whether user pressed Esc button(27 ASCII) in 10 ms
        if (cv::waitKey(10) == 27) {
            printf("Esc key is pressed\n");
            break;
        }
    }

    // flushed and closes the video file
    vidWriter.release();
}
