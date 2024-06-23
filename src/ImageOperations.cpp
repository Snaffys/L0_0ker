#include "ImageOperations.h"

// SINGLETONS__________________________________________________________________________________________

cv::Mat* imageSingleton() {
    static cv::Mat image;
    return &image;
};

cv::Mat* originalImageSingleton() {
    static cv::Mat image;
    return &image;
};

char* windowSingleton() {
    static char windowName[20] = "Image";
    return windowName;
};

int* brightnessSingleton() {
    static int brightness = 50;
    return &brightness;
};

int* contrastSingleton() {
    static int contrast = 50;
    return &contrast;
};

int* angleSingleton() {
    static int angle = 180;
    return &angle;
};

int* scaleSingleton() {
    static int scale = 50;
    return &scale;
};

int* borderModeSingleton() {
    static int borderMode = 0;
    return &borderMode;
};

int* LowHSingleton() {
    static int LowH = 0;
    return &LowH;
};

int* LowSSingleton() {
    static int LowS = 0;
    return &LowS;
};

int* LowVSingleton() {
    static int LowV = 0;
    return &LowV;
};

int* HighHSingleton() {
    static int HighH = 179;
    return &HighH;
};

int* HighSSingleton() {
    static int HighS = 255;
    return &HighS;
};

int* HighVSingleton() {
    static int HighV = 255;
    return &HighV;
};


void createTrackbar(const char trackbarName[], const char windowName[], int selectedValue, int maxValue, void(*f)(int, void*)) {
    cv::createTrackbar(trackbarName, windowName, nullptr, maxValue, f);
    cv::setTrackbarPos(trackbarName, windowName, selectedValue);
}


// CALLBACKS____________________________________________________________________________________

void brightnessCallback(int valueForBrightness, void* userData) {
    int* brightnessValue = brightnessSingleton();
    *brightnessValue = valueForBrightness;
    changeImage();
}

void contrastCallback(int valueForContrast, void* userData) {
    int* contrastValue = contrastSingleton();
    *contrastValue = valueForContrast;
    changeImage();
}

void angleCallback(int angleValue, void* userData) {
    int* angleValueSingleton = angleSingleton();
    *angleValueSingleton = angleValue;
    changeImage();
}

void scaleCallback(int scaleValue, void* userData) {
    int* scaleValueSingleton = scaleSingleton();
    *scaleValueSingleton = scaleValue;
    changeImage();
}

void borderModeCallback(int borderModeValue, void* userData) {
    int* borderModeValueSingleton = borderModeSingleton();
    *borderModeValueSingleton = borderModeValue;
    changeImage();
}

cv::Mat rotateImage(cv::Mat changeableImage) {

    int imageHeightCenter = changeableImage.rows / 2;
    int imageWidthCenter = changeableImage.cols / 2;

    int* angleValue = angleSingleton();
    int* scale = scaleSingleton();
    int* borderMode = borderModeSingleton();

    // creates affine transformation matrix
    cv::Mat matRotation = cv::getRotationMatrix2D(cv::Point(imageWidthCenter, imageHeightCenter), *angleValue - 180, *scale / 50.0);
    cv::Mat rotatedImage;
    // applies affine transformation to an image
    cv::warpAffine(changeableImage, rotatedImage, matRotation, changeableImage.size(), cv::INTER_LINEAR, *borderMode, cv::Scalar());
    return rotatedImage;
}

void changeImage() {
    cv::Mat* originalImage = originalImageSingleton();

    cv::Mat* image = imageSingleton();

    originalImage->copyTo(*image);

    int* contrastValue = contrastSingleton();
    int* brightnessValue = brightnessSingleton();

    image->convertTo(*image, -1, *contrastValue / 50.0, *brightnessValue - 50);

    char* windowName = windowSingleton();

    cv::Mat rotatedImage = rotateImage(*image);

    rotatedImage.copyTo(*image);
}

void lowHCallback(int lowH, void* userData) {
    int* lowHValueSingleton = LowHSingleton();
    *lowHValueSingleton = lowH;
};

void lowSCallback(int lowS, void* userData) {
    int* lowSValueSingleton = LowSSingleton();
    *lowSValueSingleton = lowS;
};

void lowVCallback(int lowV, void* userData) {
    int* lowVValueSingleton = LowVSingleton();
    *lowVValueSingleton = lowV;
};

void HighHCallback(int HighH, void* userData) {
    int* HighHValueSingleton = HighHSingleton();
    *HighHValueSingleton = HighH;
};

void HighSCallback(int HighS, void* userData) {
    int* HighSValueSingleton = HighSSingleton();
    *HighSValueSingleton = HighS;
};

void HighVCallback(int HighV, void* userData) {
    int* HighVValueSingleton = HighVSingleton();
    *HighVValueSingleton = HighV;
};

void colorDetection() {
    cv::Mat imageHSV;
    cv::Mat* image = imageSingleton();

    // size of windows with color detection
    int newHeight = 400;
    int newWidth = image->cols * newHeight / image->rows;

    // creates windows with color detection
    char coloredWindowName[] = "Color detection";
    char maskWindowName[] = "Color detection mask";
    cv::namedWindow(coloredWindowName, cv::WINDOW_NORMAL);
    cv::namedWindow(maskWindowName, cv::WINDOW_NORMAL);

    // gets trackbar values
    int* lowH = LowHSingleton();
    int* lowS = LowSSingleton();
    int* lowV = LowVSingleton();
    int* highH = HighHSingleton();
    int* highS = HighSSingleton();
    int* highV = HighVSingleton();

    // converts BGR changeableImage to HSV
    cv::cvtColor(*image, imageHSV, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    // checks whether changeableImage between lowHSV and highHSV (if so - 255, otherwise - 0) 
    cv::inRange(imageHSV, cv::Scalar(*lowH, *lowS, *lowV), cv::Scalar(*highH, *highS, *highV), mask);

    cv::Mat coloredImage;
    // compares bits from a mask and bits from an changeableImage (if equals - keep pixel, black pixel otherwise)
    cv::bitwise_and(*image, *image, coloredImage, mask = mask);

    // shows colored image
    cv::resizeWindow(coloredWindowName, newWidth, newHeight);
    cv::imshow(coloredWindowName, coloredImage);
    cv::moveWindow(coloredWindowName, 0, 0);

    // makes shapes smoother
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    // shows mask
    cv::resizeWindow(maskWindowName, newWidth, newHeight);
    cv::imshow(maskWindowName, mask);
    cv::moveWindow(maskWindowName, 0, newHeight);
}


// FACE DETECTION_________________________________________________________________________________________________________________________________________

cv::Mat detectFace(cv::Mat detectableImage, cv::Ptr<cv::face::Facemark> facemark) {

    cv::Mat image = detectableImage.clone();

    // loads face detector
    cv::CascadeClassifier faceDetector("Content\\xmlDetection\\haarcascade_frontalface_alt2.xml");

    // grayscale image
    cv::Mat grayImage;

    // vector of faces
    std::vector<cv::Rect> faces;

    // faceDetector requires grayscale image
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // detects faces
    faceDetector.detectMultiScale(grayImage, faces, 1.2, 3,  0, cv::Size(50,50));

    // vector of vector of landmarks(points), because we can have multiple faces in the image
    std::vector< std::vector<cv::Point2f> > landmarks;

    // runs landmark detector
    bool success = facemark->fit(image, faces, landmarks);

    if (success)
    {
        // if successful, render the landmarks on the face
        for (int i = 0; i < landmarks.size(); i++)
            drawLandmarks(image, landmarks[i]);
    }

    return image;
}

void drawPolyline(cv::Mat& image, const std::vector<cv::Point2f>& landmarks, const int start, const int end, bool isClosed = false)
{
    // gathers all points between the start and end indices
    std::vector <cv::Point> points;
    for (int i = start; i <= end; i++)
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    // draws polylines 
    cv::polylines(image, points, isClosed, cv::Scalar(255, 200, 0), 2, 16);

}

void drawLandmarks(cv::Mat& image, std::vector<cv::Point2f>& landmarks)
{
    // draws face for the 68-point model
    if (landmarks.size() == 68)
    {
        drawPolyline(image, landmarks, 0, 16);           // Jaw line
        drawPolyline(image, landmarks, 17, 21);          // Left eyebrow
        drawPolyline(image, landmarks, 22, 26);          // Right eyebrow
        drawPolyline(image, landmarks, 27, 30);          // Nose bridge
        drawPolyline(image, landmarks, 30, 35, true);    // Lower nose
        drawPolyline(image, landmarks, 36, 41, true);    // Left eye
        drawPolyline(image, landmarks, 42, 47, true);    // Right Eye
        drawPolyline(image, landmarks, 48, 59, true);    // Outer lip
        drawPolyline(image, landmarks, 60, 67, true);    // Inner lip
    }
    else // if the number of points is not 68, we do not know which points correspond to which facial features 
    { 
      // draws 1 dot per landmark
        for (int i = 0; i < landmarks.size(); i++)
            cv::circle(image, landmarks[i], 3, cv::Scalar(255, 200, 0), cv::FILLED);
    }
}


// POSE DETECTOIN_________________________________________________________________________________________________________________________________________

cv::Mat detectPose(cv::Mat detectableImage) {

    cv::Mat image = detectableImage.clone();

    const int POSE_PAIRS[14][2] =
    {
        {0,1},                      // head - neck
        {1,2}, {2,3}, {3,4},        // right arm
        {1,5}, {5,6}, {6,7},        // left arm
        {1,14},                     // neck - body
        {14,8}, {8,9}, {9,10},      // body - left leg
        {14,11}, {11,12}, {12,13}   // body - right leg
    };

    // file with text description of the network architecture
    char protoFile[] = "Content\\poseDetection\\pose_deploy_linevec_faster_4_stages.prototxt";
    // file with learned network
    char weightsFile[] = "Content\\poseDetection\\pose_iter_160000.caffemodel";

    // reads a network model
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

    // converts image from opencv format to Caffe blob format
    cv::Mat inpBlob = cv::dnn::blobFromImage(image, 1.0 / 255, cv::Size(229, 229), cv::Scalar(0, 0, 0), false, false);

    // sets the new input value for the network
    net.setInput(inpBlob);

    cv::Mat output = net.forward();

    int H = output.size[2];
    int W = output.size[3];

    // vector of pose landmarks
    std::vector<cv::Point> points(15);

    for (int n = 0; n < 15; n++)
    {
        // Probability map of corresponding body's part.
        cv::Mat probMap(H, W, CV_32F, output.ptr(0, n));

        cv::Point2f p(-1, -1);
        cv::Point maxLoc;
        double prob;

        cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        if (prob > 0.1)
        {
            p = maxLoc;
            p.x *= (float)image.cols / W;
            p.y *= (float)image.rows / H;
        }
        points[n] = p;
    }

    int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected body/hand parts
        cv::Point2f partA = points[POSE_PAIRS[n][0]];
        cv::Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;

        // draws landmarks and lines between them
        cv::line(image, partA, partB, cv::Scalar(0, 255, 255), 8);
        cv::circle(image, partA, 8, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, partB, 8, cv::Scalar(0, 0, 255), -1);
    }

    return image;
}


// HAND DETECTION_________________________________________________________________________________________________________________________________________

cv::Mat detectHand(cv::Mat detectableImage) {
    cv::Mat image = detectableImage.clone();
    const int HAND_PAIRS[20][2] =
    {
        {0,1}, {1,2}, {2,3}, {3,4},         // thumb
        {0,5}, {5,6}, {6,7}, {7,8},         // index
        {0,9}, {9,10}, {10,11}, {11,12},    // middle
        {0,13}, {13,14}, {14,15}, {15,16},  // ring
        {0,17}, {17,18}, {18,19}, {19,20}   // small
    };

    float aspect_ratio = image.cols / (float)image.rows;
    int inHeight = 368;
    int inWidth = (int(aspect_ratio * inHeight) * 8) / 8;

    // file with text description of the network architecture
    char protoFile[] = "Content\\handDetection\\pose_deploy.prototxt";
    // file with learned network
    char weightsFile[] = "Content\\handDetection\\pose_iter_102000.caffemodel";

    // hand points
    int nPoints = 22;

    // reads a network model
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

    // converts image from opencv format to Caffe blob format
    cv::Mat inpBlob = cv::dnn::blobFromImage(image, 1.0 / 255, cv::Size(inWidth, inHeight), cv::Scalar(0, 0, 0), false, false);

    net.setInput(inpBlob);

    cv::Mat output = net.forward();

    float thresh = 0.01;

    int H = output.size[2];
    int W = output.size[3];

    // find the position of the body parts
    std::vector<cv::Point> points(nPoints);
    for (int n = 0; n < nPoints; n++)
    {
        // Probability map of corresponding body's part.
        cv::Mat probMap(H, W, CV_32F, output.ptr(0, n));
        cv::resize(probMap, probMap, cv::Size(image.cols, image.rows));

        cv::Point maxLoc;
        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        points[n] = maxLoc;
    }

    int nPairs = sizeof(HAND_PAIRS) / sizeof(HAND_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected hand parts
        cv::Point2f partA = points[HAND_PAIRS[n][0]];
        cv::Point2f partB = points[HAND_PAIRS[n][1]];

        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;

        line(image, partA, partB, cv::Scalar(0, 255, 255), 8);
        circle(image, partA, 8, cv::Scalar(0, 0, 255), -1);
        circle(image, partB, 8, cv::Scalar(0, 0, 255), -1);
    }

    return image;

    //int newWidth = 400;
    //int newHeight = image.rows * newWidth / image.cols;
    //char handWindowName[] = "Hand detection";
    //cv::namedWindow(handWindowName, cv::WINDOW_NORMAL);

    //cv::resizeWindow(handWindowName, newWidth, newHeight);
    //cv::imshow(handWindowName, image);
    //cv::moveWindow(handWindowName, 0, newHeight);
}


// DOCUMENTS SCANNER_________________________________________________________________________________________________________________________

cv::Mat docScanner(cv::Mat image) {
    cv::Mat tempImage;

    // makes image more readable
    cv::cvtColor(image, tempImage, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(tempImage, tempImage, cv::Size(3, 3), 3, 0);
    cv::Canny(tempImage, tempImage, 25, 75);
    cv::dilate(tempImage, tempImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<cv::Point> initialPoints = getContours(tempImage, tempImage, 1, cv::Scalar(0,220,0), 2);

    if (initialPoints.size() == 4) {

        std::vector<cv::Point> docPoints = reorderPoints(initialPoints);

        cv::Mat imageWarp = getWarp(image, docPoints, image.cols, image.rows);

        int cropValue = 5;
        cv::Rect crop(cropValue, cropValue, imageWarp.cols - (2 * cropValue), imageWarp.rows - (2 * cropValue));

        cv::Mat imageCrop = imageWarp(crop);

        int newWidth = 400;
        int newHeight = image.rows * newWidth / image.cols;
        char docWindowName[] = "Scanned document";
        cv::namedWindow(docWindowName, cv::WINDOW_NORMAL);

        cv::resizeWindow(docWindowName, newWidth, newHeight);
        cv::imshow(docWindowName, imageCrop);
        cv::moveWindow(docWindowName, 0, newHeight);
    }

    return tempImage;
}

std::vector<cv::Point> getContours(cv::Mat tempImage, cv::Mat changeableImage, int contourId, cv::Scalar color, int thickness) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // finds contours
    cv::findContours(tempImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

    std::vector<std::vector<cv::Point>> contourPoly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());

    std::vector<cv::Point> biggestContour;
    int maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        int area = cv::contourArea(contours[i]);

        if (area > 1000) {
            float perimeter = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contourPoly[i], 0.0135 * perimeter, true);

            if (area > maxArea && contourPoly[i].size() == 4) {
                cv::drawContours(changeableImage, contourPoly, i, color, thickness);
                biggestContour = { contourPoly[i][0],contourPoly[i][1], contourPoly[i][2], contourPoly[i][3] };
                maxArea = area;
            }
        }
    }
    return biggestContour;
}

std::vector<cv::Point> reorderPoints(std::vector<cv::Point> points) {
    std::vector<cv::Point> newPoints;
    std::vector<int> sumPoints, subPoints;;

    for (int i = 0; i < 4; i++) {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    newPoints.push_back(points[std::min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    newPoints.push_back(points[std::max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[std::min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[std::max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    return newPoints;
}

cv::Mat getWarp(cv::Mat image, std::vector<cv::Point> points, float w, float h) {
    cv::Mat imageWarp;

    cv::Point2f src[4] = { points[0], points[1], points[2], points[3] };
    cv::Point2f dst[4] = { {0.0f,0.0f}, {w, 0.0f}, {0.0f,h}, { w,h } };

    cv::Mat matrix = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(image, imageWarp, matrix, cv::Point(w, h));
    return imageWarp;
}

// PLATE DETECTION_______________________________________________________________________________________________________

cv::Mat plateDetection(cv::Mat image) {

    char imageName[200];

    cv::Mat tempImage;
    image.copyTo(tempImage);
    
    cv::CascadeClassifier plateCascade;
    plateCascade.load("Content\\xmlDetection\\haarcascade_russian_plate_number.xml");
    // checks for errors
    if (plateCascade.empty())
        printf("XML file not loaded");

    // vector of plates
    std::vector<cv::Rect> plates;

    plateCascade.detectMultiScale(tempImage, plates, 1.2, 3, 0, cv::Size(50, 50));

    for (int i = 0; i < plates.size(); i++) {
        cv::Mat imageCrop = tempImage(plates[i]);
        
        // current time
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        sprintf(imageName, "Content\\Saved\\Plates\\%d) %d-%02d-%02d %02d.%02d.%02d.png", i, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

        cv::imwrite(imageName, imageCrop);
        cv::rectangle(image, plates[i].tl(), plates[i].br(), cv::Scalar(0, 255, 0), 3);
    }

    return image;
}


// SELECTIVE SEARCH______________________________________________________________________________________________________

void selectiveSearch(cv::Mat image) {
    int newHeight = 500;
    int newWidth = image.cols * newHeight / image.rows;

    cv::resize(image, image, cv::Size(newWidth, newHeight));
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    ss->setBaseImage(image);
    ss->switchToSelectiveSearchFast();
    std::vector<cv::Rect> rects;
    ss->process(rects);
    int numShowRects = 90;
    cv::Mat imOut = image.clone();

    for (int i = 0; i < rects.size(); i++) {

        if (i < numShowRects) 
            if(rects[i].area() > 0.5 * image.rows * image.cols || rects[i].area() < image.rows * image.cols)
                cv::rectangle(imOut, rects[i], cv::Scalar(0, 255, 0));
        else
            break;
    }

    char selectiveSearchWindowName[] = "Selective search";
    cv::namedWindow(selectiveSearchWindowName, cv::WINDOW_NORMAL);
    cv::imshow(selectiveSearchWindowName, imOut);
    cv::moveWindow(selectiveSearchWindowName, 0, 0);
}


// ADDITION____________________________________________________________________________________________

cv::Mat* inpaintMaskSingleton() {
    cv::Mat* image = imageSingleton();
    static cv::Mat inpaintMask = cv::Mat::zeros(image->size(), CV_8U);
    return &inpaintMask;
};

cv::Mat imageInpaint(cv::Mat changeableImage) {
    cv::Mat imageMask = changeableImage.clone();

    cv::Mat inpaintMask;
    cv::Canny(changeableImage, inpaintMask, 25, 75);

    cv::inpaint(changeableImage, inpaintMask, changeableImage, 3, cv::INPAINT_TELEA);

    return imageMask;
}

void onMouse(int event, int x, int y, int flags, void*)
{
    cv::Point prevPt(-1, -1);
    if (event == cv::EVENT_LBUTTONUP || !(flags & cv::EVENT_FLAG_LBUTTON))
        prevPt = cv::Point(-1, -1);
    else if (event == cv::EVENT_LBUTTONDOWN)
        prevPt = cv::Point(x, y);
    else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
    {
        cv::Mat* image = imageSingleton();
        cv::Point pt(x, y);
        cv::Mat* inpaintMask = inpaintMaskSingleton();
        if (prevPt.x < 0)
            prevPt = pt;
        line(*inpaintMask, prevPt, pt, cv::Scalar::all(255), 5, 8, 0);
        line(*image, prevPt, pt, cv::Scalar::all(255), 5, 8, 0);
        prevPt = pt;
        cv::imshow("changeableImage", *image);
        cv::imshow("changeableImage: mask", *inpaintMask);
    }
}

cv::Mat shapeDetection(cv::Mat changeableImage) {
    cv::Mat tempImage;
    cv::cvtColor(changeableImage, tempImage, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(tempImage, tempImage, cv::Size(3, 3), 3, 0);
    cv::Canny(tempImage, tempImage, 25, 75);
    cv::dilate(tempImage, tempImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    tempImage = getContoursShapes(tempImage, changeableImage, -1, cv::Scalar(0, 0, 212), 1);

    return changeableImage;
}

cv::Mat getContoursShapes(cv::Mat tempImage, cv::Mat changeableImage, int contourId, cv::Scalar color, int thickness) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(tempImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

    std::vector<std::vector<cv::Point>> contourPoly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    char objectType[50];

    for (int i = 0; i < contours.size(); i++) {
        int area = cv::contourArea(contours[i]);

        if (area > 1000) {
            float perimeter = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contourPoly[i], 0.0135 * perimeter, true);

            boundRect[i] = cv::boundingRect(contourPoly[i]);
            //cv::rectangle(changeableImage, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(200, 150, 210), 2);
            int objCorners = (int)contourPoly[i].size();

            if (objCorners == 3)
                strcpy_s(objectType, "Triangle");
            else if (objCorners == 4) {

                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                if (aspRatio > 0.95 && aspRatio < 1.05)
                    strcpy_s(objectType, "Square");
                else
                    strcpy_s(objectType, "Rectangle");
            }
            else if (objCorners == 5)
                strcpy_s(objectType, "Pentagon");
            else if (objCorners == 6)
                strcpy_s(objectType, "Hexagon");
            else if (objCorners == 7)
                strcpy_s(objectType, "Heptagon");
            else if (objCorners >= 8)
                strcpy_s(objectType, "Circle");
            else
                strcpy_s(objectType, "Other");

            cv::drawContours(changeableImage, contourPoly, i, color, thickness);

            cv::putText(changeableImage, objectType, { boundRect[i].x, boundRect[i].y - 5 }, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
        }
    }
    return changeableImage;
}