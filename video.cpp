#include <sstream>
#include <string>
#include <thread>
#include <iostream>
#include <mutex>
#include <mutex>
#include <mutex>
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include<math.h>

using namespace std;
using namespace cv;

/***
    Global vars
***/
std::mutex m;
int flag = 0;
shared_ptr<thread> t1;

//x and y values for the location of the object
int xb = 0, yb = 0;
int xd = 0, yd = 0;
int xo = 0, yo = 0;

//some boolean variables for different functionality within this
//program
bool trackObjects = true;
bool useMorphOps = true;

//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

int sockfd, portno, n;
struct sockaddr_in serv_addr;
struct hostent *server;

void configureSocket();
//
//void sendCommand(unsigned char c);

unsigned char buffer;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "HSV Image";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";


void error(const char *msg) {
    perror(msg);
    exit(0);
}


void on_mouse(int e, int x, int y, int d, void *ptr) {
    if (e == EVENT_LBUTTONDOWN) {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
}

void on_trackbar(int, void *) {//This function gets called whenever a
    // trackbar position is changed
}

string intToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void createTrackbars() {
    //create window for trackbars


    namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf(TrackbarName, "H_MIN", H_MIN);
    sprintf(TrackbarName, "H_MAX", H_MAX);
    sprintf(TrackbarName, "S_MIN", S_MIN);
    sprintf(TrackbarName, "S_MAX", S_MAX);
    sprintf(TrackbarName, "V_MIN", V_MIN);
    sprintf(TrackbarName, "V_MAX", V_MAX);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH),
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->
    createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
    createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
    createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
    createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
    createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
    createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}

void drawObject(int x, int y, Mat &frame) {

    //use some of the openCV drawing functions to draw crosshairs
    //on your tracked image!

    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

    circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
    if (y - 25 > 0)
        line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
    else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
    if (y + 25 < FRAME_HEIGHT)
        line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
    else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
    if (x - 25 > 0)
        line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
    else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
    if (x + 25 < FRAME_WIDTH)
        line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
    else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

    putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
    //cout << "x,y: " << x << ", " << y;

}

void morphOps(Mat &thresh) {

    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a 3px by 3px rectangle

    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

    erode(thresh, thresh, erodeElement);
    erode(thresh, thresh, erodeElement);


    dilate(thresh, thresh, dilateElement);
    dilate(thresh, thresh, dilateElement);


}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

    Mat temp;
    threshold.copyTo(temp);
    //these two vectors needed for output of findContours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    //use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if (numObjects < MAX_NUM_OBJECTS) {
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                Moments moment = moments((cv::Mat) contours[index]);
                double area = moment.m00;

                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                if (area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area > refArea) {
                    x = moment.m10 / area;
                    y = moment.m01 / area;
                    objectFound = true;
                    refArea = area;
                } else objectFound = false;


            }
            //let user know you found an object
            if (objectFound == true) {
                putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
                //draw object location on screen
                //cout << x << "," << y;
                drawObject(x, y, cameraFeed);

            }
        } else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
    }
}

//void sendCommand() {
//    printf("Please enter the message: ");
//    buffer = '\0';
//    fscanf(stdin, "%c", &buffer);
//    cout << buffer;
//    n = write(sockfd, &buffer, 1);
//    if (n < 0)
//        error("ERROR writing to socket");
//}

void thread_task(unsigned char c, int delay) {

    m.lock();
    flag = 1;
    //cout << "Sent:" << c;
    n = write(sockfd, &c, 1);
    if (n < 0)
        error("ERROR writing to socket");
    usleep(delay * 1000);
    c = 's';
    n = write(sockfd, &c, 1);
//    usleep(100 * 1000);
    flag = 0;
    m.unlock();
}

void sendCommand(unsigned char c, int delay) {
    if (flag == 0) {
        t1.reset(new std::thread([=] { thread_task(c, delay); }));
        (*t1).detach();
    }
//    } else {
//        cout << "thread in execution";
//    }
}


void configureSocket() {
    portno = 20231;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    server = gethostbyname("193.226.12.217");

    serv_addr.sin_family = AF_INET;
    bcopy((char *) server->h_addr, (char *) &serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR connecting");
}


void setCoordinates_RED(int &x, int &y, Mat &cameraFeed, Mat HSV, Mat threshold_red) {
    cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
    inRange(HSV, Scalar(0, 185, 197), Scalar(58, 256, 256), threshold_red);
    if (useMorphOps)
        morphOps(threshold_red);
    if (trackObjects) {
        trackFilteredObject(x, y, threshold_red, cameraFeed);
    }
    //cout << "x red: " << x << " " << "y red" << y << "\n";
}

void setCoordinates_BLUE(int &x, int &y, Mat &cameraFeed, Mat HSV, Mat &threshold_blue) {
    cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
    inRange(HSV, Scalar(87, 39, 17), Scalar(107, 255, 256), threshold_blue);
//    inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold_blue);
    if (useMorphOps)
        morphOps(threshold_blue);
    if (trackObjects) {
        trackFilteredObject(x, y, threshold_blue, cameraFeed);
    }
    cout << "x blue: " << x << " " << "y blue " << y << "\n";
}

void setCoordinates_YELLOW(int &x, int &y, Mat &cameraFeed, Mat HSV, Mat &threshold_yellow) {
    cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
    inRange(HSV, Scalar(0, 98, 204), Scalar(83, 152, 256), threshold_yellow);
//        inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold_yellow);
    if (useMorphOps)
        morphOps(threshold_yellow);
    if (trackObjects) {
        trackFilteredObject(x, y, threshold_yellow, cameraFeed);
    }
    cout << "x yellow: " << x << " " << " y yellow" << y << "\n";
}

void setCoordinates_GREEN(int &x, int &y, Mat &cameraFeed, Mat HSV, Mat &threshold_green) {
    cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
    inRange(HSV, Scalar(47, 95, 0), Scalar(87, 256, 256), threshold_green);
    if (useMorphOps)
        morphOps(threshold_green);
    if (trackObjects) {
        trackFilteredObject(x, y, threshold_green, cameraFeed);
    }
    cout << "x verde: " << x << " " << " y verde" << y << "\n";
}

double getDistance(int x1, int y1, int x2, int y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

int main(int argc, char *argv[]) {
    //structures for sockets
    configureSocket();

    Point p;
    Mat cameraFeed; //Matrix to store each frame of the webcam feed
    Mat HSV; //matrix storage for HSV image
    Mat threshold_green;
    Mat threshold_red;
    Mat threshold_blue;
    Mat threshold_yellow;

    //create slider bars for HSV filtering
//    createTrackbars();
    //video capture object to acquire webcam feed
    VideoCapture capture;
    //open capture object at location zero (default location for webcam)
    //capture.open("rtmp://172.16.254.63/live/live");    //set height and width of capture frame
    capture.open(1);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop
    while (1) {
        //store image to matrix
        capture.read(cameraFeed);

        setCoordinates_GREEN(xb, yb, cameraFeed, HSV, threshold_green);
        setCoordinates_YELLOW(xd, yd, cameraFeed, HSV, threshold_yellow);
        setCoordinates_BLUE(xo, yo, cameraFeed, HSV, threshold_blue);

        double bd_dist = getDistance(xb, yb, xd, yd);
        double bo_dist = getDistance(xb, yb, xo, yo);
        double do_dist = getDistance(xd, yd, xo, yo);
        double diff = (bd_dist + do_dist) - (bo_dist + 5);

        cout << "a: " << bd_dist << " b: " << do_dist << " c: " << bo_dist << " Diff: " << diff << "\n";

        if ((bd_dist + do_dist) > bo_dist + 5) {
            if (diff > 6) {
                sendCommand('r', diff * 1.5);
            } else {
                sendCommand('r', 70);
            }
        } else {
//            sendCommand('s', 500);
            sendCommand('f', 700);
        }

        //show frames
//          imshow(windowName2, threshold_yellow);
//        imshow(windowName, cameraFeed);
//        imshow(windowName1, cameraFeed);
//        setMouseCallback("Original Image", on_mouse, &p);
        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
//        waitKey(30);
    }

    return 0;
}
