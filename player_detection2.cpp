
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() 
{
    // Reading the video
    cv::VideoCapture vidcap("cutvideo.mp4");
    if (!vidcap.isOpened()) 
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat image;
    int count = 0;
    int idx = 0;

    // Resize factor
    double resize_factor = 0.48;

    // Read the video frame by frame
    while (true) 
    {
        vidcap >> image;
        if (image.empty()) 
        {
            break;
        }

        // Resize frames
        cv::Mat resized_frame;
        cv::resize(image, resized_frame, cv::Size(), resize_factor, resize_factor);

        // Display original frame
        cv::imshow("Original Frame", resized_frame);

        // Convert BGR to HSV
        cv::Mat hsv, resized_hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        cv::resize(hsv, resized_hsv, cv::Size(), resize_factor, resize_factor);
        cv::imshow("HSV Image", resized_hsv);

        // Define color ranges
        cv::Scalar lower_green(40, 40, 40), upper_green(70, 255, 255);
        cv::Scalar lower_blue(110, 50, 50), upper_blue(130, 255, 255);
        cv::Scalar lower_red(0, 31, 255), upper_red(176, 255, 255);
        cv::Scalar lower_white(0, 0, 0), upper_white(0, 0, 255);

        // Create masks
        cv::Mat mask_green, mask_blue, mask_red, mask_white;
        cv::inRange(hsv, lower_green, upper_green, mask_green);
        cv::inRange(hsv, lower_blue, upper_blue, mask_blue);
        cv::inRange(hsv, lower_red, upper_red, mask_red);
        cv::inRange(hsv, lower_white, upper_white, mask_white);

        // Bitwise AND operation
        cv::Mat res_green, res_blue, res_red, res_white;
        cv::bitwise_and(image, image, res_green, mask_green);
        cv::bitwise_and(image, image, res_blue, mask_blue);
        cv::bitwise_and(image, image, res_red, mask_red);
        cv::bitwise_and(image, image, res_white, mask_white);

        cv::Mat resized_res_green, resized_res_blue, resized_res_red, resized_res_white;
        cv::resize(res_green, resized_res_green, cv::Size(), resize_factor, resize_factor);
        cv::resize(res_blue, resized_res_blue, cv::Size(), resize_factor, resize_factor);
        cv::resize(res_red, resized_res_red, cv::Size(), resize_factor, resize_factor);
        cv::resize(res_white, resized_res_white, cv::Size(), resize_factor, resize_factor);

        cv::imshow("Green Mask", resized_res_green);
        cv::imshow("Blue Mask", resized_res_blue);
        cv::imshow("Red Mask", resized_res_red);
        cv::imshow("White Mask", resized_res_white);

        cv::Mat res_gray, thresh;
        cv::cvtColor(res_green, res_gray, cv::COLOR_BGR2GRAY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
        cv::threshold(res_gray, thresh, 127, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        int font_face = cv::FONT_HERSHEY_SIMPLEX;

        for (const auto& c : contours) 
        {
            cv::Rect rect = cv::boundingRect(c);
            int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
            if (h >= 1.5 * w) 
            {
                if (w > 15 && h >= 15) 
                {
                    idx++;
                    cv::Mat player_img = image(cv::Rect(x, y, w, h));
                    cv::Mat player_hsv;
                    cv::cvtColor(player_img, player_hsv, cv::COLOR_BGR2HSV);
                    int nzCount_blue = cv::countNonZero(cv::inRange(player_hsv, lower_blue, upper_blue));
                    int nzCount_red = cv::countNonZero(cv::inRange(player_hsv, lower_red, upper_red));
                    if (nzCount_blue >= 20) 
                    {
                        cv::putText(image, "France", cv::Point(x - 2, y - 2), font_face, 0.8, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                        cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 3);
                    }
                    if (nzCount_red >= 20) 
                    {
                        cv::putText(image, "Belgium", cv::Point(x - 2, y - 2), font_face, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                        cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 3);
                    }
                }
            }
            if (1 <= h && h <= 30 && 1 <= w && w <= 30) 
            {
                cv::Mat small_img = image(cv::Rect(x, y, w, h));
                cv::Mat small_hsv;
                cv::cvtColor(small_img, small_hsv, cv::COLOR_BGR2HSV);
                int nzCount_white = cv::countNonZero(cv::inRange(small_hsv, lower_white, upper_white));
                if (nzCount_white >= 3) 
                {
                    cv::putText(image, "football", cv::Point(x - 2, y - 2), font_face, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 3);
                }
            }
        }

        cv::Mat resized_image, resized_thresh;
        cv::resize(image, resized_image, cv::Size(), resize_factor, resize_factor);
        cv::resize(thresh, resized_thresh, cv::Size(), resize_factor, resize_factor);

        cv::imshow("Processed Frame", resized_image);
        cv::imshow("Thresholded Image", resized_thresh);

        std::string filename = "./Cropped/frame" + std::to_string(count) + ".jpg";
        cv::imwrite(filename, res_green);
        std::cout << "Read a new frame: " << !image.empty() << std::endl;
        count++;

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 'Q') 
        {
            break;
        }
    }

    vidcap.release();
    cv::destroyAllWindows();

    return 0;
}

