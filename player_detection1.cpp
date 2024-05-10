
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() 
{
    cv::VideoCapture vidcap("cutvideo.mp4"); 
    if (!vidcap.isOpened()) 
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat image;
    int count = 0;
    int idx = 0;

    while (true) 
    {
        vidcap >> image;
        if (image.empty()) 
        {
            break;
        }

        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

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

        cv::Mat res_gray;
        cv::cvtColor(res_green, res_gray, cv::COLOR_BGR2GRAY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
        cv::Mat thresh;
        cv::threshold(res_gray, thresh, 127, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        int font_face = cv::FONT_HERSHEY_SIMPLEX;

        for (const auto& c : contours) 
        {
            cv::Rect rect = cv::boundingRect(c);
            if (rect.height >= 1.5 * rect.width) 
            {
                if (rect.width > 15 && rect.height >= 15) 
                {
                    idx++;
                    cv::Mat player_img = image(cv::Rect(rect.x, rect.y, rect.width, rect.height));
                    cv::Mat player_hsv;
                    cv::cvtColor(player_img, player_hsv, cv::COLOR_BGR2HSV);
                    int nzCount_blue = cv::countNonZero(cv::inRange(player_hsv, lower_blue, upper_blue));
                    int nzCount_red = cv::countNonZero(cv::inRange(player_hsv, lower_red, upper_red));
                    if (nzCount_blue >= 20) {
                        cv::putText(image, "France", cv::Point(rect.x - 2, rect.y - 2), font_face, 0.8, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                        cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 3);
                    }
                    if (nzCount_red >= 20) 
                    {
                        cv::putText(image, "Belgium", cv::Point(rect.x - 2, rect.y - 2), font_face, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                        cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 3);
                    }
                }
            }
            if (rect.height >= 1 && rect.height <= 30 && rect.width >= 1 && rect.width <= 30) 
            {
                cv::Mat roi = image(cv::Rect(rect.x, rect.y, rect.width, rect.height));
                cv::Mat roi_hsv;
                cv::cvtColor(roi, roi_hsv, cv::COLOR_BGR2HSV);
                int nzCount_white = cv::countNonZero(cv::inRange(roi_hsv, lower_white, upper_white));
                if (nzCount_white >= 3) 
                {
                    cv::putText(image, "football", cv::Point(rect.x - 2, rect.y - 2), font_face, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 3);
                }
            }
        }

        std::string filename = "./Cropped/frame" + std::to_string(count) + ".jpg";
        cv::imwrite(filename, res_green);
        std::cout << "Read a new frame: " << !image.empty() << std::endl;
        count++;
        cv::imshow("Match Detection", image);
        if (cv::waitKey(1) == 'q') 
        {
            break;
        }
    }

    vidcap.release();
    cv::destroyAllWindows();
    return 0;
}

