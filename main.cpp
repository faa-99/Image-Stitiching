#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <stdio.h>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace detail;

Mat crop_image(Mat result)
{
    // //Finding the largest contour i.e remove the black region from image

    Mat img_gray;
    img_gray = result.clone();
    img_gray.convertTo(img_gray, CV_8UC1);
    cvtColor(img_gray, img_gray, COLOR_BGR2GRAY);
    threshold(img_gray, img_gray, 25, 255, THRESH_BINARY); //Threshold the gray

    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
    findContours(img_gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image
    int largest_area = 0;
    int largest_contour_index = 0;
    Rect bounding_rect;


    for (int i = 0; i< contours.size(); i++) // iterate through each contour.
    {
        double a = contourArea(contours[i], false);  //  Find the area of contour
        if (a>largest_area)
        {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
            bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour

        }

    }
    result = result(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
    return result;

}

int main()
{

//  Initializing variables

    //  Read images
    Mat image1 = imread("images/Hill1.jpg");
    Mat image2 = imread("images/Hill2.jpg");
    Mat im1 = image1;

    //  Detector and Descriptor
    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create();
    Ptr<FREAK> descriptor = FREAK::create();

    //  Key-points
    vector<KeyPoint> lastFramekeypoints1, lastFramekeypoints2;

    //  Descriptors
    Mat lastFrameDescriptors1, lastFrameDescriptors2;

    //  Matcher
    Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;

    //  Finding key-points and their descriptors
    Mat img1_keypoints, img2_keypoints;
    detector->detect(image1, lastFramekeypoints1);
    detector->detect(image2, lastFramekeypoints2);
    descriptor->compute(image1, lastFramekeypoints1, lastFrameDescriptors1);
    descriptor->compute(image2, lastFramekeypoints2, lastFrameDescriptors2);

    if(lastFrameDescriptors1.type()!=CV_32F)
    {
        lastFrameDescriptors1.convertTo(lastFrameDescriptors1, CV_32F);
    }

    if(lastFrameDescriptors2.type()!=CV_32F)
    {
        lastFrameDescriptors2.convertTo(lastFrameDescriptors2, CV_32F);
    }

    //Draw Keypoints on images
    drawKeypoints(image1, lastFramekeypoints1, img1_keypoints, Scalar::all(-1),
                  DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, lastFramekeypoints2, img2_keypoints, Scalar::all(-1),
                  DrawMatchesFlags::DEFAULT);
    //  Show detected (drawn) key-points on images
    imwrite("output_images/img1_keypoints.jpeg", img1_keypoints);
    imwrite("output_images/img2_keypoints.jpeg", img2_keypoints);

    //  Match the descriptors between the two images
    matcher->knnMatch( lastFrameDescriptors1,lastFrameDescriptors2, knn_matches, 2);

    //  Filter matches using the Lowe's ratio test (KNN)
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    // Draw matches
    Mat img_matches;
    drawMatches(image1, lastFramekeypoints1, image2, lastFramekeypoints2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("output_images/goodmatches.jpeg", img_matches);

    // Localize the object
    std::vector<Point2f> image1_goodpoints;
    std::vector<Point2f> image2_goodpoints;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        // Get the key-points from the good matches
        image1_goodpoints.push_back( lastFramekeypoints1[ good_matches[i].queryIdx ].pt );
        image2_goodpoints.push_back( lastFramekeypoints2[ good_matches[i].trainIdx ].pt );
    }

    //  Warp Image
    //  Warp image 2 into image 1
    Mat H = findHomography(image2_goodpoints, image1_goodpoints, RANSAC);
    Mat warped_image2;
    warpPerspective(image2, warped_image2, H, Size(image1.cols + image2.cols, image1.rows));
    Mat half1 = warped_image2(Rect(0,0,warped_image2.cols/2, warped_image2.rows));
    Mat half2 = warped_image2(Rect(warped_image2.cols/2,0,warped_image2.cols/2, warped_image2.rows));
    imwrite("output_images/warped_image.jpeg", warped_image2);

    int rows = image1.rows;
    int cols = image1.cols + image2.cols;

    //  Create black canvas
    Mat3b result(rows, cols, Vec3b(0, 0, 0));

    warped_image2.copyTo(result(Rect(0, 0, warped_image2.cols, warped_image2.rows)));
    im1.copyTo(result(Rect(0, 0, im1.cols, im1.rows)));
    result = crop_image(result);
    imwrite("output_images/Stitched.jpeg", result);

    // Blending using Gaussian and Laplacian Pyramids
    vector<Mat> g_pyramid_1;
    buildPyramid(result, g_pyramid_1, 6);

    vector<Mat> l_pyramid_1;
    for(int i = 5; i>0; i--){
        Mat gaussian_expanded_1;
        pyrUp(g_pyramid_1[i], gaussian_expanded_1);
        resize(gaussian_expanded_1, gaussian_expanded_1, g_pyramid_1[i-1].size());
        l_pyramid_1.push_back(g_pyramid_1[i-1] - gaussian_expanded_1);
        }

    vector<Mat> pyramid;

    for(int i =0; i<l_pyramid_1.size(); i++){
        Mat3b laplacian(l_pyramid_1[i].rows, l_pyramid_1[i].cols, Vec3b(1,1,1));
        l_pyramid_1[i].copyTo(laplacian(Rect(0, 0, l_pyramid_1[i].cols, l_pyramid_1[i].rows)));
        pyramid.push_back(laplacian);
    }
    Mat pyramid_reconstruct = pyramid[0];
    for(int i = 1; i<5; i++){
        pyrUp(pyramid[i], pyramid[i]);
        resize(pyramid_reconstruct, pyramid_reconstruct, pyramid[i].size());
        pyramid_reconstruct = pyramid_reconstruct + pyramid[i];
    }

    imwrite("output_images/blended_image.jpeg", pyramid_reconstruct);

    return 0;

}
