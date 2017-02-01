#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <string>
#include <thread>
#include <functional>
#include <mutex>
#include <map>
#include <cmath>
#include <algorithm>

#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define CANNY_THRESHOLD 90
#define CANNY_RATIO 3
#define SQUARE_SIZE 8
#define ROI_SIZE_LIMIT 200
#define WEIGHTING_THRESHOLD 20
#define CLUSTER_DISTANCE 25


bool pointPredicate(const Point2f &a, const Point2f &b);
Mat cannyEdgeDetection(Mat input_matrix);
void findRegionsOfInterest(Mat proc_matrix, Mat disp_matrix, map<pair<int,int>, int > *top_left_points_of_interest);
void drawRectangles(Mat source_image, vector<set<pair<int,int> > > *known_clusters);
void processSection(Mat proc_matrix, Mat disp_matrix, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset);
void formClusters(map<pair<int,int>, int > *top_left_points, vector<set<pair<int,int> > >  *known_clusters);
void updateCluster(pair<int,int> point, set<pair<int,int> > *cluster);
void startDownSampleThreads(Mat source_image, map<pair<int,int>, int > *top_left_points_of_interest);
void downSampleImage(Mat source_image, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset);
void downSampleMat(Mat input_matrix);
void setChannel(Mat &mat, unsigned int channel, unsigned char value);

// Avoid segfaults when multithreading occurs in drawRegionsOfInterest()
mutex points_of_interest_mutex;

int main(int argc, char** argv)
{
    int compression_level;
    vector<string> files_to_read;

    string help_message = "Input options:\n"
                          "\t-f:<file name> OR -d:<directory path>\n"
                          "\t<compression level>\n"
                          "\t-disp (OPTIONAL)";
    if (!(argc == 3 || argc == 4))
    {
        cout << help_message << endl;
        return -1;
    }

    string input_1 = argv[1];
    if (input_1.substr(0,3).compare("-f:") == 0)
    {
        files_to_read.push_back(input_1.substr(3));
    }
    else if (input_1.substr(0,3).compare("-d:") == 0)
    {
        string dir_path = input_1.substr(3) + "*.jpg";
        glob(dir_path, files_to_read, true);
    }
    else
    {
        cout << help_message << endl;
        return -1;
    }

    int input_2 = atoi(argv[2]);
    if (0 <= input_2 && input_2 <= 100)
    {
        compression_level = input_2;
    }
    else
    {
        cout << "Compression level must be between 0 and 100." << endl;
        return -1;
    }

    bool display_result = false;
    if (argc == 4)
    {
        string input_3 = argv[3];
        if (input_3.compare("-disp") == 0)
            display_result = true;
    }

    for (auto curr_file = files_to_read.begin(); curr_file != files_to_read.end(); ++curr_file)
    {
        // Check that this image hasn't already been compressed
        int len = curr_file->size();
        if (curr_file->substr(len - 15).compare("_COMPRESSED.jpg") == 0)
            continue;

        // Announce the current file name
        cout << "--------------------------------------------------\n"
             << "Current file: " << *curr_file << "\n"
             << endl;
        Mat source_image = imread(*curr_file);
        if (source_image.empty())
        {
            cout << "No image data." << endl;
            continue;
        }


        // Timing
        double t = (double)getTickCount();

        // Pad the source image so that the square size fits without missing spots
        int pad_down = source_image.size().height % SQUARE_SIZE;
        int pad_right = source_image.size().width % SQUARE_SIZE;
        copyMakeBorder(source_image, source_image, 0, pad_down, 0, pad_right, BORDER_REPLICATE);

        // Convert to HSV
        Mat hsv_source;
        vector<Mat> hsv_channels, edge_channel;
        cvtColor(source_image, hsv_source, CV_BGR2HSV);
        split(hsv_source, hsv_channels);
        hsv_source.release();

        // Perform the edge-detection on each HSV channel
        edge_channel.push_back(cannyEdgeDetection(hsv_channels[0]));
        edge_channel.push_back(cannyEdgeDetection(hsv_channels[1]));
        edge_channel.push_back(cannyEdgeDetection(hsv_channels[2]));

        // Clusters of edges are grouped to be highlighted
        vector<set<pair<int,int> > >  known_clusters;
        map<pair<int,int>, int > top_left_points_of_interest; // x,y,weighting

        for (int i = 0; i != 3; ++i)
            findRegionsOfInterest(edge_channel[i], source_image, &top_left_points_of_interest);

        formClusters(&top_left_points_of_interest, &known_clusters);
        startDownSampleThreads(source_image, &top_left_points_of_interest);
        drawRectangles(source_image, &known_clusters);

        // Compress and write image
        vector<int> compVec = {CV_IMWRITE_JPEG_QUALITY, compression_level};
        string image_name = *curr_file;
        int i = image_name.find(".");
        image_name = image_name.substr(0,i);
        image_name.insert(i, "_COMPRESSED.jpg");
        imwrite(image_name, source_image, compVec);

        // Announce timing and end output block
        t = ((double)getTickCount() - t)/getTickFrequency();
        std::cout << "Processing time: " << t << " s\n"
                  << "--------------------------------------------------"
                  << std::endl;

        if (display_result)
        {
            namedWindow("Processed", WINDOW_AUTOSIZE);
            imshow("Processed", source_image);
            waitKey(0);
        }
    }
    return 0;
}

bool pointPredicate(const Point2f &a, const Point2f &b)
{
    int dist = sqrt(pow((a.x - b.x),2) + pow((a.y - b.y),2));
    return (dist < CLUSTER_DISTANCE);
}

Mat cannyEdgeDetection(Mat input_matrix)
{
    Mat ret_matrix;
    // Reduce noise with blur
    blur(input_matrix, ret_matrix, Size(3,3)); // TODO: Try different blur kernel sizes

    // Canny edge detection
    // TODO: Change kernel size and canny ratio
    Canny(ret_matrix, ret_matrix, CANNY_THRESHOLD, CANNY_THRESHOLD*CANNY_RATIO, 3);

    return ret_matrix;
}

// proc_matrix is used to find areas of interest
// disp_matrix is used to display these regions
void findRegionsOfInterest(Mat proc_matrix, Mat disp_matrix, map<pair<int,int>, int > *top_left_points_of_interest)
{
    // Highlight regions of interest on 4 threads
    int r = proc_matrix.rows;
    int c = proc_matrix.cols;
    int adj_r = 0;
    int adj_c = 0;
    if (r % 2 == 0)
        adj_r = 1;
    if (c % 2 == 0)
        adj_c = 1;

    Mat proc1 = proc_matrix(Rect(0,0,adj_c+c/2,adj_r+r/2));
    Mat src1 = disp_matrix(Rect(0,0,c/2,r/2));
    auto regionHandle1 = bind(processSection, proc1, src1, top_left_points_of_interest, 0, 0);

    Mat proc2 = proc_matrix(Rect(c/2,0,c/2,adj_r+r/2));
    Mat src2 = disp_matrix(Rect(c/2,0,c/2,r/2));
    auto regionHandle2 = bind(processSection, proc2, src2, top_left_points_of_interest, c/2, 0);

    Mat proc3 = proc_matrix(Rect(0,r/2,adj_c+c/2,r/2));
    Mat src3 = disp_matrix(Rect(0,r/2,c/2,r/2));
    auto regionHandle3 = bind(processSection, proc3, src3, top_left_points_of_interest, 0, r/2);

    Mat proc4 = proc_matrix(Rect(c/2,r/2,c/2,r/2));
    Mat src4 = disp_matrix(Rect(c/2,r/2,c/2,r/2));
    auto regionHandle4 = bind(processSection, proc4, src4, top_left_points_of_interest, c/2, r/2);

    thread regionThread1 (regionHandle1);
    thread regionThread2 (regionHandle2);
    thread regionThread3 (regionHandle3);
    thread regionThread4 (regionHandle4);

    regionThread1.join();
    regionThread2.join();
    regionThread3.join();
    regionThread4.join();
}

void drawRectangles(Mat source_image, vector<set<pair<int,int> > >  *known_clusters)
{
    for (auto cluster = known_clusters->begin(); cluster != known_clusters->end(); ++cluster)
    {
        int min_x = -1;
        int min_y = -1;
        int max_x = -1;
        int max_y = -1;
        for (auto pixel_point = cluster->begin(); pixel_point != cluster->end(); ++pixel_point)
        {

            if (min_x == -1)
                min_x = pixel_point->first;
            if (min_y == -1)
                min_y = pixel_point->second;

            min_x = min(min_x, pixel_point->first);
            min_y = min(min_y, pixel_point->second);
            max_x = max(max_x, pixel_point->first);
            max_y = max(max_y, pixel_point->second);
        }
        if (!(max_x - min_x > ROI_SIZE_LIMIT || max_y - min_y > ROI_SIZE_LIMIT))
            rectangle(source_image, Point(min_x, min_y), Point(max_x + SQUARE_SIZE, max_y + SQUARE_SIZE), Scalar(0,200,50), 1);
    }
}

void processSection(Mat proc_matrix, Mat disp_matrix, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset)
{
    int height = proc_matrix.size().height;
    int width = proc_matrix.size().width;
    double total = sum(proc_matrix)[0];


    for (int i = 0; i < width - SQUARE_SIZE; i += SQUARE_SIZE)
    {
        for (int j = 0; j < height - SQUARE_SIZE; j += SQUARE_SIZE)
        {
            // Process 8x8 sections
            Mat section = proc_matrix(Rect(i,j,SQUARE_SIZE,SQUARE_SIZE));
            double sub_total = sum(section)[0];

            int section_weight = 10000*sub_total/total;
            if (section_weight > WEIGHTING_THRESHOLD) // Region of interest
            {
                // Add to existing POI or create a new one
                auto point_of_interest = top_left_points_of_interest->find({i+i_offset,j+j_offset});
                if (point_of_interest != top_left_points_of_interest->end())
                {
                    point_of_interest->second += 1;
                }
                else
                {
                    lock_guard<mutex> lock(points_of_interest_mutex);
                    top_left_points_of_interest->insert({make_pair(i+i_offset,j+j_offset), 1});
                }
            }
        }
    }
}

void formClusters(map<pair<int,int>, int > *top_left_points, vector<set<pair<int,int> > >  *known_clusters)
{
    // Convert pairs to Point2fs
    vector<Point2f> points_to_cluster;
    for (auto point = top_left_points->begin(); point != top_left_points->end(); ++point)
    {
        points_to_cluster.push_back(Point2f(point->first.first, point->first.second));
    }

    // Use the partition algorithm to determine clusters
    vector<int> labels;
    int num_clusters = partition(points_to_cluster, labels, pointPredicate);
    cout << "Regions of interest found: " << num_clusters << endl;

    // Record the clusters found
    known_clusters->resize(num_clusters);
    for (int i = 0; i != labels.size(); ++i)
    {
        auto *tmp_cluster = &(*known_clusters)[labels[i]];
        auto this_point = make_pair(points_to_cluster[i].x, points_to_cluster[i].y);
        tmp_cluster->insert(this_point);
    }
}

void updateCluster(pair<int,int> point, set<pair<int,int> > *cluster)
{
    int x = point.first;
    int y = point.second;
    cluster->insert({x,y});
    cluster->insert({x + SQUARE_SIZE,y});
    cluster->insert({x,y + SQUARE_SIZE});
    cluster->insert({x + SQUARE_SIZE,y + SQUARE_SIZE});
}

void startDownSampleThreads(Mat source_image, map<pair<int,int>, int > *top_left_points_of_interest)
{
    // Highlight regions of interest on 4 threads
    int r = source_image.rows;
    int c = source_image.cols;
    int adj_r = 0;
    int adj_c = 0;
    if (r % 2 == 0)
        adj_r = 1;
    if (c % 2 == 0)
        adj_c = 1;

    Mat src1 = source_image(Rect(0,0,adj_c+c/2,adj_r+r/2));
    auto downSampleHandle1 = bind(downSampleImage, src1, top_left_points_of_interest, 0, 0);

    Mat src2 = source_image(Rect(c/2,0,c/2,adj_r+r/2));
    auto downSampleHandle2 = bind(downSampleImage, src2, top_left_points_of_interest, c/2, 0);

    Mat src3 = source_image(Rect(0,r/2,adj_c+c/2,r/2));
    auto downSampleHandle3 = bind(downSampleImage, src3, top_left_points_of_interest, 0, r/2);

    Mat src4 = source_image(Rect(c/2,r/2,c/2,r/2));
    auto downSampleHandle4 = bind(downSampleImage, src4, top_left_points_of_interest, c/2, r/2);

    thread downSampleThread1 (downSampleHandle1);
    thread downSampleThread2 (downSampleHandle2);
    thread downSampleThread3 (downSampleHandle3);
    thread downSampleThread4 (downSampleHandle4);

    downSampleThread1.join();
    downSampleThread2.join();
    downSampleThread3.join();
    downSampleThread4.join();
}

void downSampleImage(Mat source_image, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset)
{
    int height = source_image.size().height;
    int width = source_image.size().width;

    for (int i = 0; i < width - SQUARE_SIZE; i += SQUARE_SIZE)
    {
        for (int j = 0; j < height - SQUARE_SIZE; j += SQUARE_SIZE)
        {
            auto point = top_left_points_of_interest->find({i+i_offset, j+j_offset});
            if (point == top_left_points_of_interest->end())
            {
                Mat source_section = source_image(Rect(i,j,SQUARE_SIZE,SQUARE_SIZE));
                downSampleMat(source_section);
            }
        }
    }
}

void downSampleMat(Mat input_matrix)
{
    int height = input_matrix.size().height;
    int width = input_matrix.size().width;

    double totalB = sum(input_matrix)[0];
    double totalG = sum(input_matrix)[1];
    double totalR = sum(input_matrix)[2];

    int avgColourB = totalB/(height*width);
    int avgColourG = totalG/(height*width);
    int avgColourR = totalR/(height*width);

    setChannel(input_matrix, 0, avgColourB);
    setChannel(input_matrix, 1, avgColourG);
    setChannel(input_matrix, 2, avgColourR);
}

void setChannel(Mat &mat, unsigned int channel, unsigned char value)
{
    // Set all values in the input matrix to a value
    if (mat.channels() < channel + 1)
        return;

    const int cols = mat.cols;
    const int step = mat.channels();
    const int rows = mat.rows;
    for (int y = 0; y < rows; y++)
    {
        unsigned char *p_row = mat.ptr(y) + channel;
        unsigned char *row_end = p_row + cols*step;
        for (; p_row != row_end; p_row += step)
            *p_row = value;
    }
}
