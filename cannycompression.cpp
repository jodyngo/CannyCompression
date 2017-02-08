#include "opencv2/opencv.hpp"
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
#include <fstream>
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

bool readConfigFile(vector<double> *config_params, bool &size_thresholding);
bool pointPredicate(const Point2f &a, const Point2f &b);
void cannyEdgeDetection(Mat input_matrix, Mat output_matrix);
void findRegionsOfInterest(Mat proc_matrix, map<pair<int,int>, int > *top_left_points_of_interest);
void drawRectangles(Mat source_image, vector<set<pair<int,int> > > *known_clusters);
void processSection(Mat proc_matrix, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset);
void trimPointsOfInterest(map<pair<int,int>, int > *top_left_points_of_interest, vector<int> *point_weightings);
void formClusters(map<pair<int,int>, int > *top_left_points, vector<set<pair<int,int> > >  *known_clusters, bool size_thresholding, vector<double> viewing_data);
bool acceptableClusterSize(set<pair<int,int> > *cluster, double pixel_length_ratio, double altitude, double viewing_angle, double target_size);
int findPeople(Mat source_image, vector<set<pair<int,int> > >  *known_clusters);
void startDownSampleThreads(Mat source_image, vector<set<pair<int,int> > >  *known_clusters);
void downSampleImage(Mat source_image, vector<set<pair<int,int> > >  *known_clusters, int i_offset, int j_offset);
void downSampleMat(Mat input_matrix);
void setChannel(Mat &mat, int channel, unsigned char value);

// Avoid segfaults when multithreading occurs in drawRegionsOfInterest()
mutex points_of_interest_mutex;

int main(int argc, char** argv)
{
    int compression_level;
    vector<string> files_to_read;

    string help_message = "Input options:\n"
                          "\t-f:<file name> OR -d:<directory path>\n"
                          "\tOPTIONAL: -disp";
    if (!(argc == 2 || argc == 3))
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

    bool display_result = false;
    if (argc == 3)
    {
        string input_2 = argv[2];
        if (input_2.compare("-disp") == 0)
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

        // Update all configuration parameters from config file
        // {target_size, focal_length, altitude, pixel_size, compression_level};
        vector<double> config_params (5);
        bool size_thresholding = false;
        if (!readConfigFile(&config_params, size_thresholding))
            return -1;

        cout << "target_size: " << config_params[0] << "\n"
             << "focal_length: " << config_params[1] << "\n"
             << "altitude: " << config_params[2] << "\n"
             << "pixel_size: " << config_params[3] << "\n"
             << "compression_level: " << config_params[4] << endl;

        // Timing
        double t = (double)getTickCount();

        // Pad the source image so that the square size fits without missing spots
        int pad_down = source_image.size().height % 2*SQUARE_SIZE;
        int pad_right = source_image.size().width % 2*SQUARE_SIZE;
        copyMakeBorder(source_image, source_image, 0, pad_down, 0, pad_right, BORDER_REPLICATE);

        // Convert to HSV
        Mat hsv_source;
        vector<Mat> hsv_channels, edge_channel;
        cvtColor(source_image, hsv_source, CV_BGR2HSV);
        split(hsv_source, hsv_channels);
        hsv_source.release();

        // Perform the edge-detection on each HSV channel
        // Multithreading for speed
        edge_channel.resize(3);
        edge_channel[0].create(hsv_channels[0].size(), hsv_channels[0].type());
        edge_channel[1].create(hsv_channels[1].size(), hsv_channels[1].type());
        edge_channel[2].create(hsv_channels[2].size(), hsv_channels[2].type());


        auto cannyHandle0 = bind(cannyEdgeDetection, hsv_channels[0], edge_channel[0]);
        auto cannyHandle1 = bind(cannyEdgeDetection, hsv_channels[1], edge_channel[1]);
        auto cannyHandle2 = bind(cannyEdgeDetection, hsv_channels[2], edge_channel[2]);

        thread cannyThread0 (cannyHandle0);
        thread cannyThread1 (cannyHandle1);
        thread cannyThread2 (cannyHandle2);

        cannyThread0.join();
        cannyThread1.join();
        cannyThread2.join();

        // Clusters of edges are grouped to be highlighted
        vector<set<pair<int,int> > >  known_clusters;
        map<pair<int,int>, int > top_left_points_of_interest; // x,y,weighting
        vector<int> point_weightings;

        for (int i = 0; i != 3; ++i)
            findRegionsOfInterest(edge_channel[i], &top_left_points_of_interest);

        trimPointsOfInterest(&top_left_points_of_interest, &point_weightings);
        formClusters(&top_left_points_of_interest, &known_clusters, size_thresholding, config_params);
        startDownSampleThreads(source_image, &known_clusters);
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

bool readConfigFile(vector<double> *config_params, bool &size_thresholding)
{
    static const map<string, int> param_indices = {
        {"target_size", 0},
        {"focal_length", 1},
        {"altitude", 2},
        {"pixel_size", 3},
        {"compression_level", 4}
    };

    // Default values for the parameters in case of config file mistakes
    (*config_params)[0] =   2.0;
    (*config_params)[1] =   8.5;
    (*config_params)[2] = 100.0;
    (*config_params)[3] =  3.69;
    (*config_params)[4] =    80;

    ifstream config_file;
    config_file.open("config.conf");

    if (!config_file.is_open())
    {
        cout << "Unable to open or locate the configuration file."
             << "\nUsing default values." << endl;
        return -1;
    }

    while (config_file)
    {
        string config_line;
        getline(config_file, config_line);

        size_t i = config_line.find('=');
        if (i == string::npos)
            continue;
        // Extract the parameter name and value, stripping the string of
        // excess spaces
        string param_name = config_line.substr(0, i);
        size_t j = param_name.find_first_of(' ');
        param_name = param_name.substr(0, j);

        auto param_iter = param_indices.find(param_name);
        if (param_iter == param_indices.end() &&
            param_name.compare("size_thresholding") != 0)
        {
            cout << "Warning: Config file parameter \""
                 << param_name
                 << "\" not recognised." << endl;
            continue;
        }

        string param_value = config_line.substr(i + 1);
        j = param_value.find_last_of(' ');
        param_value = param_value.substr(j + 1);

        if (param_name.compare("size_thresholding") == 0)
        {
            if (param_value.compare("true") == 0)
            {
                size_thresholding = true;
            }
            else if (param_value.compare("false") == 0)
            {
                size_thresholding = false;
            }
            else
            {
                cout << "Warning: Config file value \""
                     << param_value
                     << "\" not recognised as a boolean." << endl;
            }
        }
        else
        {
            (*config_params)[param_iter->second] = stod(param_value);
        }
    }

    config_file.close();
    return 1;
}

bool pointPredicate(const Point2f &a, const Point2f &b)
{
    int dist = sqrt(pow((a.x - b.x),2) + pow((a.y - b.y),2));
    return (dist < CLUSTER_DISTANCE);
}

void cannyEdgeDetection(Mat input_matrix, Mat output_matrix)
{
    // Reduce noise with blur
    bilateralFilter(input_matrix, output_matrix, 8, 100, 100);

    // Canny edge detection
    Canny(output_matrix, output_matrix, CANNY_THRESHOLD, CANNY_THRESHOLD*CANNY_RATIO, 3);
}

// proc_matrix is used to find areas of interest
// disp_matrix is used to display these regions
void findRegionsOfInterest(Mat proc_matrix, map<pair<int,int>, int > *top_left_points_of_interest)
{
    // Highlight regions of interest on 4 threads
    int cx = proc_matrix.cols/2;
    int cy = proc_matrix.rows/2;

    Mat proc1 = proc_matrix(Rect(0,0,cx + 8,cy + 8));
    auto regionHandle1 = bind(processSection, proc1, top_left_points_of_interest, 0, 0);

    Mat proc2 = proc_matrix(Rect(cx,0,cx,cy + 8));
    auto regionHandle2 = bind(processSection, proc2, top_left_points_of_interest, cx, 0);

    Mat proc3 = proc_matrix(Rect(0,cy,cx + 8,cy));
    auto regionHandle3 = bind(processSection, proc3, top_left_points_of_interest, 0, cy);

    Mat proc4 = proc_matrix(Rect(cx,cy,cx,cy));
    auto regionHandle4 = bind(processSection, proc4, top_left_points_of_interest, cx, cy);

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
        rectangle(source_image, Point(min_x, min_y), Point(max_x + SQUARE_SIZE, max_y + SQUARE_SIZE), Scalar(12,12,175), 1);
    }
}

void processSection(Mat proc_matrix, map<pair<int,int>, int > *top_left_points_of_interest, int i_offset, int j_offset)
{
    int scaling = 1;
    if (i_offset == -1 && j_offset == -1)
    {
        scaling = 1;
        i_offset = 0;
        j_offset = 0;
    }
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

            double section_weight = scaling*10000*sub_total/total;
            if (section_weight > WEIGHTING_THRESHOLD) // Region of interest
            {
                // Add to existing POI or create a new one
                auto point_of_interest = top_left_points_of_interest->find({i+i_offset,j+j_offset});
                if (point_of_interest != top_left_points_of_interest->end())
                {
                    // Existing POI
                    point_of_interest->second += section_weight;
                }
                else
                {
                    // New POI
                    lock_guard<mutex> lock(points_of_interest_mutex);
                    top_left_points_of_interest->insert({make_pair(i+i_offset,j+j_offset), section_weight});
                }
            }
        }
    }
}

void trimPointsOfInterest(map<pair<int,int>, int > *top_left_points_of_interest, vector<int> *point_weightings)
{
    int num_points_of_interest = top_left_points_of_interest->size();
    if (num_points_of_interest == 0)
        return;

    // Preallocate for efficiency
    point_weightings->resize(num_points_of_interest);

    // Put all weightings into the list
    for (auto point = top_left_points_of_interest->begin(); point != top_left_points_of_interest->end(); ++point)
    {
        point_weightings->push_back(point->second);
    }

    // Sort the vector
    sort(point_weightings->begin(), point_weightings->end());

    // Determine the threshold
    int threshold_index = 0.7*point_weightings->size();
    int threshold = (*point_weightings)[threshold_index];

    // Remove points of interest that fall below the threshold
    for (auto point = top_left_points_of_interest->begin(); point != top_left_points_of_interest->end(); ++point)
    {
        if (point->second < threshold)
            top_left_points_of_interest->erase(point);
    }
}

void formClusters(map<pair<int,int>, int > *top_left_points, vector<set<pair<int,int> > >  *known_clusters, bool size_thresholding, vector<double> config_params)
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
    for (unsigned int i = 0; i != labels.size(); ++i)
    {
        auto *tmp_cluster = &(*known_clusters)[labels[i]];
        auto this_point = make_pair(points_to_cluster[i].x, points_to_cluster[i].y);
        tmp_cluster->insert(this_point);
    }

    for (auto cluster = known_clusters->begin(); cluster != known_clusters->end(); ++cluster)
    {
        if (!acceptableClusterSize(&*cluster, config_params[0], config_params[1], config_params[2], config_params[3]))
            known_clusters->erase(cluster);

        // If the last cluster needs to be erased, this will catch it
        if (cluster == known_clusters->end())
            return;
    }

}

bool acceptableClusterSize(set<pair<int,int> > *cluster, double target_size, double focal_length, double altitude, double pixel_size)
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
    double max_apparent_size = target_size * focal_length / altitude;
    int maximum_pixel_size = ceil(max_apparent_size / pixel_size);
    if (max_x - min_x > maximum_pixel_size || max_y - min_y > maximum_pixel_size)
    {
        return false;
    }

    return true;
}

// Note: This function will probably be removed in future
// TEMP
int findPeople(Mat source_image, vector<set<pair<int,int> > >  *known_clusters)
{
    int person_count = 0;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> people_found;

    int height = source_image.size().height;
    int width = source_image.size().width;
    if (width < 64 || height < 128)
    {
        cout << "Image too small to identify people effectively." << endl;
        return 0;
    }

    // Find appropriate clusters and scale them to 64x128
    for (auto cluster : *known_clusters)
    {
        // Scaling
        int min_x = -1;
        int min_y = -1;
        int max_x = -1;
        int max_y = -1;
        for (auto pixel_point = cluster.begin(); pixel_point != cluster.end(); ++pixel_point)
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
        // Skip excessively small clusters
        if (max_x - min_x < 3 || max_y - min_y < 3)
            continue;

        // Find the multiple of 64x128 that this cluster will need to be extended to
        int max_scale = max(ceil((max_x - min_x) / 64.0),
                            ceil((max_y - min_y) / 128.0));
        int dim_req_x =  64 * max_scale;
        int dim_req_y = 128 * max_scale;

        // Don't check an entire image for people
        if (dim_req_x > width || dim_req_y > height)
            continue;

        // The amount of extension this cluster requires to reach a multiple of 64x128
        int extension_x = dim_req_x - (max_x - min_x);
        int extension_y = dim_req_y - (max_y - min_y);

        Point top_left = Point(min_x, min_y),
              bottom_right = Point(max_x, max_y);

        // Default extensions
        top_left.x = min_x - extension_x/2;
        top_left.y = min_y - extension_y/2;
        bottom_right.x = max_x + (extension_x - extension_x/2);
        bottom_right.y = max_y + (extension_y - extension_y/2);

        if (min_x < extension_x / 2)
        {
            // Not enough room to the left
            top_left.x = 0;
            bottom_right.x = max_x + extension_x - min_x;
        }

        if (min_y < extension_y / 2)
        {
            // Not enough room above
            top_left.y = 0;
            bottom_right.y = max_y + extension_y - min_y;
        }

        if (width - max_x < extension_x / 2)
        {
            // Not enough room to the right
            bottom_right.x = width;
            top_left.x = min_x - (extension_x - (width - max_x));
        }

        if (height - max_y < extension_y / 2)
        {
            // Not enough room below
            bottom_right.y = height;
            top_left.y = min_y - (extension_y - (height - max_y));
        }

        Mat source_cluster = source_image(Rect(top_left, bottom_right));
        Mat source_cluster_clone = source_cluster.clone();
        resize(source_cluster_clone, source_cluster_clone, Size(256, 512));


        // // Clusters that are deemed too small to contain people are skipped
        // if (max_x - min_x < 3 || max_y - min_y < 3)
        //     continue;
        // Mat source_cluster = source_image(Rect(Point(min_x, min_y), Point(max_x, max_y)));
        // Mat source_cluster_clone = source_cluster.clone();
        // resize(source_cluster_clone, source_cluster_clone, Size(64,128));

        // Detection
        hog.detectMultiScale(source_cluster_clone, people_found, 0, Size(8,8), Size(0,0), 1.05, 2);
        person_count += people_found.size();
    }


    return person_count;
}

void startDownSampleThreads(Mat source_image, vector<set<pair<int,int> > >  *known_clusters)
{
    // Highlight regions of interest on 4 threads
    int cx = source_image.cols/2;
    int cy = source_image.rows/2;

    Mat src1 = source_image(Rect(0,0,cx + 8,cy + 8));
    auto downSampleHandle1 = bind(downSampleImage, src1, known_clusters, 0, 0);

    Mat src2 = source_image(Rect(cx,0,cx,cy + 8));
    auto downSampleHandle2 = bind(downSampleImage, src2, known_clusters, cx, 0);

    Mat src3 = source_image(Rect(0,cy,cx + 8,cy));
    auto downSampleHandle3 = bind(downSampleImage, src3, known_clusters, 0, cy);

    Mat src4 = source_image(Rect(cx,cy,cx,cy));
    auto downSampleHandle4 = bind(downSampleImage, src4, known_clusters, cx, cy);

    thread downSampleThread1 (downSampleHandle1);
    thread downSampleThread2 (downSampleHandle2);
    thread downSampleThread3 (downSampleHandle3);
    thread downSampleThread4 (downSampleHandle4);

    downSampleThread1.join();
    downSampleThread2.join();
    downSampleThread3.join();
    downSampleThread4.join();
}

void downSampleImage(Mat source_image, vector<set<pair<int,int> > >  *known_clusters, int i_offset, int j_offset)
{
    // Find the cluster rectangles
    vector<Rect> cluster_rectangles;
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
        cluster_rectangles.push_back(Rect(min_x, min_y, max_x-min_x+SQUARE_SIZE, max_y-min_y+SQUARE_SIZE));
    }

    int height = source_image.size().height;
    int width = source_image.size().width;

    for (int i = 0; i < width - SQUARE_SIZE; i += SQUARE_SIZE)
    {
        for (int j = 0; j < height - SQUARE_SIZE; j += SQUARE_SIZE)
        {
            // If this point is enclosed in a known cluster - don't downsample
            bool should_downsample = true;
            Point2i this_point (i+i_offset, j+j_offset);
            for (auto rectangle = cluster_rectangles.begin(); rectangle != cluster_rectangles.end(); ++rectangle)
            {
                if (rectangle->contains(this_point))
                {
                    should_downsample = false;
                    break;
                }
            }

            if (should_downsample)
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

void setChannel(Mat &mat, int channel, unsigned char value)
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
