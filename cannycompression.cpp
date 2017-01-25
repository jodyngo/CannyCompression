#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <string>
#include <thread>
#include <functional>

#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define CANNY_THRESHOLD 90
#define CANNY_RATIO 3
#define SQUARE_SIZE 8
#define ROI_SIZE_LIMIT 200

Mat cannyEdgeDetection(Mat input_matrix);
void drawRegionsOfInterest(Mat proc_matrix, Mat disp_matrix);
void processSection(Mat proc_matrix, Mat disp_matrix, set<pair<int,int> > *top_left_points_of_interest, int i_offset, int j_offset);
int totalVals(Mat &matrix, int chanSum = -1);
void formClusters(set<pair<int, int> > *top_left_points, vector<set<pair<int,int> > >  *known_clusters);
void updateCluster(pair<int,int> point, set<pair<int,int> > *cluster);
void downSampleMat(Mat input_matrix);
void setChannel(Mat &mat, unsigned int channel, unsigned char value);

// Input: filename compression_level display
int main(int argc, char** argv)
{
    // Load the image and compression level
    if ( argc != 3 )
    {
        cout << "Usage: CannyCompression <file name> <compression level>." << endl;
        return -1;
    }
    Mat source_image = imread(argv[1]);
    if (!source_image.data)
    {
        cout << "No image data." << endl;
        return -1;
    }
    int compression_level = atoi(argv[2]);

    // Timing
    double t = (double)getTickCount();

    // Pad the source image so that the square size fits without missing spots
    int pad_down = source_image.size().height % SQUARE_SIZE;
    int pad_right = source_image.size().width % SQUARE_SIZE;
    copyMakeBorder(source_image, source_image, 0, pad_down, 0, pad_right, BORDER_REPLICATE);

    // Create another matrix of the same size and type for the output
    // of the Canny filter
    Mat processed_image;
    processed_image.create(source_image.size(), source_image.type());

    // Convert the source image to greyscale
    Mat source_grey;
    cvtColor(source_image, source_grey, CV_BGR2GRAY);

    // Perform the edge-detection
    processed_image = cannyEdgeDetection(source_grey);

    drawRegionsOfInterest(processed_image, source_image);

    // Compress and write image
    vector<int> compVec = {CV_IMWRITE_JPEG_QUALITY, compression_level};
    string image_name (argv[1]);
    int i = image_name.find(".");
    image_name.insert(i, "_COMPRESSED");
    imwrite(image_name, source_image, compVec);

    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "Processing time: " << t << " s."<< std::endl;

    namedWindow("Processed", WINDOW_AUTOSIZE);
    imshow("Processed", source_image);

    waitKey(0);
    return 0;
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
void drawRegionsOfInterest(Mat proc_matrix, Mat disp_matrix)
{
    // Clusters of edges are grouped to be highlighted
    vector<set<pair<int,int> > >  known_clusters;
    set<pair<int,int> > top_left_points_of_interest;

    // Highlight regions of interest on 4 threads
    int r = proc_matrix.rows;
    int c = proc_matrix.cols;
    cout << r << c << endl;
    int adj_r = 0;
    int adj_c = 0;
    if (r % 2 == 0)
        adj_r = 1;
    if (c % 2 == 0)
        adj_c = 1;

    Mat proc1 = proc_matrix(Rect(0,0,adj_c+c/2,adj_r+r/2));
    Mat src1 = disp_matrix(Rect(0,0,c/2,r/2));
    auto regionHandle1 = bind(processSection, proc1, src1, &top_left_points_of_interest, 0, 0);

    Mat proc2 = proc_matrix(Rect(c/2,0,c/2,adj_r+r/2));
    Mat src2 = disp_matrix(Rect(c/2,0,c/2,r/2));
    auto regionHandle2 = bind(processSection, proc2, src2, &top_left_points_of_interest, c/2, 0);

    Mat proc3 = proc_matrix(Rect(0,r/2,adj_c+c/2,r/2));
    Mat src3 = disp_matrix(Rect(0,r/2,c/2,r/2));
    auto regionHandle3 = bind(processSection, proc3, src3, &top_left_points_of_interest, 0, r/2);

    Mat proc4 = proc_matrix(Rect(c/2,r/2,c/2,r/2));
    Mat src4 = disp_matrix(Rect(c/2,r/2,c/2,r/2));
    auto regionHandle4 = bind(processSection, proc4, src4, &top_left_points_of_interest, c/2, r/2);

    thread regionThread1 (regionHandle1);
    thread regionThread2 (regionHandle2);
    thread regionThread3 (regionHandle3);
    thread regionThread4 (regionHandle4);


    regionThread1.join();
    regionThread2.join();
    regionThread3.join();
    regionThread4.join();

    formClusters(&top_left_points_of_interest, &known_clusters);

    // Now draw rectangles around each cluster
    cout << "Number of clusters found: " << known_clusters.size() << endl;
    for (auto cluster = known_clusters.begin(); cluster != known_clusters.end(); ++cluster)
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
            rectangle(disp_matrix, Point(min_x, min_y), Point(max_x, max_y), Scalar(9,9,179), 1);
    }
}

void processSection(Mat proc_matrix, Mat disp_matrix, set<pair<int,int> > *top_left_points_of_interest, int i_offset, int j_offset)
{
    int height = proc_matrix.size().height;
    int width = proc_matrix.size().width;
    int total = totalVals(proc_matrix);

    for (int i = 0; i < width - SQUARE_SIZE; i += SQUARE_SIZE)
    {
        for (int j = 0; j < height - SQUARE_SIZE; j += SQUARE_SIZE)
        {
            // Process 8x8 sections
            Mat section = proc_matrix(Rect(i,j,SQUARE_SIZE,SQUARE_SIZE));
            int sub_total = totalVals(section);
            // TODO: Test different cutoff values from 10
            int section_weight;
            if (total > 0)
                section_weight = (1000/SQUARE_SIZE)*(255*sub_total)/total;
            else
                section_weight = 0;
            if (section_weight > 255)
                section_weight = 255;
            if (section_weight > 10) // Region of interest
            {
                top_left_points_of_interest->insert({i+i_offset,j+j_offset});
            }
            else
            {
                Mat source_section = disp_matrix(Rect(i,j,SQUARE_SIZE,SQUARE_SIZE));
                downSampleMat(source_section);
            }
        }
    }
}

int totalVals(Mat &matrix, int chanSum)
{
    int sum = 0;
    const int channels = matrix.channels();
    switch(channels)
    {
    case 1:
        {
            MatIterator_<uchar> it, end;
            for( it = matrix.begin<uchar>(), end = matrix.end<uchar>(); it != end; ++it)
                sum += *it;
        }
    case 3:
        {
            MatIterator_<Vec3b> it, end;
            for( it = matrix.begin<Vec3b>(), end = matrix.end<Vec3b>(); it != end; ++it)
            {
                switch (chanSum)
                {
                    case -1: sum += (*it)[0];
                             sum += (*it)[1];
                             sum += (*it)[2];
                             break;
                    case 0: sum += (*it)[0];
                            break;
                    case 1: sum += (*it)[1];
                            break;
                    case 2: sum += (*it)[2];
                            break;
                }

            }
        }
    }
    return sum;
}

void formClusters(set<pair<int, int> > *top_left_points, vector<set<pair<int,int> > >  *known_clusters)
{
    // Select a point from top_left_points
    // Calculate all its other edges
    // Check if they are in top_left_points
    // If so, remove them from top_left_points and calculate their edges
    // Add every point found to known_clusters
    while (top_left_points->size() != 0)
    {
        // Extract the next starting point from top_left_points
        auto seed_point = top_left_points->begin();
        int x = seed_point->first;
        int y = seed_point->second;
        top_left_points->erase(seed_point);

        // Calculate all other points on the square and record them
        set<pair<int,int> > seed_square;
        seed_square.insert({x,y});
        seed_square.insert({x + SQUARE_SIZE,y});
        seed_square.insert({x,y + SQUARE_SIZE});
        seed_square.insert({x + SQUARE_SIZE,y + SQUARE_SIZE});
        known_clusters->push_back(seed_square);

        set<pair<int, int> > points_to_check;
        points_to_check.insert({x + SQUARE_SIZE,y});
        points_to_check.insert({x,y + SQUARE_SIZE});
        points_to_check.insert({x + SQUARE_SIZE,y + SQUARE_SIZE});

        // Search for connecting squares
        while (points_to_check.size() != 0)
        {
            auto this_point = points_to_check.begin();
            if (top_left_points->find(*this_point) != top_left_points->end())
            {
                top_left_points->erase(*this_point);
                // A connected square has been found
                x = this_point->first;
                y = this_point->second;

                // Check down and to the right
                points_to_check.insert({x + SQUARE_SIZE,y});
                points_to_check.insert({x,y + SQUARE_SIZE});
                points_to_check.insert({x + SQUARE_SIZE,y + SQUARE_SIZE});
                // Check down and to the left
                points_to_check.insert({x - SQUARE_SIZE,y});
                points_to_check.insert({x - SQUARE_SIZE,y + SQUARE_SIZE});
                // Check up and to the left
                points_to_check.insert({x,y - SQUARE_SIZE});
                points_to_check.insert({x - SQUARE_SIZE,y - SQUARE_SIZE});
                // Check up and to the right
                points_to_check.insert({x + SQUARE_SIZE,y - SQUARE_SIZE});

                updateCluster(*this_point, &known_clusters->back());
            }
            points_to_check.erase(this_point);
        }
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

void downSampleMat(Mat input_matrix)
{
    int height = input_matrix.size().height;
    int width = input_matrix.size().width;

    long totalB = totalVals(input_matrix, 0);
    long totalG = totalVals(input_matrix, 1);
    long totalR = totalVals(input_matrix, 2);

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
