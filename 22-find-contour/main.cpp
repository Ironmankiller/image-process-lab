#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <map>
#include <vector>

#define PI 3.1415926535

using namespace cv;
using namespace std;
clock_t start() {
	return clock();
}

double finish(clock_t start) {
	return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void find_contour_offical(const Mat& img, vector<Point>& cont) 
{
	
	vector<vector<Point> > _contoursQuery;
	clock_t s = start();
	findContours(img, _contoursQuery, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	printf("<<<<<Offical Findcontour Duration: %lf>>>>>>>\n", finish(s));
	for (size_t border = 0; border < _contoursQuery.size(); border++)
	{
		for (size_t p = 0; p < _contoursQuery[border].size(); p++)
		{
			cont.push_back(_contoursQuery[border][p]);
		}
	}

}

void find_contour_personal(Mat& img, vector<Point>& cont) 
{
	clock_t s = start();

	int h = img.rows;
	int w = img.cols;

	int start_x = -1;
	int start_y = -1;
	bool start_flag = false;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (img.at<uchar>(i, j) == 255) {
				start_x = i;
				start_y = j;
				start_flag = true;
				break;
			}
		}
		if (start_flag) {
			break;
		}
	}
	if (!start_flag) {
		return;
	}
	const int neibor_len = 8;
	//int neibor[neibor_len][2] = { {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1} };  // 逆时针链码，两两一组
	//int temp = 5;   // 链码值，从5号位，左下方开始

	int neibor[neibor_len][2] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} }; // 顺时针链码
	int temp = 0;   // 链码值，从0号位，正右开始
	cont.push_back(Point(start_y, start_x));

	int current_x = start_x; 
	int current_y = start_y;

	int neibor_x = current_x + neibor[temp][0];
	int neibor_y = current_y + neibor[temp][1];

	do {
		bool find_contour_point = false;
		int num = 0;        // 避免无限循环
		while (!find_contour_point && num++ != neibor_len) {
			if (neibor_x >= 0 && neibor_x < h && neibor_y >=0 && neibor_y < w && img.at<uchar>(neibor_x, neibor_y) == 255) {
				current_x = neibor_x;
				current_y = neibor_y;
				cont.push_back(Point(neibor_y, neibor_x));
				temp = temp - 2;
				if (temp < 0) {
					temp += neibor_len;
				}
				find_contour_point = true;
			}
			else {
				temp++;
				if (temp >= neibor_len) {
					temp -= neibor_len;
				}
			}
			neibor_x = current_x + neibor[temp][0];
			neibor_y = current_y + neibor[temp][1];
		}
		if (!find_contour_point) {
			cont.clear();
			return;
		}
	} while (current_x != start_x || current_y != start_y);

	printf("<<<<<Personal Findcontour Duration: %lf>>>>>>>\n", finish(s));
}

void imshow_contour(Mat& img, vector<Point>& cont, bool isOffical) 
{
	string name;
	Mat to_show;
	if (isOffical) {
		name = "Offical Findcontour Result";
	}
	else {
		name = "Personal Findcontour Result";
	}
	cvtColor(img, to_show, COLOR_GRAY2BGR);
	namedWindow(name, WINDOW_AUTOSIZE);
	for (auto i = 0; i < cont.size(); i++) {
		to_show.at<Vec3b>(cont[i].y, cont[i].x)[0] = 0;
		to_show.at<Vec3b>(cont[i].y, cont[i].x)[1] = 0;
		to_show.at<Vec3b>(cont[i].y, cont[i].x)[2] = 255;
		if (!isOffical) {
			imshow(name, to_show);
			waitKey(1);
		}

	}
	imshow(name, to_show);
}


int main(int argc, char** argv)
{
	Mat img;
	if (argc == 2)
	{
		img = imread(argv[1], IMREAD_GRAYSCALE);
	}
	else {
		img = imread("2.png", IMREAD_GRAYSCALE);
	}


	if (img.empty())
	{
		fprintf(stderr, "Could not open or find the image");
		return -1;
	}
	// 测试轮廓位于边界的情况，测试通过
	//img = img(Rect(140, 40, 140, 260));
	threshold(img, img, 50, 255, THRESH_BINARY);

	// 1. 显示原图
	namedWindow("Origin Binary Image", WINDOW_AUTOSIZE);
	imshow("Origin Binary Image", img);

	// 2. 显示官方轮廓跟踪结果
	vector<Point> contour_offical;
	find_contour_offical(img, contour_offical);
	imshow_contour(img, contour_offical, 1);


	// 3. 显示个人实现的轮廓跟踪结果
	vector<Point> contour_personal;
	find_contour_personal(img, contour_personal);
	imshow_contour(img, contour_personal, 0);

	waitKey(0); // 任意键退出
	return 0;
}