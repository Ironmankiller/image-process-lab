#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define PI 3.1415926535

using namespace cv;
using namespace std;
clock_t start() {
	return clock();
}

double finish(clock_t start) {
	return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void rotate_personal(Mat& src, Mat& dst, const double& angle) {
	auto theta = (double)(angle * PI / 180.0);
	auto sina = sin(theta);
	auto cosa = cos(theta);

	// 将原图中心点作为旋转坐标
	auto centerX = src.cols / 2;
	auto centerY = src.rows / 2;

	// 新坐标下原图的四个顶点坐标
	// 1 2
	// 3 4
	auto srcX1 = -centerX;
	auto srcY1 = centerY;
	auto srcX2 = centerX;
	auto srcY2 = centerY;
	auto srcX3 = -centerX;
	auto srcY3 = -centerY;
	auto srcX4 = centerX;
	auto srcY4 = -centerY;

	// 旋转后的四个点的坐标
	// 2 4
	// 1 3
	auto dstX1 = (srcX1 * cosa - srcY1 * sina);
	auto dstY1 = (srcX1 * sina + srcY1 * cosa);
	auto dstX2 = (srcX2 * cosa - srcY2 * sina);
	auto dstY2 = (srcX2 * sina + srcY2 * cosa);
	auto dstX3 = (srcX3 * cosa - srcY3 * sina);
	auto dstY3 = (srcX3 * sina + srcY3 * cosa);
	auto dstX4 = (srcX4 * cosa - srcY4 * sina);
	auto dstY4 = (srcX4 * sina + srcY4 * cosa);

	// 计算最大宽度和高度，做为变换后图像的大小
	auto dst_width = (int)max(fabs(dstX1 - dstX4), fabs(dstX2 - dstX3)) + 1;
	auto dst_height = (int)max(fabs(dstY1 - dstY4), fabs(dstY2 - dstY3)) + 1;

	dst = cv::Mat(dst_height, dst_width, src.type());

	auto new_centerX = dst_width / 2;
	auto new_centerY = dst_height / 2;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < dst_height; i++) {
		for (int j = 0; j < dst_width; j++) {
			// 将目标图像的坐标后向映射回原图，坐标远点位于左上角
			double y0 = centerY + (j - new_centerX) * sina + (i - new_centerY) * cosa;
			double x0 = centerX + (j - new_centerX) * cosa - (i - new_centerY) * sina;

			if (y0 < 0 || y0 >= src.rows || x0 < 0 || x0 >= src.cols) {
				continue;
			}

			int y1 = floor(y0);
			int y2 = ceil(y0);
			if (y2 == src.rows) {
				y2--;
			}
			double u = y0 - y1;

			int x1 = floor(x0);
			int x2 = ceil(x0);
			if (x2 == src.cols) {
				x2--;
			}
			double v = x0 - x1;



			if (src.channels() == 1) {
				// 灰度图像
				dst.at<uchar>(i, j) = (1 - u)*(1 - v)*src.at<uchar>(y1, x1) + (1 - u)*v*src.at<uchar>(y1, x2) + u * (1 - v)*src.at<uchar>(y2, x2) + u * v*src.at<uchar>(y2, x2);
			}
			else {
				// 彩色图像
				dst.at<cv::Vec3b>(i, j)[0] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(y1, x1)[0] + (1 - u)*v*src.at<cv::Vec3b>(y1, x2)[0] + u * (1 - v)*src.at<cv::Vec3b>(y2, x1)[0] + u * v*src.at<cv::Vec3b>(y2, x2)[0];
				dst.at<cv::Vec3b>(i, j)[1] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(y1, x1)[1] + (1 - u)*v*src.at<cv::Vec3b>(y1, x2)[1] + u * (1 - v)*src.at<cv::Vec3b>(y2, x1)[1] + u * v*src.at<cv::Vec3b>(y2, x2)[1];
				dst.at<cv::Vec3b>(i, j)[2] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(y1, x1)[2] + (1 - u)*v*src.at<cv::Vec3b>(y1, x2)[2] + u * (1 - v)*src.at<cv::Vec3b>(y2, x1)[2] + u * v*src.at<cv::Vec3b>(y2, x2)[2];
			}

		}
	}

}


int main(int argc, char** argv)
{
	Mat img, img_offical, img_personal;
	double angle;
	
	if (argc == 3) {
		angle = atoi(argv[2]);
		img = imread(argv[1], IMREAD_COLOR);
	}
	else {           //default
		angle = 30;
		img = imread("../../1.tif");
	}

	if (img.empty())
	{
		fprintf(stderr, "Could not open or find the image");
		return -1;
	}

	// 1. 显示原图
	namedWindow("Origin Image", WINDOW_AUTOSIZE);
	imshow("Origin Image", img);

	// 2. 显示个人rotate实现结果
	clock_t s = start();
	rotate_personal(img, img_personal, angle);
	
	printf("<<<<<Personal Rotated Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Personal Rotated Image", WINDOW_AUTOSIZE);
	imshow("Personal Rotated Image", img_personal);




	waitKey(0); // 任意键退出
	return 0;
}