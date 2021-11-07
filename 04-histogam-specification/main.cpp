#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <map>
#include <string>

#define PI 3.1415926535

using namespace cv;
using namespace std;
clock_t start() {
	return clock();
}

double finish(clock_t start) {
	return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void histogram_draw(Mat& imageGray, string name) {
	const int histSize = 255;   //定义灰度级数量
	float histR[] = { 0,255 };   //定义每个灰度级下取值范围
	const float *histRange = histR;
	Mat imageHist, imageNormalize;

	//计算直方图
	calcHist(&imageGray, 1, 0, Mat(), imageHist, 1, &histSize, &histRange, true, false);
	//直方图归一化到范围[0,histSize]
	normalize(imageHist, imageNormalize, 0, histSize, NORM_MINMAX, -1, Mat());
	//创建直方图画布
	Mat imageShowHist(histSize, histR[1], CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < histSize; i++)
	{
		line(imageShowHist, Point(i, histR[1]), Point(i, histR[1] - cvRound(imageNormalize.at<float>(i))), Scalar(0, 0, 255), 1, 8, 0);
	}
	imshow(name, imageShowHist);


}

void histogram_specification(Mat& src, Mat& dst, float target_hist[]) {
	int height = src.rows;
	int width = src.cols;


	src.copyTo(dst);     //src拷贝到dst

	int hist[256] = { 0 };

	// 对原图进行直方图统计
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			hist[src.at<uchar>(i, j)]++;
		}
	}

	// 提前算出前缀和，避免均衡化时的流依赖，空间换时间
	float prefix_sum[256] = { 0 };     
	prefix_sum[0] = hist[0];
	for (int i = 1; i < 256; i++) {
		prefix_sum[i] = hist[i] + prefix_sum[i - 1];
	}

	// 均衡化，得到均衡化映射S(r)，其中r是原图的灰度级
	int hist_map[256] = { 0 };
	float total = height * width;

	for (int i = 0; i < 256; i++) {
		hist_map[i] = (int)((255.0f * (float)prefix_sum[i] / total) + 0.5);
	}
	//histogram_draw(hist_map, 256, "junhenghua");

	// 对target_hist同理操作，先计算前缀和，然后均衡化得到映射V(z)，其中z是目标图像灰度级
	prefix_sum[0] = target_hist[0];
	for (int i = 1; i < 256; i++) {
		prefix_sum[i] = target_hist[i] + prefix_sum[i - 1];
	}
	int target_hist_map[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		target_hist_map[i] = (int)((255.0f * (float)prefix_sum[i]) + 0.5);
	}



	// 由于原图和目标图在均衡化后的直方图应该是相同的，所以我们要以均衡化后的直方图为桥梁，寻找r和z的映射关系，即V^(-1)(z)到S^(-1)(r)的关系
	// 在寻找这个映射之前，首先要计算出在z和r下，V(z)和S(r)的距离，以便于接下来判断
	float diff_hist[256][256];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			diff_hist[i][j] = fabs(hist_map[i] - target_hist_map[j]);

	// 寻找使得V(z)和S(r)的距离最小的z和r的映射关系
	map<int, int> map;

	for (int i = 0; i < 256; i++) {
		int min = diff_hist[i][0];
		int z = 0;

		for (int j = 1; j < 256; j++) {
			if (min > diff_hist[i][j]) {
				min = diff_hist[i][j];
				z = j;
			}
		}
		map[i] = z;
	}

	// 对原图进行直方图规定化
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < height; i++) {
		uchar *pdata = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++) {
			*(pdata + j) = map[*(pdata + j)];
		}
	}
}

void build_target_hist(float target[]) {
	// 在此构建目标直方图
	//for (int i = 128; i < 256; i++) {
	//	target[i] = 0.5 * 1.0 / 256.0;
	//}
	//for (int i = 0; i < 129; i++) {
	//	target[i] = 1.5 * 1.0 / 256.0;
	//}

	// 先利用均衡化的直方图测试一下代码是否正确
	for (int i = 0; i < 256; i++) {
		target[i] = 1.0 / 256.0;
	}
}


int main(int argc, char** argv)
{

	Mat img, img_gray, offical_hist_equalize, img_personal;
	if (argc == 2)
	{
		img = imread(argv[1], IMREAD_COLOR);
	}
	else {           //default
		img = imread("1.tiff", IMREAD_COLOR);
	}



	if (img.empty())
	{
		fprintf(stderr, "Could not open or find the image");
		return -1;
	}
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	// 1. 显示原图
	imshow("Origin Gray Image", img_gray);
	histogram_draw(img_gray, "Origin Hist");

	// 2. 显示官方直方图均衡化结果
	clock_t s = start();
	equalizeHist(img_gray, offical_hist_equalize);
	printf("<<<<<Offical Histogram Equalize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Offical Histogram Equalize Image", WINDOW_AUTOSIZE);
	imshow("Offical Histogram Equalize Image", offical_hist_equalize);
	histogram_draw(offical_hist_equalize, "Offical Hist");

	// 3. 显示个人直方图规定化实现结果
	float target_hist[256] = { 0 };
	build_target_hist(target_hist);
	s = start();
	histogram_specification(img_gray, img_personal, target_hist);
	printf("<<<<<Personal Histogram Specification Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Personal Histogram Specification Image", WINDOW_AUTOSIZE);
	imshow("Personal Histogram Specification Image", img_personal);
	histogram_draw(img_personal, "Personal Hist");




	waitKey(0); // 任意键退出
	return 0;
}