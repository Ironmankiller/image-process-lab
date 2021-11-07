#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <omp.h>
using namespace cv;
using namespace std;
clock_t start() {
	return clock();
}

double finish(clock_t start) {
	return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void resize_personal_wo_optimize(cv::Mat& src, cv::Mat& dst, const Size& size, double fx = 0, double fy = 0) {
	int dst_rows = round(fx*src.rows);
	int dst_cols = round(fy*src.cols);
	if (!size.empty()) {
		dst_rows = size.height;
		dst_cols = size.width;
		fx = (double)dst_rows / src.rows;
		fy = (double)dst_cols / src.cols;
	}
	dst = cv::Mat(dst_rows, dst_cols, src.type());
	for (int i = 0; i < dst.rows; i++) {
		double index_i = (i + 0.5) / fx - 0.5;
		if (index_i < 0) index_i = 0;
		if (index_i >= src.rows - 1) index_i = src.rows - 1;
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		double u = index_i - i1;
		for (int j = 0; j < dst.cols; j++) {
			double index_j = (j + 0.5) / fy - 0.5;
			if (index_j < 0) index_j = 0;
			if (index_j >= src.cols - 1) index_j = src.cols - 1;
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			double v = index_j - j1;
			dst.at<cv::Vec3b>(i, j)[0] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[0] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[0] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[0] + u * v*src.at<cv::Vec3b>(i2, j2)[0];
			dst.at<cv::Vec3b>(i, j)[1] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[1] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[1] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[1] + u * v*src.at<cv::Vec3b>(i2, j2)[1];
			dst.at<cv::Vec3b>(i, j)[2] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[2] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[2] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[2] + u * v*src.at<cv::Vec3b>(i2, j2)[2];
		}
	}
}

// 经过五次运行取平均后发现，优化后的代码实现了5.16倍的加速比，本机器是Intel i7 7700，4核8线程，官方手册给出的超线程可以提高
// 30%的性能，可知理论上多核并行运行该代码时最大加速比应为4+4*0.3=5.2，该优化是有效的，但是相比opencv官方实现给出的24倍加速比
// 本代码还是有着很大的差距，推测问题可能出在访寸上，做双线性插值时，由于需要不同的两行的数据，可能导致cache出现抖动。具体的优化
// 方案可以是将矩阵拆分成适合存入缓存行的大小小块，并将每个小块分配给各个线程。所以后续使用MapReduce进行拆分和规约是可行的。
void resize_personal(cv::Mat& src, cv::Mat& dst, const Size& size ,double fx = 0, double fy = 0) {
	auto src_rows = src.rows;           // 使用自动变量，方便存在寄存器中
	auto src_cols = src.cols;
	auto isSizeEmpty = size.empty();
	auto dst_rows = isSizeEmpty ? round(fx * src_rows) : size.height;  // 条件传送语句，使得流水线充盈
	auto dst_cols = isSizeEmpty ? round(fy * src_cols) : size.width;
	auto ifx = isSizeEmpty ? 1.0 / fx : (double)src_rows / dst_rows;  // 将缩放参数翻转，从而避免在循环中出现耗时的除法操作
	auto ify = isSizeEmpty ? 1.0 / fy : (double)src_cols / dst_cols;

	dst = cv::Mat(dst_rows, dst_cols, src.type());
// vs不支持collapse循环展开，所以这里只能展开单层循环
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < dst.rows; i++) {
		double index_i = (i + 0.5) * ifx - 0.5;
		//double index_i = (i ) * ifx ;
		if (index_i < 0) index_i = 0;
		if (index_i >= src.rows - 1) index_i = src.rows - 1;
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		double u = index_i - i1;
		for (int j = 0; j < dst.cols; j++) {

			double index_j = (j + 0.5) * ify - 0.5;
			//double index_j = (j ) * ify;
			if (index_j < 0) index_j = 0;
			if (index_j >= src.cols - 1) index_j = src.cols - 1;
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			double v = index_j - j1;
			dst.at<cv::Vec3b>(i, j)[0] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[0] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[0] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[0] + u * v*src.at<cv::Vec3b>(i2, j2)[0];
			dst.at<cv::Vec3b>(i, j)[1] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[1] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[1] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[1] + u * v*src.at<cv::Vec3b>(i2, j2)[1];
			dst.at<cv::Vec3b>(i, j)[2] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[2] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[2] + u * (1 - v)*src.at<cv::Vec3b>(i2, j1)[2] + u * v*src.at<cv::Vec3b>(i2, j2)[2];
		}
	}
}

void help(char **argv) {
	fprintf(stdout, "The program %s scale image to a specific size recive 3 augment \n \
		\t[path] image path\n \
		\t[rows] target image height\n \
		\t[cols] target image width\n", argv[0]);
}

int main(int argc, char** argv)
{
	int rows;
	int cols;
	Mat img, img_offical, img_personal;
	if (argc == 4) {
		help(argv);
		rows = atoi(argv[2]);
		cols = atoi(argv[3]);
		img = imread(argv[1], IMREAD_COLOR);
	}
	else {
		rows = 900;
		cols = 900;
		img = imread("1.png", IMREAD_COLOR);
	}

	
	if (img.empty())
	{
		fprintf(stderr, "Could not open or find the image");
		return -1;
	}

	// 1. 显示原图
	namedWindow("Origin Image", WINDOW_AUTOSIZE);
	imshow("Origin Image", img);

	// 2. 显示官方resize实现结果
	clock_t s = start();
	resize(img, img_offical, Size(cols, rows));
	printf("<<<<<Official Resize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Official Resized Image", WINDOW_AUTOSIZE);
	imshow("Official Resized Image", img_offical);


	// 3. 显示个人resize实现结果
	s = start();

	resize_personal(img, img_personal, Size(cols, rows));
	printf("<<<<<Personal Resize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Personal Resized Image", WINDOW_AUTOSIZE);
	imshow("Personal Resized Image", img_personal);


	Mat diffImg = img_offical - img_personal;
	imshow("Difference of two achievement", diffImg);

	waitKey(0); // 任意键退出
	return 0;
}