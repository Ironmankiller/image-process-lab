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

// �����������ȡƽ�����֣��Ż���Ĵ���ʵ����5.16���ļ��ٱȣ���������Intel i7 7700��4��8�̣߳��ٷ��ֲ�����ĳ��߳̿������
// 30%�����ܣ���֪�����϶�˲������иô���ʱ�����ٱ�ӦΪ4+4*0.3=5.2�����Ż�����Ч�ģ��������opencv�ٷ�ʵ�ָ�����24�����ٱ�
// �����뻹�����źܴ�Ĳ�࣬�Ʋ�������ܳ��ڷô��ϣ���˫���Բ�ֵʱ��������Ҫ��ͬ�����е����ݣ����ܵ���cache���ֶ�����������Ż�
// ���������ǽ������ֳ��ʺϴ��뻺���еĴ�СС�飬����ÿ��С�����������̡߳����Ժ���ʹ��MapReduce���в�ֺ͹�Լ�ǿ��еġ�
void resize_personal(cv::Mat& src, cv::Mat& dst, const Size& size ,double fx = 0, double fy = 0) {
	auto src_rows = src.rows;           // ʹ���Զ�������������ڼĴ�����
	auto src_cols = src.cols;
	auto isSizeEmpty = size.empty();
	auto dst_rows = isSizeEmpty ? round(fx * src_rows) : size.height;  // ����������䣬ʹ����ˮ�߳�ӯ
	auto dst_cols = isSizeEmpty ? round(fy * src_cols) : size.width;
	auto ifx = isSizeEmpty ? 1.0 / fx : (double)src_rows / dst_rows;  // �����Ų�����ת���Ӷ�������ѭ���г��ֺ�ʱ�ĳ�������
	auto ify = isSizeEmpty ? 1.0 / fy : (double)src_cols / dst_cols;

	dst = cv::Mat(dst_rows, dst_cols, src.type());
// vs��֧��collapseѭ��չ������������ֻ��չ������ѭ��
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

	// 1. ��ʾԭͼ
	namedWindow("Origin Image", WINDOW_AUTOSIZE);
	imshow("Origin Image", img);

	// 2. ��ʾ�ٷ�resizeʵ�ֽ��
	clock_t s = start();
	resize(img, img_offical, Size(cols, rows));
	printf("<<<<<Official Resize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Official Resized Image", WINDOW_AUTOSIZE);
	imshow("Official Resized Image", img_offical);


	// 3. ��ʾ����resizeʵ�ֽ��
	s = start();

	resize_personal(img, img_personal, Size(cols, rows));
	printf("<<<<<Personal Resize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Personal Resized Image", WINDOW_AUTOSIZE);
	imshow("Personal Resized Image", img_personal);


	Mat diffImg = img_offical - img_personal;
	imshow("Difference of two achievement", diffImg);

	waitKey(0); // ������˳�
	return 0;
}