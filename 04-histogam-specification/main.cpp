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

#define PI 3.1415926535

using namespace cv;
using namespace std;
clock_t start() {
	return clock();
}

double finish(clock_t start) {
	return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void histogram_specification(Mat& src, Mat& dst, float target_hist[]) {
	int height = src.rows;
	int width = src.cols;


	src.copyTo(dst);     //src������dst

	int hist[256] = { 0 };

	// ��ԭͼ����ֱ��ͼͳ��
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			hist[src.at<uchar>(i, j)]++;
		}
	}

	// ��ǰ���ǰ׺�ͣ�������⻯ʱ�����������ռ任ʱ��
	float prefix_sum[256] = { 0 };     
	prefix_sum[0] = hist[0];
	for (int i = 1; i < 256; i++) {
		prefix_sum[i] = hist[i] + prefix_sum[i - 1];
	}

	// ���⻯���õ����⻯ӳ��S(r)������r��ԭͼ�ĻҶȼ�
	int hist_map[256] = { 0 };
	int total = height * width;
	for (int i = 0; i < 256; i++) {
		hist_map[i] = (int)((255.0f * (float)prefix_sum[i] / total) + 0.5);
	}

	// ��target_histͬ��������ȼ���ǰ׺�ͣ�Ȼ����⻯�õ�ӳ��V(z)������z��Ŀ��ͼ��Ҷȼ�
	prefix_sum[0] = target_hist[0];
	for (int i = 1; i < 256; i++) {
		prefix_sum[i] = target_hist[i] + prefix_sum[i - 1];
	}
	int target_hist_map[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		target_hist_map[i] = (int)((255.0f * (float)prefix_sum[i]) + 0.5);
	}



	// ����ԭͼ��Ŀ��ͼ�ھ��⻯���ֱ��ͼӦ������ͬ�ģ���������Ҫ�Ծ��⻯���ֱ��ͼΪ������Ѱ��r��z��ӳ���ϵ����V^(-1)(z)��S^(-1)(r)�Ĺ�ϵ
	// ��Ѱ�����ӳ��֮ǰ������Ҫ�������z��r�£�V(z)��S(r)�ľ��룬�Ա��ڽ������ж�
	float diff_hist[256][256];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			diff_hist[i][j] = fabs(hist_map[i] - target_hist_map[j]);

	// Ѱ��ʹ��V(z)��S(r)�ľ�����С��z��r��ӳ���ϵ
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


	// ��ԭͼ����ֱ��ͼ�涨��
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < height; i++) {
		uchar *pdata = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++) {
			*(pdata + j) = map[*(pdata + j)];
		}
	}
}

void build_target_hist(float target[]) {
	// �ڴ˹���Ŀ��ֱ��ͼ
	for (int i = 128; i < 256; i++) {
		target[i] = 0.5 * 1.0 / 256.0;
	}
	for (int i = 0; i < 129; i++) {
		target[i] = 1.5 * 1.0 / 256.0;
	}

	//// �����þ��⻯��ֱ��ͼ����һ�´����Ƿ���ȷ
	//for (int i = 0; i < 256; i++) {
	//	target[i] = 1.0 / 256.0;
	//}
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr,
			"Usage: %s\n"
			"\t[imge path]\n",
			argv[0]);
		return -1;
	}
	Mat img, img_gray, offical_hist_equalize,img_personal;

	img = imread(argv[1], IMREAD_COLOR);

	if (img.empty())
	{
		fprintf(stderr, "Could not open or find the image");
		return -1;
	}
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	// 1. ��ʾԭͼ
	namedWindow("Origin Image", WINDOW_AUTOSIZE);
	imshow("Origin Gray Image", img_gray);

	// 2. ��ʾ�ٷ�ֱ��ͼ���⻯���
	clock_t s = start();
	equalizeHist(img_gray, offical_hist_equalize);
	printf("<<<<<Offical Histogram Equalize Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Offical Histogram Equalize Image", WINDOW_AUTOSIZE);
	imshow("Offical Histogram Equalize Image", offical_hist_equalize);

	// 3. ��ʾ����ֱ��ͼ�涨��ʵ�ֽ��
	float target_hist[256] = { 0 };
	build_target_hist(target_hist);
	s = start();
	histogram_specification(img_gray, img_personal, target_hist);
	printf("<<<<<Personal Histogram Specification Duration: %lf>>>>>>>\n", finish(s));
	namedWindow("Personal Histogram Specification Image", WINDOW_AUTOSIZE);
	imshow("Personal Histogram Specification Image", img_personal);




	waitKey(0); // ������˳�
	return 0;
}