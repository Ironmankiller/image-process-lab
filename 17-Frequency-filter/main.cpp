#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

cv::Mat ideal_Low_Pass_Filter(Mat &src, float sigma);
Mat ideal_lbrf_kernel(Mat &scr, float sigma);
Mat freqfilt(Mat &scr, Mat &blur);

int main(int argc, char *argv[])
{
	const char* filename = argc >= 2 ? argv[1] : "../data/lena.jpg";

	Mat input = imread(filename, IMREAD_GRAYSCALE);
	if (input.empty())
		return -1;
	imshow("input", input);//��ʾԭͼ

	cv::Mat ideal = ideal_Low_Pass_Filter(input, 100);
	ideal = ideal(cv::Rect(0, 0, input.cols, input.rows));
	imshow("����", ideal);
	waitKey();
	return 0;
}

//*****************�����ͨ�˲���***********************
Mat ideal_lbrf_kernel(Mat &scr, float sigma)
{
	Mat ideal_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
	float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
			if (d <= d0) {
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else {
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}
	string name = "�����ͨ�˲���d0=" + std::to_string(sigma);
	imshow(name, ideal_low_pass);
	return ideal_low_pass;
}

cv::Mat ideal_Low_Pass_Filter(Mat &src, float sigma)
{
	int M = getOptimalDFTSize(src.rows);
	int N = getOptimalDFTSize(src.cols);
	Mat padded;                 //����ͼ����ٸ���Ҷ�任
	copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1); //��ͼ��ת��Ϊflaot��

	Mat ideal_kernel = ideal_lbrf_kernel(padded, sigma);//�����ͨ�˲���
	Mat result = freqfilt(padded, ideal_kernel);
	return result;
}
//*****************Ƶ�����˲�*******************
Mat freqfilt(Mat &scr, Mat &blur)
{
	//***********************DFT*******************
	Mat plane[] = { scr, Mat::zeros(scr.size() , CV_32FC1) }; //����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
	Mat complexIm;
	merge(plane, 2, complexIm);//�ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
	dft(complexIm, complexIm);//���и���Ҷ�任���������������

	//***************���Ļ�********************
	split(complexIm, plane);//����ͨ����������룩
//    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));//����Ϊʲô&��-2����鿴opencv�ĵ�
//    //��ʵ��Ϊ�˰��к��б��ż�� -2�Ķ�������11111111.......10 ���һλ��0
	int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;//���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
	Mat part1_r(plane[0], Rect(0, 0, cx, cy));  //Ԫ�������ʾΪ(cx,cy)
	Mat part2_r(plane[0], Rect(cx, 0, cx, cy));
	Mat part3_r(plane[0], Rect(0, cy, cx, cy));
	Mat part4_r(plane[0], Rect(cx, cy, cx, cy));

	Mat temp;
	part1_r.copyTo(temp);  //���������½���λ��(ʵ��)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp);  //���������½���λ��(ʵ��)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	Mat part1_i(plane[1], Rect(0, 0, cx, cy));  //Ԫ������(cx,cy)
	Mat part2_i(plane[1], Rect(cx, 0, cx, cy));
	Mat part3_i(plane[1], Rect(0, cy, cx, cy));
	Mat part4_i(plane[1], Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp);  //���������½���λ��(�鲿)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp);  //���������½���λ��(�鲿)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);

	//*****************�˲���������DFT����ĳ˻�****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r); //�˲���ʵ�����˲���ģ���ӦԪ����ˣ�
	multiply(plane[1], blur, blur_i);//�˲����鲿���˲���ģ���ӦԪ����ˣ�
	Mat plane1[] = { blur_r, blur_i };
	merge(plane1, 2, BLUR);//ʵ�����鲿�ϲ�

	  //*********************�õ�ԭͼƵ��ͼ***********************************
	magnitude(plane[0], plane[1], plane[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ�鲿����Ϊ��ά����Ҷ�任����Ǹ���
	plane[0] += Scalar::all(1);  //����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�
	log(plane[0], plane[0]);    // float�͵ĻҶȿռ�Ϊ[0��1])
	normalize(plane[0], plane[0], 1, 0, NORM_INF);  //��һ��������ʾ
    imshow("ԭͼ��Ƶ��ͼ",plane[0]);

	idft(BLUR, BLUR);    //idft���ҲΪ����
	split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��
	magnitude(plane[0], plane[1], plane[0]);  //���ֵ(ģ)
	normalize(plane[0], plane[0], 1, 0, NORM_INF);  //��һ��������ʾ
	return plane[0];//���ز���
}
