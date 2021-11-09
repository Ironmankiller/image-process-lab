#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;
//*****************Ƶ�����˲�*******************
void fftshift(Mat &src) {
	int cx = src.cols / 2;
	int cy = src.rows / 2;//���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
	Mat part1(src, Rect(0, 0, cx, cy));  //Ԫ�������ʾΪ(cx,cy)
	Mat part2(src, Rect(cx, 0, cx, cy));
	Mat part3(src, Rect(0, cy, cx, cy));
	Mat part4(src, Rect(cx, cy, cx, cy));

	Mat temp;
	part1.copyTo(temp);  //���������½���λ��(ʵ��)
	part4.copyTo(part1);
	temp.copyTo(part4);

	part2.copyTo(temp);  //���������½���λ��(ʵ��)
	part3.copyTo(part2);
	temp.copyTo(part3);
}

Mat freqfilt(Mat &src, Mat &blur)
{
	//***********************DFT*******************
	Mat plane[] = { src, Mat::zeros(src.size() , CV_32FC1) }; //����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
	Mat complexIm;
	merge(plane, 2, complexIm);//�ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
	dft(complexIm, complexIm);//���и���Ҷ�任���������������

	//***************���Ļ�********************
	fftshift(complexIm);
	split(complexIm, plane);//����ͨ����������룩


	//*********************��ʾԭͼƵ��ͼ***********************************
	Mat before;
	magnitude(plane[0], plane[1], before);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ�鲿����Ϊ��ά����Ҷ�任����Ǹ���
	before += Scalar::all(1);  //����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�
	log(before, before);    // float�͵ĻҶȿռ�Ϊ[0��1])
	normalize(before, before, 1, 0, NORM_INF);  //��һ��������ʾ
	imshow("ԭͼ��Ƶ��ͼ", before);

	//*****************�˲���������DFT����ĳ˻�****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r); //�˲���ʵ�����˲���ģ���ӦԪ����ˣ�
	multiply(plane[1], blur, blur_i);//�˲����鲿���˲���ģ���ӦԪ����ˣ�
	Mat plane1[] = { blur_r, blur_i };

	//*********************��ʾ�˲����Ƶ��ͼ***********************************
	Mat after;
	magnitude(plane1[0], plane1[1], after);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ�鲿����Ϊ��ά����Ҷ�任����Ǹ���
	after += Scalar::all(1);  //����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�
	log(after, after);    // float�͵ĻҶȿռ�Ϊ[0��1])
	normalize(after, after, 1, 0, NORM_INF);  //��һ��������ʾ
	imshow("�˲���Ƶ��ͼ", after);
	

	//*****************��ȥ���Ļ�*********************************
	merge(plane1, 2, BLUR);//ʵ�����鲿�ϲ�
	fftshift(BLUR);


	idft(BLUR, BLUR);    //idft���ҲΪ����
	split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��
	magnitude(plane[0], plane[1], plane[0]);  //���ֵ(ģ)
	normalize(plane[0], plane[0], 1, 0, NORM_INF);  //��һ��������ʾ
	return plane[0];//���ز���
}


//*****************�����ͨ�˲���***********************
Mat ideal_lbrf_kernel(Mat &src, float sigma)
{
	Mat ideal_low_pass(src.size(), CV_32FC1); //��CV_32FC1
	float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double d = sqrt(pow((i - src.rows / 2), 2) + pow((j - src.cols / 2), 2));//����,����pow����Ϊfloat��
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
	padded.convertTo(padded, CV_32FC1); //��ͼ��ת��Ϊfloat��

	Mat ideal_kernel = ideal_lbrf_kernel(padded, sigma);//�����ͨ�˲���
	Mat result = freqfilt(padded, ideal_kernel);
	return result;
}


int main(int argc, char *argv[])
{
	const char* filename = argc >= 2 ? argv[1] : "1.tiff";

	Mat input = imread(filename, IMREAD_GRAYSCALE);
	if (input.empty())
		return -1;
	imshow("input", input);//��ʾԭͼ

	cv::Mat ideal = ideal_Low_Pass_Filter(input, 20);
	ideal = ideal(cv::Rect(0, 0, input.cols, input.rows));
	imshow("�˲����", ideal);
	waitKey();
	return 0;
}