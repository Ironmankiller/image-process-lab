#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;
//*****************频率域滤波*******************
void fftshift(Mat &src) {
	int cx = src.cols / 2;
	int cy = src.rows / 2;//以下的操作是移动图像  (零频移到中心)
	Mat part1(src, Rect(0, 0, cx, cy));  //元素坐标表示为(cx,cy)
	Mat part2(src, Rect(cx, 0, cx, cy));
	Mat part3(src, Rect(0, cy, cx, cy));
	Mat part4(src, Rect(cx, cy, cx, cy));

	Mat temp;
	part1.copyTo(temp);  //左上与右下交换位置(实部)
	part4.copyTo(part1);
	temp.copyTo(part4);

	part2.copyTo(temp);  //右上与左下交换位置(实部)
	part3.copyTo(part2);
	temp.copyTo(part3);
}

Mat freqfilt(Mat &src, Mat &blur)
{
	//***********************DFT*******************
	Mat plane[] = { src, Mat::zeros(src.size() , CV_32FC1) }; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
	Mat complexIm;
	merge(plane, 2, complexIm);//合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
	dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身

	//***************中心化********************
	fftshift(complexIm);
	split(complexIm, plane);//分离通道（数组分离）


	//*********************显示原图频谱图***********************************
	Mat before;
	magnitude(plane[0], plane[1], before);//获取幅度图像，0通道为实部通道，1为虚部，因为二维傅立叶变换结果是复数
	before += Scalar::all(1);  //傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
	log(before, before);    // float型的灰度空间为[0，1])
	normalize(before, before, 1, 0, NORM_INF);  //归一化便于显示
	imshow("原图像频谱图", before);

	//*****************滤波器函数与DFT结果的乘积****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r); //滤波（实部与滤波器模板对应元素相乘）
	multiply(plane[1], blur, blur_i);//滤波（虚部与滤波器模板对应元素相乘）
	Mat plane1[] = { blur_r, blur_i };

	//*********************显示滤波后的频谱图***********************************
	Mat after;
	magnitude(plane1[0], plane1[1], after);//获取幅度图像，0通道为实部通道，1为虚部，因为二维傅立叶变换结果是复数
	after += Scalar::all(1);  //傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
	log(after, after);    // float型的灰度空间为[0，1])
	normalize(after, after, 1, 0, NORM_INF);  //归一化便于显示
	imshow("滤波后频谱图", after);
	

	//*****************消去中心化*********************************
	merge(plane1, 2, BLUR);//实部与虚部合并
	fftshift(BLUR);


	idft(BLUR, BLUR);    //idft结果也为复数
	split(BLUR, plane);//分离通道，主要获取通道
	magnitude(plane[0], plane[1], plane[0]);  //求幅值(模)
	normalize(plane[0], plane[0], 1, 0, NORM_INF);  //归一化便于显示
	return plane[0];//返回参数
}


//*****************理想低通滤波器***********************
Mat ideal_lbrf_kernel(Mat &src, float sigma)
{
	Mat ideal_low_pass(src.size(), CV_32FC1); //，CV_32FC1
	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double d = sqrt(pow((i - src.rows / 2), 2) + pow((j - src.cols / 2), 2));//分子,计算pow必须为float型
			if (d <= d0) {
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else {
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}
	string name = "理想低通滤波器d0=" + std::to_string(sigma);
	imshow(name, ideal_low_pass);
	return ideal_low_pass;
}

cv::Mat ideal_Low_Pass_Filter(Mat &src, float sigma)
{
	int M = getOptimalDFTSize(src.rows);
	int N = getOptimalDFTSize(src.cols);
	Mat padded;                 //调整图像加速傅里叶变换
	copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1); //将图像转换为float型

	Mat ideal_kernel = ideal_lbrf_kernel(padded, sigma);//理想低通滤波器
	Mat result = freqfilt(padded, ideal_kernel);
	return result;
}


int main(int argc, char *argv[])
{
	const char* filename = argc >= 2 ? argv[1] : "1.tiff";

	Mat input = imread(filename, IMREAD_GRAYSCALE);
	if (input.empty())
		return -1;
	imshow("input", input);//显示原图

	cv::Mat ideal = ideal_Low_Pass_Filter(input, 20);
	ideal = ideal(cv::Rect(0, 0, input.cols, input.rows));
	imshow("滤波结果", ideal);
	waitKey();
	return 0;
}