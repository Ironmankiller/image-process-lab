#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>  
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

//******************灰度转换函数*************************
//第一个参数image输入的彩色RGB图像；
//第二个参数imageGray是转换后输出的灰度图像；
//*************************************************************
void ConvertRGB2GRAY(const Mat& image, Mat& imageGray);


//******************高斯卷积核生成函数*************************
//第一个参数gaus是一个指向含有N个double类型数组的指针；
//第二个参数size是高斯卷积核的尺寸大小；
//第三个参数sigma是卷积核的标准差
//*************************************************************
void GetGaussianKernel(double** gaus, const int size, const double sigma);

//******************高斯滤波*************************
//第一个参数imageSource是待滤波原始图像；
//第二个参数imageGaussian是滤波后输出图像；
//第三个参数gaus是一个指向含有N个double类型数组的指针；
//第四个参数size是滤波核的尺寸
//*************************************************************
void GaussianFilter(const Mat imageSource, Mat& imageGaussian, double** gaus, int size);

//******************Sobel算子计算梯度和方向********************
//第一个参数imageSourc原始灰度图像；
//第二个参数imageSobelX是X方向梯度图像；
//第三个参数imageSobelY是Y方向梯度图像；
//第四个参数pointDrection是梯度方向数组指针
//*************************************************************
void SobelGradDirction(const Mat imageSource, Mat& imageSobelX, Mat& imageSobelY, double*& pointDrection);

//******************计算Sobel的X和Y方向梯度幅值*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第三个参数SobelAmpXY是输出的X、Y方向梯度图像幅值
//*************************************************************
void SobelAmplitude(const Mat imageGradX, const Mat imageGradY, Mat& SobelAmpXY);

//******************局部极大值抑制*************************
//第一个参数imageInput输入的Sobel梯度图像；
//第二个参数imageOutPut是输出的局部极大值抑制图像；
//第三个参数pointDrection是图像上每个点的梯度方向数组指针
//*************************************************************
void LocalMaxValue(const Mat imageInput, Mat& imageOutput, double* pointDrection);

//******************双阈值处理*************************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//******************************************************
void DoubleThreshold(Mat& imageIput, double lowThreshold, double highThreshold);

//******************双阈值中间像素连接处理*********************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//*************************************************************
void DoubleThresholdLink(Mat& imageInput, double lowThreshold, double highThreshold);

Mat imageSource;
Mat imageGray;
Mat imageGaussian;

int main(int argc, char* argv[])
{
	imageSource = imread("4.png");  //读入RGB图像
	imshow("RGB Image", imageSource);
	ConvertRGB2GRAY(imageSource, imageGray); //RGB转换为灰度图
	imshow("Gray Image", imageGray);
	int size = 5; //定义卷积核大小
	double** gaus = new double* [size];  //卷积核数组
	for (int i = 0; i < size; i++)
	{
		gaus[i] = new double[size];  //动态生成矩阵
	}
	GetGaussianKernel(gaus, 5, 1); //生成5*5 大小高斯卷积核，Sigma=1；
	imageGaussian = Mat::zeros(imageGray.size(), CV_8UC1);
	GaussianFilter(imageGray, imageGaussian, gaus, 5);  //高斯滤波
	imshow("Gaussian Image", imageGaussian);
	Mat imageSobelY;
	Mat imageSobelX;
	double* pointDirection = new double[(imageSobelX.cols - 1) * (imageSobelX.rows - 1)];  //定义梯度方向角数组
	SobelGradDirction(imageGaussian, imageSobelX, imageSobelY, pointDirection);  //计算X、Y方向梯度和方向角
	imshow("Sobel Y", imageSobelY);
	imshow("Sobel X", imageSobelX);
	Mat SobelGradAmpl;
	SobelAmplitude(imageSobelX, imageSobelY, SobelGradAmpl);   //计算X、Y方向梯度融合幅值
	imshow("Soble XYRange", SobelGradAmpl);
	Mat imageLocalMax;
	LocalMaxValue(SobelGradAmpl, imageLocalMax, pointDirection);  //局部非极大值抑制
	imshow("Non-Maximum Image", imageLocalMax);
	Mat cannyImage;
	cannyImage = Mat::zeros(imageLocalMax.size(), CV_8UC1);
	DoubleThreshold(imageLocalMax, 90, 160);        //双阈值处理
	imshow("Double Threshold Image", imageLocalMax);
	DoubleThresholdLink(imageLocalMax, 90, 160);   //双阈值中间阈值滤除及连接
	imshow("Canny Image", imageLocalMax);
	waitKey();
	system("pause");
	return 0;
}

//******************高斯卷积核生成函数*************************
//第一个参数gaus是一个指向含有N个double类型数组的指针；
//第二个参数size是高斯卷积核的尺寸大小；
//第三个参数sigma是卷积核的标准差
//*************************************************************
void GetGaussianKernel(double** gaus, const int size, const double sigma)
{
	const double PI = 4.0 * atan(1.0); //圆周率π赋值
	int center = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] = (1 / (2 * PI * sigma * sigma)) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
			sum += gaus[i][j];
		}
	}
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] /= sum;
			cout << gaus[i][j] << "  ";
		}
		cout << endl << endl;
	}
	return;
}

//******************灰度转换函数*************************
//第一个参数image输入的彩色RGB图像；
//第二个参数imageGray是转换后输出的灰度图像；
//*************************************************************
void ConvertRGB2GRAY(const Mat& image, Mat& imageGray)
{
	if (!image.data || image.channels() != 3)
	{
		return;
	}
	imageGray = Mat::zeros(image.size(), CV_8UC1);
	uchar* pointImage = image.data;
	uchar* pointImageGray = imageGray.data;
	int stepImage = image.step;
	int stepImageGray = imageGray.step;
	for (int i = 0; i < imageGray.rows; i++)
	{
		for (int j = 0; j < imageGray.cols; j++)
		{
			pointImageGray[i * stepImageGray + j] = 0.114 * pointImage[i * stepImage + 3 * j] + 0.587 * pointImage[i * stepImage + 3 * j + 1] + 0.299 * pointImage[i * stepImage + 3 * j + 2];
		}
	}
}

//******************高斯滤波*************************
//第一个参数imageSource是待滤波原始图像；
//第二个参数imageGaussian是滤波后输出图像；
//第三个参数gaus是一个指向含有N个double类型数组的指针；
//第四个参数size是滤波核的尺寸
//*************************************************************
void GaussianFilter(const Mat imageSource, Mat& imageGaussian, double** gaus, int size)
{
	imageGaussian = Mat::zeros(imageSource.size(), CV_8UC1);
	if (!imageSource.data || imageSource.channels() != 1)
	{
		return;
	}
	double gausArray[100];
	for (int i = 0; i < size * size; i++)
	{
		gausArray[i] = 0;  //赋初值，空间分配
	}
	int array = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)

		{
			gausArray[array] = gaus[i][j];//二维数组到一维 方便计算
			array++;
		}
	}
	//滤波
	for (int i = 0; i < imageSource.rows; i++)
	{
		for (int j = 0; j < imageSource.cols; j++)
		{
			int k = 0;
			for (int l = -size / 2; l <= size / 2; l++)
			{
				for (int g = -size / 2; g <= size / 2; g++)
				{
					//以下处理针对滤波后图像边界处理，为超出边界的值赋值为边界值
					int row = i + l;
					int col = j + g;
					row = row < 0 ? 0 : row;
					row = row >= imageSource.rows ? imageSource.rows - 1 : row;
					col = col < 0 ? 0 : col;
					col = col >= imageSource.cols ? imageSource.cols - 1 : col;
					//卷积和
					imageGaussian.at<uchar>(i, j) += gausArray[k] * imageSource.at<uchar>(row, col);
					k++;
				}
			}
		}
	}
}
//******************Sobel算子计算X、Y方向梯度和梯度方向角********************
//第一个参数imageSourc原始灰度图像；
//第二个参数imageSobelX是X方向梯度图像；
//第三个参数imageSobelY是Y方向梯度图像；
//第四个参数pointDrection是梯度方向角数组指针
//*************************************************************
void SobelGradDirction(const Mat imageSource, Mat& imageSobelX, Mat& imageSobelY, double*& pointDrection)
{
	pointDrection = new double[(imageSource.rows - 1) * (imageSource.cols - 1)];
	for (int i = 0; i < (imageSource.rows - 1) * (imageSource.cols - 1); i++)
	{
		pointDrection[i] = 0;
	}
	imageSobelX = Mat::zeros(imageSource.size(), CV_32SC1);
	imageSobelY = Mat::zeros(imageSource.size(), CV_32SC1);
	uchar* P = imageSource.data;
	uchar* PX = imageSobelX.data;
	uchar* PY = imageSobelY.data;

	int step = imageSource.step;
	int stepXY = imageSobelX.step;
	int k = 0;
	int m = 0;
	int n = 0;
	for (int i = 1; i < (imageSource.rows - 1); i++)
	{
		for (int j = 1; j < (imageSource.cols - 1); j++)
		{
			//通过指针遍历图像上每一个像素 
			double gradY = P[(i - 1) * step + j + 1] + P[i * step + j + 1] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[i * step + j - 1] * 2 - P[(i + 1) * step + j - 1];
			PY[i * stepXY + j * (stepXY / step)] = abs(gradY);
			double gradX = P[(i + 1) * step + j - 1] + P[(i + 1) * step + j] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[(i - 1) * step + j] * 2 - P[(i - 1) * step + j + 1];
			PX[i * stepXY + j * (stepXY / step)] = abs(gradX);
			if (gradX == 0)
			{
				gradX = 0.00000000000000001;  //防止除数为0异常
			}
			pointDrection[k] = atan(gradY / gradX) * 57.3;//弧度转换为度
			pointDrection[k] += 90;
			k++;
		}
	}
	convertScaleAbs(imageSobelX, imageSobelX);
	convertScaleAbs(imageSobelY, imageSobelY);
}
//******************计算Sobel的X和Y方向梯度幅值*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第三个参数SobelAmpXY是输出的X、Y方向梯度图像幅值
//*************************************************************
void SobelAmplitude(const Mat imageGradX, const Mat imageGradY, Mat& SobelAmpXY)
{
	SobelAmpXY = Mat::zeros(imageGradX.size(), CV_32FC1);
	for (int i = 0; i < SobelAmpXY.rows; i++)
	{
		for (int j = 0; j < SobelAmpXY.cols; j++)
		{
			SobelAmpXY.at<float>(i, j) = sqrt(imageGradX.at<uchar>(i, j) * imageGradX.at<uchar>(i, j) + imageGradY.at<uchar>(i, j) * imageGradY.at<uchar>(i, j));
		}
	}
	convertScaleAbs(SobelAmpXY, SobelAmpXY);
}
//******************局部极大值抑制*************************
//第一个参数imageInput输入的Sobel梯度图像；
//第二个参数imageOutPut是输出的局部极大值抑制图像；
//第三个参数pointDrection是图像上每个点的梯度方向数组指针
//*************************************************************
void LocalMaxValue(const Mat imageInput, Mat& imageOutput, double* pointDrection)
{
	//imageInput.copyTo(imageOutput);
	imageOutput = imageInput.clone();
	int k = 0;
	for (int i = 1; i < imageInput.rows - 1; i++)
	{
		for (int j = 1; j < imageInput.cols - 1; j++)
		{
			int value00 = imageInput.at<uchar>((i - 1), j - 1);
			int value01 = imageInput.at<uchar>((i - 1), j);
			int value02 = imageInput.at<uchar>((i - 1), j + 1);
			int value10 = imageInput.at<uchar>((i), j - 1);
			int value11 = imageInput.at<uchar>((i), j);
			int value12 = imageInput.at<uchar>((i), j + 1);
			int value20 = imageInput.at<uchar>((i + 1), j - 1);
			int value21 = imageInput.at<uchar>((i + 1), j);
			int value22 = imageInput.at<uchar>((i + 1), j + 1);

			if (pointDrection[k] > 0 && pointDrection[k] <= 45)
			{
				if (value11 <= (value12 + (value02 - value12) * tan(pointDrection[i * imageOutput.rows + j])) || (value11 <= (value10 + (value20 - value10) * tan(pointDrection[i * imageOutput.rows + j]))))
				{
					imageOutput.at<uchar>(i, j) = 0;
				}
			}
			if (pointDrection[k] > 45 && pointDrection[k] <= 90)

			{
				if (value11 <= (value01 + (value02 - value01) / tan(pointDrection[i * imageOutput.cols + j])) || value11 <= (value21 + (value20 - value21) / tan(pointDrection[i * imageOutput.cols + j])))
				{
					imageOutput.at<uchar>(i, j) = 0;

				}
			}
			if (pointDrection[k] > 90 && pointDrection[k] <= 135)
			{
				if (value11 <= (value01 + (value00 - value01) / tan(180 - pointDrection[i * imageOutput.cols + j])) || value11 <= (value21 + (value22 - value21) / tan(180 - pointDrection[i * imageOutput.cols + j])))
				{
					imageOutput.at<uchar>(i, j) = 0;
				}
			}
			if (pointDrection[k] > 135 && pointDrection[k] <= 180)
			{
				if (value11 <= (value10 + (value00 - value10) * tan(180 - pointDrection[i * imageOutput.cols + j])) || value11 <= (value12 + (value22 - value11) * tan(180 - pointDrection[i * imageOutput.cols + j])))
				{
					imageOutput.at<uchar>(i, j) = 0;
				}
			}
			k++;
		}
	}
}

//******************双阈值处理*************************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//******************************************************
void DoubleThreshold(Mat& imageIput, double lowThreshold, double highThreshold)
{
	for (int i = 0; i < imageIput.rows; i++)
	{
		for (int j = 0; j < imageIput.cols; j++)
		{
			if (imageIput.at<uchar>(i, j) > highThreshold)
			{
				imageIput.at<uchar>(i, j) = 255;
			}
			if (imageIput.at<uchar>(i, j) < lowThreshold)
			{
				imageIput.at<uchar>(i, j) = 0;
			}
		}
	}
}
//******************双阈值中间像素连接处理*********************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//*************************************************************
void DoubleThresholdLink(Mat& imageInput, double lowThreshold, double highThreshold)
{
	for (int i = 1; i < imageInput.rows - 1; i++)
	{
		for (int j = 1; j < imageInput.cols - 1; j++)
		{
			if (imageInput.at<uchar>(i, j) > lowThreshold && imageInput.at<uchar>(i, j) < 255)
			{
				if (imageInput.at<uchar>(i - 1, j - 1) == 255 || imageInput.at<uchar>(i - 1, j) == 255 || imageInput.at<uchar>(i - 1, j + 1) == 255 ||
					imageInput.at<uchar>(i, j - 1) == 255 || imageInput.at<uchar>(i, j) == 255 || imageInput.at<uchar>(i, j + 1) == 255 ||
					imageInput.at<uchar>(i + 1, j - 1) == 255 || imageInput.at<uchar>(i + 1, j) == 255 || imageInput.at<uchar>(i + 1, j + 1) == 255)
				{
					imageInput.at<uchar>(i, j) = 255;
					DoubleThresholdLink(imageInput, lowThreshold, highThreshold); //递归调用
				}
				else
				{
					imageInput.at<uchar>(i, j) = 0;
				}
			}
		}
	}
}
