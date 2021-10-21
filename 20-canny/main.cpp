#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>  
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

//******************�Ҷ�ת������*************************
//��һ������image����Ĳ�ɫRGBͼ��
//�ڶ�������imageGray��ת��������ĻҶ�ͼ��
//*************************************************************
void ConvertRGB2GRAY(const Mat& image, Mat& imageGray);


//******************��˹��������ɺ���*************************
//��һ������gaus��һ��ָ����N��double���������ָ�룻
//�ڶ�������size�Ǹ�˹����˵ĳߴ��С��
//����������sigma�Ǿ���˵ı�׼��
//*************************************************************
void GetGaussianKernel(double** gaus, const int size, const double sigma);

//******************��˹�˲�*************************
//��һ������imageSource�Ǵ��˲�ԭʼͼ��
//�ڶ�������imageGaussian���˲������ͼ��
//����������gaus��һ��ָ����N��double���������ָ�룻
//���ĸ�����size���˲��˵ĳߴ�
//*************************************************************
void GaussianFilter(const Mat imageSource, Mat& imageGaussian, double** gaus, int size);

//******************Sobel���Ӽ����ݶȺͷ���********************
//��һ������imageSourcԭʼ�Ҷ�ͼ��
//�ڶ�������imageSobelX��X�����ݶ�ͼ��
//����������imageSobelY��Y�����ݶ�ͼ��
//���ĸ�����pointDrection���ݶȷ�������ָ��
//*************************************************************
void SobelGradDirction(const Mat imageSource, Mat& imageSobelX, Mat& imageSobelY, double*& pointDrection);

//******************����Sobel��X��Y�����ݶȷ�ֵ*************************
//��һ������imageGradX��X�����ݶ�ͼ��
//�ڶ�������imageGradY��Y�����ݶ�ͼ��
//����������SobelAmpXY�������X��Y�����ݶ�ͼ���ֵ
//*************************************************************
void SobelAmplitude(const Mat imageGradX, const Mat imageGradY, Mat& SobelAmpXY);

//******************�ֲ�����ֵ����*************************
//��һ������imageInput�����Sobel�ݶ�ͼ��
//�ڶ�������imageOutPut������ľֲ�����ֵ����ͼ��
//����������pointDrection��ͼ����ÿ������ݶȷ�������ָ��
//*************************************************************
void LocalMaxValue(const Mat imageInput, Mat& imageOutput, double* pointDrection);

//******************˫��ֵ����*************************
//��һ������imageInput���������ĵ�Sobel�ݶȷ�ֵͼ��
//�ڶ�������lowThreshold�ǵ���ֵ
//����������highThreshold�Ǹ���ֵ
//******************************************************
void DoubleThreshold(Mat& imageIput, double lowThreshold, double highThreshold);

//******************˫��ֵ�м��������Ӵ���*********************
//��һ������imageInput���������ĵ�Sobel�ݶȷ�ֵͼ��
//�ڶ�������lowThreshold�ǵ���ֵ
//����������highThreshold�Ǹ���ֵ
//*************************************************************
void DoubleThresholdLink(Mat& imageInput, double lowThreshold, double highThreshold);

Mat imageSource;
Mat imageGray;
Mat imageGaussian;

int main(int argc, char* argv[])
{
	imageSource = imread("4.png");  //����RGBͼ��
	imshow("RGB Image", imageSource);
	ConvertRGB2GRAY(imageSource, imageGray); //RGBת��Ϊ�Ҷ�ͼ
	imshow("Gray Image", imageGray);
	int size = 5; //�������˴�С
	double** gaus = new double* [size];  //���������
	for (int i = 0; i < size; i++)
	{
		gaus[i] = new double[size];  //��̬���ɾ���
	}
	GetGaussianKernel(gaus, 5, 1); //����5*5 ��С��˹����ˣ�Sigma=1��
	imageGaussian = Mat::zeros(imageGray.size(), CV_8UC1);
	GaussianFilter(imageGray, imageGaussian, gaus, 5);  //��˹�˲�
	imshow("Gaussian Image", imageGaussian);
	Mat imageSobelY;
	Mat imageSobelX;
	double* pointDirection = new double[(imageSobelX.cols - 1) * (imageSobelX.rows - 1)];  //�����ݶȷ��������
	SobelGradDirction(imageGaussian, imageSobelX, imageSobelY, pointDirection);  //����X��Y�����ݶȺͷ����
	imshow("Sobel Y", imageSobelY);
	imshow("Sobel X", imageSobelX);
	Mat SobelGradAmpl;
	SobelAmplitude(imageSobelX, imageSobelY, SobelGradAmpl);   //����X��Y�����ݶ��ںϷ�ֵ
	imshow("Soble XYRange", SobelGradAmpl);
	Mat imageLocalMax;
	LocalMaxValue(SobelGradAmpl, imageLocalMax, pointDirection);  //�ֲ��Ǽ���ֵ����
	imshow("Non-Maximum Image", imageLocalMax);
	Mat cannyImage;
	cannyImage = Mat::zeros(imageLocalMax.size(), CV_8UC1);
	DoubleThreshold(imageLocalMax, 90, 160);        //˫��ֵ����
	imshow("Double Threshold Image", imageLocalMax);
	DoubleThresholdLink(imageLocalMax, 90, 160);   //˫��ֵ�м���ֵ�˳�������
	imshow("Canny Image", imageLocalMax);
	waitKey();
	system("pause");
	return 0;
}

//******************��˹��������ɺ���*************************
//��һ������gaus��һ��ָ����N��double���������ָ�룻
//�ڶ�������size�Ǹ�˹����˵ĳߴ��С��
//����������sigma�Ǿ���˵ı�׼��
//*************************************************************
void GetGaussianKernel(double** gaus, const int size, const double sigma)
{
	const double PI = 4.0 * atan(1.0); //Բ���ʦи�ֵ
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

//******************�Ҷ�ת������*************************
//��һ������image����Ĳ�ɫRGBͼ��
//�ڶ�������imageGray��ת��������ĻҶ�ͼ��
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

//******************��˹�˲�*************************
//��һ������imageSource�Ǵ��˲�ԭʼͼ��
//�ڶ�������imageGaussian���˲������ͼ��
//����������gaus��һ��ָ����N��double���������ָ�룻
//���ĸ�����size���˲��˵ĳߴ�
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
		gausArray[i] = 0;  //����ֵ���ռ����
	}
	int array = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)

		{
			gausArray[array] = gaus[i][j];//��ά���鵽һά �������
			array++;
		}
	}
	//�˲�
	for (int i = 0; i < imageSource.rows; i++)
	{
		for (int j = 0; j < imageSource.cols; j++)
		{
			int k = 0;
			for (int l = -size / 2; l <= size / 2; l++)
			{
				for (int g = -size / 2; g <= size / 2; g++)
				{
					//���´�������˲���ͼ��߽紦��Ϊ�����߽��ֵ��ֵΪ�߽�ֵ
					int row = i + l;
					int col = j + g;
					row = row < 0 ? 0 : row;
					row = row >= imageSource.rows ? imageSource.rows - 1 : row;
					col = col < 0 ? 0 : col;
					col = col >= imageSource.cols ? imageSource.cols - 1 : col;
					//�����
					imageGaussian.at<uchar>(i, j) += gausArray[k] * imageSource.at<uchar>(row, col);
					k++;
				}
			}
		}
	}
}
//******************Sobel���Ӽ���X��Y�����ݶȺ��ݶȷ����********************
//��һ������imageSourcԭʼ�Ҷ�ͼ��
//�ڶ�������imageSobelX��X�����ݶ�ͼ��
//����������imageSobelY��Y�����ݶ�ͼ��
//���ĸ�����pointDrection���ݶȷ��������ָ��
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
			//ͨ��ָ�����ͼ����ÿһ������ 
			double gradY = P[(i - 1) * step + j + 1] + P[i * step + j + 1] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[i * step + j - 1] * 2 - P[(i + 1) * step + j - 1];
			PY[i * stepXY + j * (stepXY / step)] = abs(gradY);
			double gradX = P[(i + 1) * step + j - 1] + P[(i + 1) * step + j] * 2 + P[(i + 1) * step + j + 1] - P[(i - 1) * step + j - 1] - P[(i - 1) * step + j] * 2 - P[(i - 1) * step + j + 1];
			PX[i * stepXY + j * (stepXY / step)] = abs(gradX);
			if (gradX == 0)
			{
				gradX = 0.00000000000000001;  //��ֹ����Ϊ0�쳣
			}
			pointDrection[k] = atan(gradY / gradX) * 57.3;//����ת��Ϊ��
			pointDrection[k] += 90;
			k++;
		}
	}
	convertScaleAbs(imageSobelX, imageSobelX);
	convertScaleAbs(imageSobelY, imageSobelY);
}
//******************����Sobel��X��Y�����ݶȷ�ֵ*************************
//��һ������imageGradX��X�����ݶ�ͼ��
//�ڶ�������imageGradY��Y�����ݶ�ͼ��
//����������SobelAmpXY�������X��Y�����ݶ�ͼ���ֵ
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
//******************�ֲ�����ֵ����*************************
//��һ������imageInput�����Sobel�ݶ�ͼ��
//�ڶ�������imageOutPut������ľֲ�����ֵ����ͼ��
//����������pointDrection��ͼ����ÿ������ݶȷ�������ָ��
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

//******************˫��ֵ����*************************
//��һ������imageInput���������ĵ�Sobel�ݶȷ�ֵͼ��
//�ڶ�������lowThreshold�ǵ���ֵ
//����������highThreshold�Ǹ���ֵ
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
//******************˫��ֵ�м��������Ӵ���*********************
//��һ������imageInput���������ĵ�Sobel�ݶȷ�ֵͼ��
//�ڶ�������lowThreshold�ǵ���ֵ
//����������highThreshold�Ǹ���ֵ
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
					DoubleThresholdLink(imageInput, lowThreshold, highThreshold); //�ݹ����
				}
				else
				{
					imageInput.at<uchar>(i, j) = 0;
				}
			}
		}
	}
}
