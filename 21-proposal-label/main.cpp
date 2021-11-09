#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void fillRunVectors(const Mat& bwImage, int& NumberOfRuns, vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun)
{
	for (int i = 0; i < bwImage.rows; i++)
	{
		const uchar* rowData = bwImage.ptr<uchar>(i);

		if (rowData[0] == 1)
		{
			NumberOfRuns++;
			stRun.push_back(0);
			rowRun.push_back(i);
		}
		for (int j = 1; j < bwImage.cols; j++)
		{
			if (rowData[j - 1] == 0 && rowData[j] == 1)
			{
				NumberOfRuns++;
				stRun.push_back(j);
				rowRun.push_back(i);
			}
			else if (rowData[j - 1] == 1 && rowData[j] == 0)
			{
				enRun.push_back(j - 1);
			}
		}
		if (rowData[bwImage.cols - 1])
		{
			enRun.push_back(bwImage.cols - 1);
		}
	}
}


void firstPass(vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, int NumberOfRuns,
	vector<int>& runLabels, vector<pair<int, int>>& equivalences, int offset)
{
	runLabels.assign(NumberOfRuns, 0);
	int idxLabel = 1;
	int curRowIdx = 0;
	int firstRunOnCur = 0;
	int firstRunOnPre = 0;
	int lastRunOnPre = -1;
	for (int i = 0; i < NumberOfRuns; i++)
	{
		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i];
			firstRunOnPre = firstRunOnCur;
			lastRunOnPre = i - 1;
			firstRunOnCur = i;

		}
		for (int j = firstRunOnPre; j <= lastRunOnPre; j++)
		{
			if (stRun[i] <= enRun[j] + offset && enRun[i] >= stRun[j] - offset && rowRun[i] == rowRun[j] + 1)
			{
				if (runLabels[i] == 0) // 没有被标号过
					runLabels[i] = runLabels[j];
				else if (runLabels[i] != runLabels[j])// 已经被标号             
					equivalences.push_back(make_pair(runLabels[i], runLabels[j])); // 保存等价对
			}
		}
		if (runLabels[i] == 0) // 没有与前一列的任何run重合
		{
			runLabels[i] = idxLabel++;
		}

	}
}

void replaceSameLabel(vector<int>& runLabels, vector<pair<int, int>>&
	equivalence)
{
	int maxLabel = *max_element(runLabels.begin(), runLabels.end());
	vector<vector<bool>> eqTab(maxLabel, vector<bool>(maxLabel, false));
	vector<pair<int, int>>::iterator vecPairIt = equivalence.begin();
	while (vecPairIt != equivalence.end())
	{
		eqTab[vecPairIt->first - 1][vecPairIt->second - 1] = true;
		eqTab[vecPairIt->second - 1][vecPairIt->first - 1] = true;
		vecPairIt++;
	}
	vector<int> labelFlag(maxLabel, 0);
	vector<vector<int>> equaList;
	vector<int> tempList;
	cout << maxLabel << endl;
	for (int i = 1; i <= maxLabel; i++)
	{
		if (labelFlag[i - 1])
		{
			continue;
		}
		labelFlag[i - 1] = equaList.size() + 1;
		tempList.push_back(i);
		for (vector<int>::size_type j = 0; j < tempList.size(); j++)
		{
			for (vector<bool>::size_type k = 0; k != eqTab[tempList[j] - 1].size(); k++)
			{
				if (eqTab[tempList[j] - 1][k] && !labelFlag[k])
				{
					tempList.push_back(k + 1);
					labelFlag[k] = equaList.size() + 1;
				}
			}
		}
		equaList.push_back(tempList);
		tempList.clear();
	}
	cout << equaList.size() << endl;
	for (vector<int>::size_type i = 0; i != runLabels.size(); i++)
	{
		runLabels[i] = labelFlag[runLabels[i] - 1];
	}
}

void fillImage(Mat& labelImg, vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, vector<int>& runLabels) {
	int NumberOfRuns = stRun.size();
	int curRowIdx = -1;
	int* rowData = nullptr;
	for (int i = 0; i < NumberOfRuns; i++) {
		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i];
			rowData = labelImg.ptr<int>(rowRun[i]);
		}
		for (int j = stRun[i]; j <= enRun[i]; j++) {
			rowData[j] = runLabels[i];
		}
	}
}

void findConnectedProposal(const Mat& bwImage, Mat& labelImg) {
	int NumberOfRuns = 0;
	vector<int> stRun;
	vector<int> enRun;
	vector<int> rowRun;
	fillRunVectors(bwImage, NumberOfRuns, stRun, enRun, rowRun);

	vector<pair<int, int>> equivalences;
	vector<int> runLabels;
	int offset = 1;   // 0是四邻域连通，1是八邻域连通
	firstPass(stRun, enRun, rowRun, NumberOfRuns, runLabels, equivalences, offset);

	replaceSameLabel(runLabels, equivalences);

	labelImg = Mat(bwImage.size(), CV_32SC1, Scalar(0));
	fillImage(labelImg, stRun, enRun, rowRun, runLabels);
}

cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	int num = 0;
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
					num++;
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}

	printf("color num : %d \n", num);
}


void Two_Pass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	//connected component analysis (4-component)
	//use two-pass algorithm
	//1. first pass: label each foreground pixel with a label
	//2. second pass: visit each labeled pixel and merge neighbor label
	//
	//foreground pixel: _binImg(x,y) = 1
	//background pixel: _binImg(x,y) = 0

	if (_binImg.empty() || _binImg.type() != CV_8UC1)
	{
		return;
	}

	// 1. first pass
	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2
	std::vector<int> labelSet;
	labelSet.push_back(0);   //background: 0
	labelSet.push_back(1);   //foreground: 1

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _lableImg.ptr<int>(i - 1);
		int* data_curRow = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2); //reserve(n)  预分配n个元素的存储空间
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}
				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);   //assign to a new label
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					//save equivalence
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}

			}
		}
	}
	//update equivalent labels
	//assigned with the smallest label in each equivalent label set
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}

	//2. second pass
	for (int i = 0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
		}
	}
}


//---------------------------------【种子填充法老版】-------------------------------
void SeedFill(const cv::Mat& binImg, cv::Mat& lableImg)   //种子填充法
{
	// 4邻接方法


	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	lableImg.release();
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1;

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = lableImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j));     // 像素位置: <i,j>
				++label;  // 没有重复的团，开始新的标签
				while (!neighborPixels.empty())
				{
					std::pair<int, int> curPixel = neighborPixels.top(); //如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它
					int curX = curPixel.first;
					int curY = curPixel.second;
					lableImg.at<int>(curX, curY) = label;

					neighborPixels.pop();

					if (lableImg.at<int>(curX, curY - 1) == 1)
					{//左边
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (lableImg.at<int>(curX, curY + 1) == 1)
					{// 右边
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					if (lableImg.at<int>(curX - 1, curY) == 1)
					{// 上边
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (lableImg.at<int>(curX + 1, curY) == 1)
					{// 下边
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
			}
		}
	}

}



int main() {

	cv::Mat binImage = cv::imread("3.png", 0);
	// 原图显示
	imshow("binImage", binImage);
	// 二值化
	cv::threshold(binImage, binImage, 50, 1, THRESH_BINARY);

	cv::Mat labelImg;
	double time;
	time = getTickCount();
	//findConnectedProposal(binImage, labelImg);
	//Two_Pass(binImage, labelImg);
	SeedFill(binImage, labelImg);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << std::fixed << time << "ms" << endl;


	//彩色显示
	cv::Mat colorLabelImg;
	LabelColor(labelImg * 10, colorLabelImg);
	cv::imshow("colorImg", colorLabelImg);

	////灰度显示
	cv::Mat grayImg;
	labelImg *= 10;
	labelImg.convertTo(grayImg, CV_8UC1);
	imshow("labelImg", grayImg);

	waitKey(0);
}