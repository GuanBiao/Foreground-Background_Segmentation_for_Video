#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;

void getImgSeqFromDir(const char *imgDir, const char *gtDir, int nimg, vector<cv::Mat> &imgs, vector<cv::Mat> &gts) {
	imgs.reserve(nimg);
	gts.reserve(nimg);

	cv::Mat resized;
	char path[500];
	for (int imgIndex = 0; imgIndex < nimg; imgIndex++) {
		sprintf(path, "%s\\%05d.jpg", imgDir, imgIndex);
		cv::resize(cv::imread(path), resized, cv::Size(), 0.4, 0.4);
		imgs.push_back(resized.clone());

		sprintf(path, "%s\\%05d.png", gtDir, imgIndex);
		cv::resize(cv::imread(path), resized, cv::Size(), 0.4, 0.4);
		gts.push_back(resized.clone());
	}
}

double calcAccuracy(const cv::Mat &img, const cv::Mat &gt) {
	int nmatch = 0;
	cv::Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (img.at<uchar>(p) == 255 && gt.at<cv::Vec3b>(p) == cv::Vec3b(255, 255, 255))
				nmatch++;
			else if (img.at<uchar>(p) == 0 && gt.at<cv::Vec3b>(p) == cv::Vec3b(0, 0, 0))
				nmatch++;
		}
	}

	return (double)nmatch / (img.rows * img.cols);
}

#endif