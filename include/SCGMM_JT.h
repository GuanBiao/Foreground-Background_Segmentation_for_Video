#ifndef __SCGMM_JT_H__
#define __SCGMM_JT_H__

#include "utils.h"
#include "graphcut.hpp"

class SCGMM_JT {
private:
	cv::TermCriteria termCrit;

	/*
	 * Parameters for the foreground SCGMM model.
	 */
	int fgdNclusters;
	cv::Mat fgdWeights, fgdMeans, fgdProbs;
	vector<cv::Mat> fgdCovs;
	cv::EM fgdEm;

	/*
	 * Parameters for the background SCGMM model.
	 */
	int bgdNclusters;
	cv::Mat bgdWeights, bgdMeans, bgdProbs;
	vector<cv::Mat> bgdCovs;
	cv::EM bgdEm;

	/*
	 * Parameters for SCGMM joint tracking.
	 *   s: spatial, c: color.
	 */
	int nclusters;
	cv::Mat weights;
	cv::Mat sMeans, cMeans;
	cv::Mat probs;
	vector<cv::Mat> sCovs, sInvCovs, cCovs, cInvCovs;
	cv::Mat sSqrtDetCovs, cSqrtDetCovs;

	cv::Mat trainSamples;
	cv::Mat trainLogLikelihoods;

	void combineParams();
	void decomposeParams();
	void eStep();
	void mStep();
	void predict(const cv::Mat &src);
	void postUpdate(const cv::Mat &mask);

public:
	SCGMM_JT(int fgdNclusters = 5, int bgdNclusters = 10, cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-3));
	void init(const cv::Mat &src, const cv::Mat &fgdSamples, const cv::Mat &bgdSamples);
	void run(const cv::Mat &src, cv::Mat &dst);
};

void SCGMM_JT::combineParams() {
	sMeans.create(nclusters, 2, CV_64FC1);
	sCovs.resize(nclusters);
	sInvCovs.resize(nclusters);
	sSqrtDetCovs.create(1, nclusters, CV_64FC1);
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
		if (clusterIndex < fgdNclusters) {
			fgdMeans.row(clusterIndex).colRange(0, 2).copyTo(sMeans.row(clusterIndex));
			fgdCovs.at(clusterIndex)(cv::Rect(0, 0, 2, 2)).copyTo(sCovs.at(clusterIndex));
		}
		else {
			bgdMeans.row(clusterIndex - fgdNclusters).colRange(0, 2).copyTo(sMeans.row(clusterIndex));
			bgdCovs.at(clusterIndex - fgdNclusters)(cv::Rect(0, 0, 2, 2)).copyTo(sCovs.at(clusterIndex));
		}

		cv::invert(sCovs.at(clusterIndex), sInvCovs.at(clusterIndex));
		sSqrtDetCovs.at<double>(clusterIndex) = sqrt(cv::determinant(sCovs.at(clusterIndex)));
	}
}

void SCGMM_JT::decomposeParams() {
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
		if (clusterIndex < fgdNclusters) {
			sMeans.row(clusterIndex).copyTo(fgdMeans.row(clusterIndex).colRange(0, 2));
			sCovs.at(clusterIndex).copyTo(fgdCovs.at(clusterIndex)(cv::Rect(0, 0, 2, 2)));

			fgdWeights.at<double>(clusterIndex) = weights.at<double>(clusterIndex);
		}
		else {
			sMeans.row(clusterIndex).copyTo(bgdMeans.row(clusterIndex - fgdNclusters).colRange(0, 2));
			sCovs.at(clusterIndex).copyTo(bgdCovs.at(clusterIndex - fgdNclusters)(cv::Rect(0, 0, 2, 2)));

			bgdWeights.at<double>(clusterIndex - fgdNclusters) = weights.at<double>(clusterIndex);
		}
	}
}

void SCGMM_JT::eStep() {
	for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++) {
		for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
			cv::Mat sDiff = trainSamples.row(sampleIndex).colRange(0, 2) - sMeans.row(clusterIndex);
			cv::Mat cDiff = trainSamples.row(sampleIndex).colRange(2, 5) - cMeans.row(clusterIndex);
			cv::Mat sExponent = -sDiff * sInvCovs.at(clusterIndex) * sDiff.t() / 2;
			cv::Mat cExponent = -cDiff * cInvCovs.at(clusterIndex) * cDiff.t() / 2;
			double prob = exp(sExponent.at<double>(0) + cExponent.at<double>(0)) / (pow(2 * CV_PI, 2.5) * sSqrtDetCovs.at<double>(clusterIndex) * cSqrtDetCovs.at<double>(clusterIndex));
			probs.at<double>(sampleIndex, clusterIndex) = weights.at<double>(clusterIndex) * prob;
		}

		cv::Scalar sum = cv::sum(probs.row(sampleIndex));
		probs.row(sampleIndex) /= sum[0];
		trainLogLikelihoods.at<double>(sampleIndex) = log(sum[0]);
	}
}

void SCGMM_JT::mStep() {
	cv::Mat Nk(1, nclusters, CV_64FC1);
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
		Nk.at<double>(clusterIndex) = cv::sum(probs.col(clusterIndex))[0];

	/*
	 * Update sMeans.
	 */
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
		cv::Mat sum = cv::Mat::zeros(1, 2, CV_64FC1);
		for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
			sum += probs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex).colRange(0, 2);

		sMeans.row(clusterIndex) = sum / Nk.at<double>(clusterIndex);
	}

	/*
	 * Update sCovs, sInvCovs and sSqrtDetCovs.
	 */
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
		sCovs.at(clusterIndex) = cv::Mat::zeros(2, 2, CV_64FC1);
		for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++) {
			cv::Mat tmp;
			cv::mulTransposed(trainSamples.row(sampleIndex).colRange(0, 2), tmp, true, sMeans.row(clusterIndex), probs.at<double>(sampleIndex, clusterIndex));
			sCovs.at(clusterIndex) += tmp;
		}
		sCovs.at(clusterIndex) /= Nk.at<double>(clusterIndex);

		cv::invert(sCovs.at(clusterIndex), sInvCovs.at(clusterIndex));
		sSqrtDetCovs.at<double>(clusterIndex) = sqrt(cv::determinant(sCovs.at(clusterIndex)));
	}

	/*
	 * Update weights.
	 */
	weights = Nk / cv::sum(Nk)[0];
}

void SCGMM_JT::predict(const cv::Mat &src) {
	fgdProbs.create(src.size(), CV_64FC1);
	bgdProbs.create(src.size(), CV_64FC1);
	for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++) {
		int row = (int)trainSamples.row(sampleIndex).at<double>(1);
		int col = (int)trainSamples.row(sampleIndex).at<double>(0);
		fgdProbs.at<double>(row, col) = cv::sum(probs.row(sampleIndex).colRange(0, fgdNclusters))[0];
		bgdProbs.at<double>(row, col) = cv::sum(probs.row(sampleIndex).colRange(fgdNclusters, nclusters))[0];
	}
}

void SCGMM_JT::postUpdate(const cv::Mat &mask) {
	cv::Mat fgdSamples, bgdSamples;
	for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++) {
		int row = (int)trainSamples.row(sampleIndex).at<double>(1);
		int col = (int)trainSamples.row(sampleIndex).at<double>(0);
		if (mask.at<uchar>(row, col) == 255)
			fgdSamples.push_back(trainSamples.row(sampleIndex));
		else
			bgdSamples.push_back(trainSamples.row(sampleIndex));
	}

	fgdEm.trainE(fgdSamples, fgdMeans, fgdCovs, fgdWeights);
	fgdWeights = fgdEm.get<cv::Mat>("weights");
	fgdMeans = fgdEm.get<cv::Mat>("means");
	fgdCovs = fgdEm.get<vector<cv::Mat>>("covs");

	bgdEm.trainE(bgdSamples, bgdMeans, bgdCovs, bgdWeights);
	bgdWeights = bgdEm.get<cv::Mat>("weights");
	bgdMeans = bgdEm.get<cv::Mat>("means");
	bgdCovs = bgdEm.get<vector<cv::Mat>>("covs");
}

SCGMM_JT::SCGMM_JT(int fgdNclusters, int bgdNclusters, cv::TermCriteria termCrit) {
	this->fgdNclusters = fgdNclusters;
	this->bgdNclusters = bgdNclusters;
	this->termCrit = termCrit;
}

void SCGMM_JT::init(const cv::Mat &src, const cv::Mat &fgdSamples, const cv::Mat &bgdSamples) {
	fgdEm = cv::EM(fgdNclusters, cv::EM::COV_MAT_GENERIC, termCrit);
	fgdEm.train(fgdSamples);
	fgdWeights = fgdEm.get<cv::Mat>("weights");
	fgdMeans = fgdEm.get<cv::Mat>("means");
	fgdCovs = fgdEm.get<vector<cv::Mat>>("covs");

	bgdEm = cv::EM(bgdNclusters, cv::EM::COV_MAT_GENERIC, termCrit);
	bgdEm.train(bgdSamples);
	bgdWeights = bgdEm.get<cv::Mat>("weights");
	bgdMeans = bgdEm.get<cv::Mat>("means");
	bgdCovs = bgdEm.get<vector<cv::Mat>>("covs");

	nclusters = fgdNclusters + bgdNclusters;
	weights.create(1, nclusters, CV_64FC1); weights.setTo(1.0 / nclusters);
	probs.create(src.rows * src.cols, nclusters, CV_64FC1);

	cMeans.create(nclusters, 3, CV_64FC1);
	cCovs.resize(nclusters);
	cInvCovs.resize(nclusters);
	cSqrtDetCovs.create(1, nclusters, CV_64FC1);
	for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
		if (clusterIndex < fgdNclusters) {
			fgdMeans.row(clusterIndex).colRange(2, 5).copyTo(cMeans.row(clusterIndex));
			fgdCovs.at(clusterIndex)(cv::Rect(2, 2, 3, 3)).copyTo(cCovs.at(clusterIndex));
		}
		else {
			bgdMeans.row(clusterIndex - fgdNclusters).colRange(2, 5).copyTo(cMeans.row(clusterIndex));
			bgdCovs.at(clusterIndex - fgdNclusters)(cv::Rect(2, 2, 3, 3)).copyTo(cCovs.at(clusterIndex));
		}

		cv::invert(cCovs.at(clusterIndex), cInvCovs.at(clusterIndex));
		cSqrtDetCovs.at<double>(clusterIndex) = sqrt(cv::determinant(cCovs.at(clusterIndex)));
	}
}

void SCGMM_JT::run(const cv::Mat &src, cv::Mat &dst) {
	combineParams();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			cv::Mat sample = (cv::Mat_<double>(1, 5) << j, i, src.at<cv::Vec3b>(i, j)[0], src.at<cv::Vec3b>(i, j)[1], src.at<cv::Vec3b>(i, j)[2]);
			trainSamples.push_back(sample);
		}
	}
	trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);

	double trainLogLikelihood, prevTrainLogLikelihood = 0.0;
	for (int curIteration = 0; ; curIteration++) {
		eStep();

		if (curIteration >= termCrit.maxCount)
			break;

		trainLogLikelihood = cv::sum(trainLogLikelihoods)[0];
		double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;
		if (curIteration != 0 && (trainLogLikelihoodDelta < -DBL_EPSILON || trainLogLikelihoodDelta < termCrit.epsilon * fabs(prevTrainLogLikelihood)))
			break;

		mStep();

		prevTrainLogLikelihood = trainLogLikelihood;
	}

	predict(src);
	dst.create(src.size(), CV_8UC1);
	dst.setTo(128);
	graphCut(src, dst, bgdProbs, fgdProbs);

	decomposeParams();

	postUpdate(dst);

	trainSamples.release();
	trainLogLikelihoods.release();
}

#endif