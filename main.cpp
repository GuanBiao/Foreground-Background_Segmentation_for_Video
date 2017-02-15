#include "SCGMM_JT.h"

cv::Mat origImg, segImg, gtImg;

cv::Mat fgdSamples, bgdSamples;

int main(int argc, char *argv[]) {
	vector<cv::Mat> origImgs, gtImgs;

	cout << "Reading " << argv[1] << " original and ground truth images... ";
	getImgSeqFromDir(argv[2], argv[3], atoi(argv[1]), origImgs, gtImgs);
	cout << "Done!" << endl;

	origImg = origImgs.front();
	gtImg = gtImgs.front();

	cv::namedWindow("original");
	cv::namedWindow("ground truth");

	cv::imshow("original", origImg);
	cv::imshow("ground truth", gtImg);

	/*
	 * Use the whole image to be our training samples for initialization.
	 */
	for (int i = 0; i < gtImg.rows; i++) {
		for (int j = 0; j < gtImg.cols; j++) {
			cv::Mat sample = (cv::Mat_<double>(1, 5) << j, i, origImg.at<cv::Vec3b>(i, j)[0], origImg.at<cv::Vec3b>(i, j)[1], origImg.at<cv::Vec3b>(i, j)[2]);
			if (gtImg.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
				fgdSamples.push_back(sample);
			else
				bgdSamples.push_back(sample);
		}
	}

	while (1) {
		int key = cv::waitKey(1);
		if (key == 27)
			break;
		else if (key == 's') {
			clock_t start, end;
			SCGMM_JT model;

			cout << "Initializing SCGMM Joint Tracking... ";
			start = clock();
			model.init(origImg, fgdSamples, bgdSamples);
			end = clock();
			cout << "Done! (" << (end - start) / (double)CLOCKS_PER_SEC << "s)" << endl;

			for (int imgIndex = 0; imgIndex < origImgs.size(); imgIndex++) {
				cout << "Processing frame " << imgIndex << "... ";
				start = clock();
				model.run(origImgs.at(imgIndex), segImg);
				end = clock();
				cout << "Done! (" << (end - start) / (double)CLOCKS_PER_SEC << "s)" << endl;
				cout << "Accuracy: " << calcAccuracy(segImg, gtImgs.at(imgIndex)) << endl;

				stringstream ss;
				ss << "result/" << imgIndex << ".png";
				cv::imwrite(ss.str(), segImg);
			}
		}
	}

	return 0;
}