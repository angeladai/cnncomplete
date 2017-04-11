#pragma once
#include "../VirtualScan/VoxelGrid.h"

struct EvalStatsDist {
	float l1sum;
	float l2sum;
	unsigned int normalization;

	EvalStatsDist() {
		l1sum = 0;
		l2sum = 0;
		normalization = 0;
	}

	void print() const {
		std::cout << "\t%l1 norm = " << l1sum / (float)normalization << "\t(" << l1sum << "/" << normalization << ")" << std::endl;
		std::cout << "\t%l2 norm = " << l2sum / (float)normalization << "\t(" << l2sum << "/" << normalization << ")" << std::endl;
	}
};

class Evaluation {
public:
	static EvalStatsDist evaluate(const VoxelGrid& inputSDF, const DistanceField3f& gtDF, const DistanceField3f& predDF, float truncation = 2.5f);

	static void evaluate(const std::string& inputDir, const std::string& groundTruthDir, const std::string& completionDir, bool bNestedDirectories = true);

private:

};