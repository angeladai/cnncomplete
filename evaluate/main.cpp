

#include "stdafx.h"
#include "Evaluation.h"
#include <random>


int _tmain(int argc, _TCHAR* argv[])
{
	try {
		const std::string inputDir = "test-images_dim32_sdf";
		const std::string targetDir = "shapenet_dim32_df";
		const std::string completionDir = "output-test-images-32";

		Evaluation::evaluate(inputDir, targetDir, completionDir);
	}
	catch (MLibException& e)
	{
		std::stringstream ss;
		ss << "exception caught:" << e.what() << std::endl;
		std::cout << ss.str() << std::endl;
	}
	std::cout << "DONE!" << std::endl;
	getchar();

	return 0;
}

