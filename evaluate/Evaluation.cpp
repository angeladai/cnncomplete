#include "stdafx.h"
#include "Evaluation.h"

void Evaluation::evaluate(const std::string& inputDir, const std::string& groundTruthDir, const std::string& completionDir, bool bNestedDirectories /*= true*/)
{
	if (!util::directoryExists(inputDir) || !util::directoryExists(groundTruthDir) || !util::directoryExists(completionDir))
		throw MLIB_EXCEPTION("evaluate: input/completion/gt dir(s) do not exist");
	std::cout << "================ EVALUATE ================" << std::endl;
	Timer t;

	//stats to compute
	std::unordered_map<std::string, std::vector<EvalStatsDist>> statsDist;
	Directory resultDir(completionDir);
	std::vector<std::string> files;
	if (bNestedDirectories) {
		const std::vector<std::string> classes = resultDir.getDirectories();
		for (const std::string& classname : classes) {
			std::vector<std::string> classfiles = Directory(resultDir.getPath() + classname).getFilesWithSuffix(".df");
			for (std::string& c : classfiles) c = classname + "/" + c;
			if (!classfiles.empty()) files.insert(files.end(), classfiles.begin(), classfiles.end());
		}
	}
	else {
		files = resultDir.getFilesWithSuffix(".df");
	}

	for (unsigned int i = 0; i < files.size(); i++) {
		const auto parts = util::splitPath(files[i]);
		const std::string modelTrajId = util::removeExtensions(parts.back());
		const std::string modelId = util::splitOnFirst(modelTrajId, "__").first; //model id
		const std::string classId = bNestedDirectories ? parts[parts.size() - 2] : "";
		
		//load gt
		const std::string gtFile = groundTruthDir + "/" + classId + "/" + modelId + "__0__.df";
		if (!util::fileExists(gtFile)) throw MLIB_EXCEPTION("gt file (" + gtFile + ") does not exist!");
		DistanceField3f gtDF; loadDF<UINT64>(gtFile, gtDF);
		//load input
		const std::string inputFile = inputDir + "/" + classId + "/" + modelTrajId + ".sdf";
		if (!util::fileExists(inputFile)) throw MLIB_EXCEPTION("input file (" + inputFile + ") does not exist!");
		VoxelGrid inputSDF(vec3ul(0, 0, 0), mat4f::identity(), 0.0f, 0.0f, 0.0f);
		inputSDF.loadFromFile(inputFile); 
		//load completion
		const std::string& predFile = completionDir + "/" + files[i];
		if (!util::fileExists(predFile)) throw MLIB_EXCEPTION("completion file (" + predFile + ") does not exist!");
		//load completion
		DistanceField3f predDF; loadDF<UINT64>(predFile, predDF);
		const auto itDist = statsDist.find(classId);
		if (itDist == statsDist.end()) statsDist[classId] = std::vector<EvalStatsDist>(1, evaluate(inputSDF, gtDF, predDF));
		else itDist->second.push_back(evaluate(inputSDF, gtDF, predDF));
		std::cout << "\r[ " << (i + 1) << " | " << files.size() << " ]";
	}
	t.stop();
	std::cout << std::endl;

	//compute summary stats
	EvalStatsDist totalStatsDist; unsigned int counter = 0;
	std::vector<std::pair<std::string, EvalStatsDist>> classStatsDist;
	for (const auto& stat : statsDist) {
		EvalStatsDist classStat;
		for (const auto& e : stat.second) {
			totalStatsDist.l1sum += e.l1sum / (float)e.normalization;
			totalStatsDist.l2sum += e.l2sum / (float)e.normalization;
			totalStatsDist.normalization++;

			classStat.l1sum += e.l1sum / (float)e.normalization;
			classStat.l2sum += e.l2sum / (float)e.normalization;
			classStat.normalization++;

			counter++;
		}
		classStatsDist.push_back(std::make_pair(stat.first, classStat));
	}
	std::cout << "\tcomputed stats for " << counter << " of " << files.size() << " test files" << std::endl;
	// ------- distance --------
	std::cout << std::endl << "*************** DISTANCE STATS ***************" << std::endl;
	totalStatsDist.print();
	std::cout << std::endl << "--------------- PER-CLASS STATS ---------------" << std::endl;
	for (const auto& stat : classStatsDist) {
		std::cout << stat.first << std::endl;
		stat.second.print();
		std::cout << std::endl;
	}
	std::cout << "\ttime taken: " << t.getElapsedTime() << " seconds" << std::endl;
	std::cout << std::endl;
}

EvalStatsDist Evaluation::evaluate(const VoxelGrid& inputSDF, const DistanceField3f& gtDF, const DistanceField3f& predDF, float truncation /*= 2.5f*/)
{
	EvalStatsDist stats;
	unsigned int scalefactor = (unsigned int)(gtDF.getDimX() / inputSDF.getDimX());
	for (unsigned int z = 0; z < gtDF.getDimZ(); z++) {
		for (unsigned int y = 0; y < gtDF.getDimY(); y++) {
			for (unsigned int x = 0; x < gtDF.getDimX(); x++) {
				const Voxel& v = inputSDF(x / scalefactor, y / scalefactor, z / scalefactor);
				float gtVal = gtDF(x, y, z);
				float predVal = predDF(x, y, z);
				if ((v.weight < 1 || v.sdf < -1.0f) && (gtVal < truncation || predVal < truncation)) { //unknown, to be predicted 
					float diff = std::abs(gtVal - predVal);
					stats.l1sum += diff;
					stats.l2sum += diff * diff;
					stats.normalization++;
				}
			} //x
		} //y
	} //z
	return stats;
}

