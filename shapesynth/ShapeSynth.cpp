// PatchOpt.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "VoxelGrid.h"
#include "GlobalAppState.h"
#include "PatchHelper.h"
#include "DistanceFieldPyramid.h"
#include "FeatureDictionary.h"


void refinePrediction(const VoxelGrid& inputSDF, const DistanceField3f& pred, const std::vector<DistanceFieldPyramid>& neighborDFs);

int _tmain(int argc, _TCHAR* argv[])
{
	if (argc == 3) {
		const std::wstring defaultParamFile = std::wstring(argv[1]);
		const std::wstring caseParamFile = std::wstring(argv[2]);
		ParameterFile parameterFileGlobalApp(std::string(defaultParamFile.begin(), defaultParamFile.end()));
		parameterFileGlobalApp.addParameterFile(std::string(caseParamFile.begin(), caseParamFile.end()));
		GlobalAppState::get().readMembers(parameterFileGlobalApp);
	}
	else {
		const std::string fileNameDescGlobalApp = "zParametersDefault.txt";
		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		GlobalAppState::get().readMembers(parameterFileGlobalApp);
	}
	
	try {
		std::ofstream out;
		{
			const auto parts = util::splitPath(GlobalAppState::get().s_inputSdfFile);
			const std::string outputDir = GlobalAppState::get().s_outputDir + parts[parts.size() - 2] + "/" + util::removeExtensions(parts.back()) + "/";
			//if (util::directoryExists(outputDir)) {
			//	std::cerr << outputDir + " ALREADY EXISTS, SKIPPING" << std::endl;
			//	return 0;
			//}
			//else {
			//	//std::cerr << "PROCESSING..." << std::endl;
			//	std::cout << outputDir << std::endl;
			//}
			if (!util::directoryExists(outputDir)) util::makeDirectory(outputDir);
			if (!util::directoryExists(outputDir)) { std::cout << "ERROR: failed to make output directory"; return -1; }
			GlobalAppState::get().s_outputDir = outputDir;
			if (argc == 3) {
				//redirect stdout to file
				out.open(outputDir + "patchopt.log");
				if (!out.is_open()) throw MLIB_EXCEPTION("failed to open log file");
				//std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
				std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt
			}
		}
		GlobalAppState::get().print();

		const unsigned int numLevels = GlobalAppState::get().s_numLevels;
		const unsigned int predDim = 32;
		const std::string baseDataDir = GlobalAppState::get().s_baseDataDir;
		const std::vector<std::string>& neighborFiles = GlobalAppState::get().s_neighborFiles;
		std::vector<std::vector<std::string>> dfNeighborFiles(neighborFiles.size());	//coarse-to-fine for each neighbor
		for (unsigned int i = 0; i < neighborFiles.size(); i++) {
			unsigned int multiplier = 1;
			for (unsigned int level = 0; level < numLevels; level++) {
				const auto parts = util::splitPath(neighborFiles[i]); MLIB_ASSERT(parts.size() == 3);
				const std::string& classId = parts.front();		const std::string& modelId = parts[1];
				dfNeighborFiles[i].push_back(baseDataDir + "shapenet_dim" + std::to_string(multiplier * predDim) + "_df/" + classId + "/" + modelId + "__0__.df");
				if (!util::fileExists(dfNeighborFiles[i].back())) throw MLIB_EXCEPTION("df neighbor file (" + dfNeighborFiles[i].back() + ") does not exist!");
				multiplier *= 2;
			} //levels
		} //neighbors

		// load in data
		const std::string completionFile = GlobalAppState::get().s_completedDfFile;
		VoxelGrid inputSDF(vec3ul(0, 0, 0), mat4f::identity(), 0.0f, 0.0f, 0.0f); inputSDF.loadFromFile(GlobalAppState::get().s_inputSdfFile);
		const float truncation = GlobalAppState::get().s_truncationDistance * (inputSDF.getDimX() - 1 - 2 * 3);
		//load prediction
		DistanceField3f pred;
		{
			BinaryDataStreamFile s(completionFile, BinaryDataBuffer::Mode::read_flag);
			s >> pred; s.close();
		}
		//construct neighbor pyramids
		std::vector<DistanceFieldPyramid> neighborPyramids;
		for (unsigned int i = 0; i < dfNeighborFiles.size(); i++) {
			neighborPyramids.push_back(DistanceFieldPyramid(dfNeighborFiles[i], GlobalAppState::get().s_radiusFine, GlobalAppState::get().s_radiusCoarse));
		}

		pred.setTruncation(truncation); //done above
		for (DistanceFieldPyramid& p : neighborPyramids) p.truncate(GlobalAppState::get().s_truncationDistance, false); //can't use trunc in m since coarse truncation will limit max truncation for synthesized

		PatchHelper::applyReweighting(pred);
		for (DistanceFieldPyramid& p : neighborPyramids) p.applyReweighting();

		refinePrediction(inputSDF, pred, neighborPyramids);
	}
	catch (MLibException& e)
	{
		std::stringstream ss;
		ss << "exception caught:" << e.what() << std::endl;
		std::cout << ss.str() << std::endl;
	}

	return 0;
}

void refinePrediction(const VoxelGrid& inputSDF, const DistanceField3f& pred, const std::vector<DistanceFieldPyramid>& neighborPyramids)
{
	//params
	const unsigned int maxK = GlobalAppState::get().s_searchMaxK;
	const float eps = GlobalAppState::get().s_searchEps;
	const bool useCoherence = GlobalAppState::get().s_useCoherence;	const float kappa = GlobalAppState::get().s_coherenceKappa;
	const unsigned int numLevels = GlobalAppState::get().s_numLevels;
	const unsigned int numIters = GlobalAppState::get().s_numItersPerLevel;
	const unsigned int featureDim = GlobalAppState::get().s_featureDim;
	const float occThresh = PatchHelper::reweight(1.0f);

	//debugging output
	const bool printVis = GlobalAppState::get().s_verbose;
	const std::string outputDir = GlobalAppState::get().s_outputDir;
	if (!util::directoryExists(outputDir)) util::makeDirectory(outputDir);
	if (printVis) { //save out input sdf
		BinaryGrid3 inputBG = inputSDF.toBinaryGridOccupied(1, 1.0f);
		MeshIOf::saveToFile(outputDir + "input-sdf.ply", TriMeshf(inputBG).computeMeshData());
	}

	//build pyramids
	DistanceFieldPyramid refTarget(numLevels, GlobalAppState::get().s_radiusFine, GlobalAppState::get().s_radiusCoarse, pred);
	DistanceFieldPyramid targetPyramid(numLevels, (unsigned int)pred.getDimX(), GlobalAppState::get().s_radiusFine, GlobalAppState::get().s_radiusCoarse, pred.getTruncation());
	targetPyramid.setLevelTruncation(0, pred.getTruncation(), false);
	const unsigned int startLevel = GlobalAppState::get().s_startLevel;
	if (startLevel >= numLevels) throw MLIB_EXCEPTION("cannot start at level greater than #levels");
	if (startLevel > 0) {
		for (unsigned int level = 0; level < startLevel; level++) {
			const std::string cachedFile = outputDir + "synth-" + std::to_string(level) + ".level";
			targetPyramid.loadLevelFromFile(cachedFile, level);
		}
	}

	for (unsigned int level = startLevel; level < numLevels; level++) {
		std::cout << "[ level " << level << " ]" << std::endl;
		const int radiusFine = targetPyramid.getRadiusFine();		const int radiusCoarse = targetPyramid.getRadiusCoarse();

		//build nn for finding patches
		Timer t; std::cout << "\tbuilding nn... ";
		//TODO have input patches in the search here too
		FeatureDictionary featureDictionary(maxK, eps, featureDim); featureDictionary.build(neighborPyramids, level, occThresh);
		t.stop(); std::cout << "done! (" << featureDictionary.getNumFeatures() << " features, " << t.getElapsedTimeMS() << " ms)" << std::endl;

		if (printVis) {
			PatchHelper::visualizePyramid(refTarget, level, outputDir + "init-level-" + std::to_string(level), 1.0f);
			if (level > 0) {
				PatchHelper::visualizePyramid(refTarget, level - 1, outputDir + "init-low-level-" + std::to_string(level), 1.0f);
				//low-res neighbor not much different from hi-res neighbor
			}
		}

		//synthesize!
		t.start();
		const vec3ui dims = targetPyramid.getDimensions(level);
		Grid3<vec4ui> indices(dims); indices.setValues(vec4ui((unsigned int)-1));
		for (unsigned int iter = 0; iter < numIters; iter++) {
			//#pragma omp parallel for
			for (int z = radiusFine; z < (int)dims.z - radiusFine; z++) {
				for (unsigned int y = radiusFine; y < dims.y - radiusFine; y++) {
					for (unsigned int x = radiusFine; x < dims.x - radiusFine; x++) {
						vec3ui curLoc(x, y, z);
						Feature curFeature;
						FEATURE_STATUS status = refTarget.getFeature(level, curLoc, curFeature, occThresh);
						if (status != FEATURE_STATUS::STANDARD) continue; //don't bother for completely empty
						//get best match by l2
						std::vector<std::pair<vec4ui, float>> bestMatch = featureDictionary.kNearest(curFeature); //(xyz, neighbor idx)
						MLIB_ASSERT(!bestMatch.empty());
						unsigned int bestMatchIdx = 0;
						vec4ui bestIdx = bestMatch[bestMatchIdx].first;

						////check if creating empty where currently occ or vice versa
						//float bestVal = std::exp(neighborPyramids[bestIdx.w].getValue(bestIdx.getVec3(), level)) - 1;
						//float curVal = std::exp(refTarget.getValue(curLoc, level)) - 1;
						//if (bestVal > occThresh && curVal <= occThresh) { //creating empty space where there wasn't...
						//	for (unsigned int k = 1; k < bestMatch.size(); k++) {
						//		bestVal = std::exp(neighborPyramids[bestMatch[k].first.w].getValue(bestMatch[k].first.getVec3(), level)) - 1;
						//		if (bestVal <= occThresh)  {
						//			bestIdx = bestMatch[k].first;
						//			break;
						//		}
						//	}
						//}
						//else if (bestVal <= occThresh && curVal > occThresh) { //creating occ where was empty...
						//	for (unsigned int k = 1; k < bestMatch.size(); k++) {
						//		bestVal = std::exp(neighborPyramids[bestMatch[k].first.w].getValue(bestMatch[k].first.getVec3(), level)) - 1;
						//		if (bestVal > occThresh) {
						//			bestIdx = bestMatch[k].first;
						//			break;
						//		}
						//	}
						//}

						////debugging
						//if (bestVal > occThresh && curVal <= occThresh) {
						//	const std::string debugDir = "debug/";
						//	PatchHelper::visualizeWorldSpacePatchDF(debugDir + "query", refTarget, level, curLoc, vec4f(0.0f, 0.0f, 1.0f, 1.0f), std::log(3.0f + 1.0f));		//blue
						//	BinaryGrid3 bgCenter(dims); bgCenter.setVoxel(curLoc);
						//	MeshIOf::saveToFile(debugDir + "query-center.ply", TriMeshf(bgCenter, mat4f::identity(), false, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).getMeshData()); //green
						//	for (unsigned int k = 0; k < bestMatch.size(); k++) {
						//		PatchHelper::visualizeWorldSpacePatchDF(debugDir + "match" + std::to_string(k), neighborPyramids[bestMatch[k].first.w], level, bestMatch[k].first.getVec3(), vec4f(1.0f, 1.0f, 1.0f, 1.0f), std::log(3.0f + 1.0f));	//white
						//		BinaryGrid3 bgCenterMatch(dims); bgCenterMatch.setVoxel(bestMatch[k].first.getVec3());
						//		MeshIOf::saveToFile(debugDir + "match" + std::to_string(k) + "-center.ply", TriMeshf(bgCenterMatch, mat4f::identity(), false, vec4f(1.0f, 1.0f, 0.0f, 1.0f)).getMeshData()); //yellow
						//	}
						//	int a = 5;
						//}
						////debugging

						if (useCoherence) { //get best coherent match
							std::vector<std::pair<vec4ui, float>> bestCoherent = featureDictionary.getBestCoherentMatch(curLoc, indices, curFeature); //TODO only coherent from neighbor and not all neighbors?
							if (!bestCoherent.empty()) {
								if (bestCoherent.front().second < bestMatch[bestMatchIdx].second * (1.0f + kappa))
									bestIdx = bestCoherent.front().first;
							} //valid coherence value
						} //useCoherence

						indices(curLoc) = bestIdx;
						targetPyramid.setValue(curLoc, level, neighborPyramids[bestIdx.w].getValue(bestIdx.getVec3(), level));
					} //x
				} //y
			} //z
			if (printVis) {
				PatchHelper::visualizePyramid(targetPyramid, level, outputDir + "level-" + std::to_string(level) + "_iter-" + std::to_string(iter), 1.0f);
				if (level > 0)
					PatchHelper::visualizePyramid(targetPyramid, level - 1, outputDir + "level-" + std::to_string(level) + "_iter-" + std::to_string(iter) + "_coarse", 1.0f);
			}
			std::cout << "\tdone iter " << iter << std::endl;
			refTarget.copyLevelFrom(targetPyramid, level);
		} //iters
		t.stop();
		std::cout << "\tsynthesis time = " << t.getElapsedTime() << " s" << std::endl;

		if (level + 1 < numLevels) { //upsample to next level
			targetPyramid.undoReweighting();
			targetPyramid.upsampleToNextLevel(level);
			targetPyramid.applyReweighting();
			targetPyramid.setLevelTruncation(level + 1, pred.getTruncation(), false);
			refTarget.copyLevelFrom(targetPyramid, level + 1);
		}
		targetPyramid.saveLevelToFile(outputDir + "synth-" + std::to_string(level) + ".level", level);
		{ //save indices
			BinaryDataStreamFile s(outputDir + "synth-" + std::to_string(level) + ".indices", true);
			s << indices;
		}
		if (level == numLevels) {
			targetPyramid.saveLevelToDFFile(outputDir + "synth.df", level);
		}
	}
}