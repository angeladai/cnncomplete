
#include "stdafx.h"
#include "FeatureDictionary.h"
#include "GlobalAppState.h"


void FeatureDictionary::build(const std::vector<DistanceFieldPyramid>& pyramids, unsigned int level, float occThresh)
{
	MLIB_ASSERT(!pyramids.empty());
	//collect all valid features
	const int radiusFine = pyramids.front().getRadiusFine();
	const int radiusCoarse = pyramids.front().getRadiusCoarse();
	const vec3ui dims = pyramids.front().getDimensions(level);

	for (unsigned int k = 0; k < pyramids.size(); k++) {
		for (unsigned int z = radiusFine; z < dims.z - radiusFine; z++) {
			for (unsigned int y = radiusFine; y < dims.y - radiusFine; y++) {
				for (unsigned int x = radiusFine; x < dims.x - radiusFine; x++) {
					vec3ui curIdx(x, y, z);
					//get feature
					Feature ft;
					FEATURE_STATUS status = pyramids[k].getFeature(level, curIdx, ft, occThresh);
					if (status == FEATURE_STATUS::STANDARD) {
						//index map
						m_globalIdx2Indices[(unsigned int)m_features.size()] = vec4ui(curIdx, k);
						auto it = m_indices2GlobalIdx.find(curIdx);
						if (it == m_indices2GlobalIdx.end()) { m_indices2GlobalIdx[curIdx] = std::vector<unsigned int>(pyramids.size(), (unsigned int)m_features.size()); }
						else								 { it->second[k] = (unsigned int)m_features.size(); }
						m_features.push_back(ft); //feature
					}
				} //x
			} //y
		} //z
	} //pyramids

	//save out 
	const std::string cacheRootDir = "cache/"; if (!util::directoryExists(cacheRootDir)) util::makeDirectory(cacheRootDir);
	const std::string cacheDir = cacheRootDir + util::removeExtensions(util::fileNameFromPath(GlobalAppState::get().s_inputSdfFile)) + "/"; if (!util::directoryExists(cacheDir)) util::makeDirectory(cacheDir);
	{
		BinaryDataStreamFile s(cacheDir + "features-" + std::to_string(level) + ".bin", true);
		s << m_features;
	}
	if (level == 0 && m_bReduceDimension)
		m_featureDimension = (unsigned int)((float)m_featureDimension * (float)(radiusFine*radiusFine*radiusFine) / (float)(radiusFine*radiusFine*radiusFine + radiusCoarse*radiusCoarse*radiusCoarse));
	if (m_featureDimension >= m_features.front().size()) m_bReduceDimension = false;
	if (m_bReduceDimension) {
		std::cout << "[ reduce feature dim from " << m_features.front().size() << " to " << m_featureDimension << " ]" << std::endl;
		auto eigenSolver = [](const DenseMatrixf &m) { return m.eigenSystem(); }; //eigensolverNR
		std::vector<const float*> v(m_features.size());
		for (unsigned int i = 0; i < m_features.size(); i++)
			v[i] = &m_features[i][0];

		m_pca.init(v, m_featureDimension, eigenSolver);
		m_pca.save(cacheDir + "pca-" + std::to_string(level) + ".bin");
		//transform
		std::vector<Feature> reduced(m_features.size());
		for (unsigned int i = 0; i < m_features.size(); i++)
			m_pca.transform(m_features[i], m_featureDimension, reduced[i]);
		m_features = reduced;
	}

	//build nn search structure
	m_search.init(m_features, m_maxK);
}
