#pragma  once
#include "DistanceFieldPyramid.h"

template<>
struct std::hash<ml::vec3ui> : public std::unary_function < ml::vec3ui, size_t > {
	size_t operator()(const ml::vec3ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1) ^ ((size_t)v.z * p2);
		return res;
	}
};

class FeatureDictionary {
public:

	FeatureDictionary(unsigned int maxK, float eps, unsigned int featureDim = (unsigned int)-1) :
		m_maxK(maxK),
		m_eps(eps),
		m_featureDimension(featureDim),
		m_search(NearestNeighborSearchFLANNf(200, 12))
	{
		m_bReduceDimension = (featureDim != (unsigned int)-1);
	}
	~FeatureDictionary() {}

	void build(const std::vector<DistanceFieldPyramid>& pyramids, unsigned int level, float occThresh);
	std::vector<std::pair<vec4ui, float>> kNearest(const Feature& query) {
		Feature searchQuery;
		if (m_bReduceDimension) m_pca.transform(query, m_featureDimension, searchQuery);
		else searchQuery = query;

		std::vector<unsigned int> nearestIndices = m_search.kNearest(searchQuery, m_maxK, m_eps);
		std::vector<float> nearestDists = m_search.getDistances((unsigned int)nearestIndices.size());

		std::vector<std::pair<vec4ui, float>> nearest(nearestIndices.size());
		for (unsigned int k = 0; k < nearestIndices.size(); k++) {
			nearest[k] = std::make_pair(m_globalIdx2Indices[nearestIndices[k]], nearestDists[k]);
		}
		return nearest;
	}
	std::vector<std::pair<vec4ui, float>> getBestCoherentMatch(const vec3ui& loc, const Grid3<vec4ui>& indices, const std::vector<float>& curFeature) {
		Feature searchQuery;
		if (m_bReduceDimension) m_pca.transform(curFeature, m_featureDimension, searchQuery);
		else searchQuery = curFeature;

		std::vector<std::pair<vec4ui, float>> candidateIndices;
		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					if (x == 0 && y == 0 && z == 0) continue;
					const vec3ui& idx = indices(loc.x + x, loc.y + y, loc.z + z).getVec3();
					if (idx.x != (unsigned int)-1) {
						vec3i cand((int)idx.x + x, (int)idx.y + y, (int)idx.z + z);
						if (cand.x >= 0 && cand.x < indices.getDimX() &&
							cand.y >= 0 && cand.y < indices.getDimY() &&
							cand.z >= 0 && cand.z < indices.getDimZ()) {
							auto it = m_indices2GlobalIdx.find(cand);
							if (it == m_indices2GlobalIdx.end()) continue; //center +/- radius goes out of bounds
							for (unsigned int k = 0; k < it->second.size(); k++) {
								float dist = computePatchDistance2(searchQuery, m_features[it->second[k]]); //TODO CHECK INDEXING HERE
								candidateIndices.push_back(std::make_pair(vec4ui(cand, k), dist));
							}
						}
					} //valid index
				} //x
			} //y
		} //z
		if (candidateIndices.empty()) return candidateIndices;
		std::sort(candidateIndices.begin(), candidateIndices.end(), [](const std::pair<vec4ui, float> &left, const std::pair<vec4ui, float> &right) {
			return fabs(left.second) < fabs(right.second);
		});
		if (candidateIndices.size() > m_maxK) candidateIndices.resize(m_maxK);
		return candidateIndices;
	}

	size_t getNumFeatures() const { return m_features.size(); }

	//using l2 (sum of squares of distances)
	static float computePatchDistance2(const Feature& f0, const Feature& f1) {
		MLIB_ASSERT(f0.size() == f1.size());
		float dist2 = 0.0f;
		for (unsigned int i = 0; i < f0.size(); i++)
			dist2 += (f0[i] - f1[i]) * (f0[i] - f1[i]);
		return dist2;
	}

	//actually would need for complete neighbors, and for hi-res complete neighbors
	////build using net features, T: feature from net (e.g., vec4f)
	//template<typename T>
	//void build(const std::vector<DistanceFieldPyramid>& pyramids, unsigned int level,
	//	const std::vector<std::string>& featureFiles, const vec3ui& dim) {
	//	for (unsigned int k = 0; k < featureFiles.size(); k++) {
	//		//read in features grid
	//		Grid3<T> featuresGrid;
	//		const std::string& featureFile = featureFiles[k];
	//		MLIB_ASSERT(util::fileExists(featureFile));
	//		BinaryDataStreamFile s(featureFile, false);
	//		featuresGrid.allocate(dim);
	//		s.readData((BYTE*)featuresGrid.getData(), sizeof(T)*featuresGrid.getNumElements());
	//		s.closeStream();
	//		DistanceFieldVis::convertFromTorch(featuresGrid);
	//
	//		const unsigned int ftDim = sizeof(T) / sizeof(float);//TODO something better...
	//
	//		//get feature for current model
	//		for (unsigned int z = 0; z < dim.z; z++) {
	//			for (unsigned int y = 0; y < dim.y; y++) {
	//				for (unsigned int x = 0; x < dim.x; x++) {
	//					vec3ui curIdx(x, y, z);
	//					//get empty status
	//					Feature ftOrig; FEATURE_STATUS status = pyramids[k].getFeature(level, curIdx, ftOrig);
	//					if (status == FEATURE_STATUS::STANDARD) {
	//						//index map
	//						m_globalIdx2Indices[(unsigned int)m_features.size()] = vec4ui(curIdx, k);
	//						auto it = m_indices2GlobalIdx.find(curIdx);
	//						if (it == m_indices2GlobalIdx.end()) { m_indices2GlobalIdx[curIdx] = std::vector<unsigned int>(pyramids.size(), (unsigned int)m_features.size()); }
	//						else								 { it->second[k] = (unsigned int)m_features.size(); }
	//						Feature ft(ftDim); for (unsigned int i = 0; i < ftDim; i++) ft[i] = featuresGrid(x, y, z)[i];
	//						m_features.push_back(ft); //feature
	//					}
	//				} //x
	//			} //y
	//		} //z
	//	}//models
	//
	//	//build nn search structure
	//	m_search.init(m_features, m_maxK);
	//}

private:
	float		 m_eps;
	unsigned int m_maxK;
	std::vector<Feature> m_features;

	//search structure
	NearestNeighborSearchFLANNf					m_search;
	std::unordered_map<unsigned int, vec4ui>	m_globalIdx2Indices;
	std::unordered_map<vec3ui, std::vector<unsigned int>> m_indices2GlobalIdx;

	//reduce dimensions
	bool			m_bReduceDimension;
	unsigned int	m_featureDimension;
	PCAf			m_pca;
};

