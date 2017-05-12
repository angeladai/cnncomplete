#pragma  once

//TODO angie: neighbors each need DF pyramid
typedef std::vector<float> Feature;
enum FEATURE_STATUS { //TODO naming
	STANDARD,
	EMPTY,
	TRUNCATION_ONLY
};

//#define USE_DIAG_FEATURES

class DistanceFieldPyramid {
public:

	//numLevels levels in pyramid, coarsest is dimension coarseDim x coarseDim x coarseDim
	//radiusFine for cur level, radiusCoarse for cur-1 level
	DistanceFieldPyramid() { clearData(); }
	DistanceFieldPyramid(unsigned int numLevels, unsigned int coarseDim, unsigned int radiusFine, unsigned int radiusCoarse, float truncationValue);
	DistanceFieldPyramid(unsigned int numLevels, unsigned int radiusFine, unsigned int radiusCoarse, const DistanceField3f& coarseDF);
	DistanceFieldPyramid(const std::vector<std::string>& dfFiles, unsigned int radiusFine, unsigned int radiusCoarse);
	~DistanceFieldPyramid() {}

	int getRadiusFine() const { return m_radiusFine; }
	int getRadiusCoarse() const { return m_radiusCoarse; }
	vec3ui getDimensions(unsigned int level) const { return m_pyramid[level].getDimensions(); }

	vec3ui convertLocationFromFineLevelToCoarseLevel(const vec3ui& loc, unsigned int fineLevel) const {
		return math::clamp(math::round(vec3f(loc) / m_scaleFactor), m_radiusCoarse, (int)m_pyramid[fineLevel - 1].getDimX() - m_radiusCoarse - 1);
	}
	void setValue(const vec3ui& loc, unsigned int level, float val) {
		m_pyramid[level](loc) = val;
	}
	float getValue(const vec3ui& loc, unsigned int level) const {
		return m_pyramid[level](loc);
	}

	//returns if status of feature
	FEATURE_STATUS getFeature(unsigned int level, const vec3ui& loc, Feature& feature, float occThresh) const;
	const DistanceField3f& getLevel(unsigned int level) const { return m_pyramid[level]; }

	void truncate(float truncation, bool useCoarseTruncationForAll);
	void applyReweighting();
	void undoReweighting();

	//upsample to next level
	void upsampleToNextLevel(unsigned int fromLevel);
	void setLevelTruncation(unsigned int level, float truncation, bool updateValues = true) {
		m_pyramid[level].setTruncation(truncation, updateValues);
	}

	void copyLevelFrom(const DistanceFieldPyramid& other, unsigned int level) {
		if (level > 0)  { MLIB_ASSERT(other.getDimensions(level - 1) == getDimensions(level - 1)); }
		m_pyramid[level] = other.getLevel(level);
	}

	void saveLevelToDFFile(const std::string& filename, unsigned int level) const;
	void saveToFile(const std::string& filename) const;
	void loadFromFile(const std::string& filename);
	//save df at level level
	void saveLevelToFile(const std::string& filename, unsigned int level) const;
	void loadLevelFromFile(const std::string& filename, unsigned int level);

	//debug stats
	void printTruncationStats(float occThresh) const {
		std::cout << "pyramid truncation stats" << std::endl;
		for (unsigned int level = 0; level < m_numLevels; level++) {
			const DistanceField3f& df = m_pyramid[level];
			const float truncation = df.getTruncation();
			unsigned int numTruncationOnlyFeatures = 0;
			for (unsigned int z = m_radiusFine; z < df.getDimZ()-m_radiusFine; z++) {
				for (unsigned int y = m_radiusFine; y < df.getDimY()-m_radiusFine; y++) {
					for (unsigned int x = m_radiusFine; x < df.getDimX()-m_radiusFine; x++) {
						Feature ft; 
						FEATURE_STATUS status = getFeature(level, vec3ui(x, y, z), ft, occThresh);
						if (status == FEATURE_STATUS::TRUNCATION_ONLY) numTruncationOnlyFeatures++;
					} //x
				} //y
			} //z
			const unsigned int numTotal = (unsigned int)((df.getDimX() - 2*m_radiusFine) * (df.getDimY() - 2*m_radiusFine) * (df.getDimZ() - 2*m_radiusFine));
			std::cout << "\tlevel " << level << ": " << (float)numTruncationOnlyFeatures / (float)numTotal << " (" << numTruncationOnlyFeatures << " of " << numTotal << ")" << std::endl;
		} //levels
	}

	static void getUpsampled(DistanceField3f& df, float scaleFactor);

private:
	void setFeatureDims(unsigned int radiusFine, unsigned int radiusCoarse) {
#ifdef USE_DIAG_FEATURES
		MLIB_ASSERT(radiusFine == 2 && radiusCoarse == 1); // only supported values
#endif
		m_radiusFine = (int)radiusFine;
		m_dimFine = 2 * radiusFine + 1;
		m_radiusCoarse = (int)radiusCoarse;
		m_dimCoarse = 2 * radiusCoarse + 1;
	}
	void alloc(unsigned int numLevels) {
		m_numLevels = numLevels;
		m_pyramid.resize(m_numLevels);

	}
	void clearData() {
		m_numLevels = 0;
		m_pyramid.clear();

		m_scaleFactor = 0.0f;
		m_radiusFine = 0;
		m_dimFine = 0;
		m_radiusCoarse = 0;
		m_dimCoarse = 0;
	}

	unsigned int					m_numLevels;
	std::vector<DistanceField3f>	m_pyramid;		//[0]coarse, [m_numLevels-1]fine
	float							m_scaleFactor;  //dim1 / dim0

	int								m_radiusFine, m_radiusCoarse;
	unsigned int					m_dimFine, m_dimCoarse;
};

