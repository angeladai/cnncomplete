
#include "stdafx.h"
#include "DistanceFieldPyramid.h"
#include "PatchHelper.h"

DistanceFieldPyramid::DistanceFieldPyramid(unsigned int numLevels, unsigned int coarseDim, unsigned int radiusFine, unsigned int radiusCoarse, float truncationValue)
{
	clearData();
	alloc(numLevels);
	setFeatureDims(radiusFine, radiusCoarse);
	m_pyramid.front().allocate(coarseDim, coarseDim, coarseDim, truncationValue);
	m_scaleFactor = 2.0f; //TODO make param
}

DistanceFieldPyramid::DistanceFieldPyramid(unsigned int numLevels, unsigned int radiusFine, unsigned int radiusCoarse, const DistanceField3f& coarseDF)
{
	clearData();
	alloc(numLevels);
	setFeatureDims(radiusFine, radiusCoarse);
	m_pyramid.front() = coarseDF;
	m_scaleFactor = 2.0f; //TODO make param
}

DistanceFieldPyramid::DistanceFieldPyramid(const std::vector<std::string>& dfFiles, unsigned int radiusFine, unsigned int radiusCoarse)
{
	clearData();
	setFeatureDims(radiusFine, radiusCoarse);
	m_scaleFactor = 2.0f; //TODO make param
	m_numLevels = (unsigned int)dfFiles.size();
	m_pyramid.resize(m_numLevels);
	for (unsigned int i = 0; i < m_numLevels; i++) {
		BinaryDataStreamFile s(dfFiles[i], false);
		s >> m_pyramid[i];
		s.close();
	}
}

FEATURE_STATUS DistanceFieldPyramid::getFeature(unsigned int level, const vec3ui& loc, Feature& feature, float occThresh) const
{
	bool empty = true, allTrunc = true;

#ifdef USE_DIAG_FEATURES
	if (level == 0) feature.resize(9);
	else			feature.resize(9 + 9);

	const std::vector<vec3i> offsetsFine = { 
		vec3i(-1, -1, -1), vec3i(-2, -1, -1), vec3i(-1, -2, -1),	//-z, top left
		vec3i(1, -1, -1), vec3i(2, -1, -1), vec3i(1, -2, -1),		//-z, top right
		vec3i(-1, 1, -1), vec3i(-2, 1, -1), vec3i(-1, 2, -1),		//-z, bottom left
		vec3i(1, 1, -1), vec3i(2, 1, -1), vec3i(1, 2, -1),			//-z, bottom right
		vec3i(-1, -1, 1), vec3i(-2, -1, 1), vec3i(-1, -2, 1),		//+z, top left
		vec3i(1, -1, 1), vec3i(2, -1, 1), vec3i(1, -2, 1),			//+z, top right
		vec3i(-1, 1, 1), vec3i(-2, 1, 1), vec3i(-1, 2, 1),			//+z, bottom left
		vec3i(1, 1, 1), vec3i(2, 1, 1), vec3i(1, 2, 1)				//+z, bottom right
	};
	const std::vector<vec3i> offsetsCoarse = { 
		vec3i(-1, -1, -1), vec3i(1, -1, -1), vec3i(-1, 1, -1), vec3i(1, 1, -1),	//-z, top left, top right, bottom left, bottom right
		vec3i(-1, -1, 1), vec3i(1, -1, 1), vec3i(-1, 1, 1), vec3i(1, 1, 1)		//+z, top left, top right, bottom left, bottom right
	};	
	//get cur level
	const DistanceField3f& df = m_pyramid[level];
	feature[0] = df(loc); //center value
	for (unsigned int i = 0; i < 8; i++) {
		float val = 0.0f;
		for (unsigned int j = 0; j < 3; j++) {
			const float v = df(vec3i(loc) + offsetsFine[i * 3 + j]);
			if (v < df.getTruncation()) {
				allTrunc = false;
				if (j == 0 && v <= occThresh) empty = false;
			}
			val += v;
		}
		val /= 3.0f; //todo necessary???
		feature[1 + i] = val;
	}
	if (level > 0) { //get coarser level
		const DistanceField3f& dfLow = m_pyramid[level - 1];
		vec3ui locLow = convertLocationFromFineLevelToCoarseLevel(loc, level);
		const unsigned int offset = 9;
		feature[offset] = dfLow(locLow); //center value
		for (unsigned int i = 0; i < offsetsCoarse.size(); i++) {
			feature[offset + 1 + i] = dfLow(locLow + offsetsCoarse[i]);
		}
	}
#else
	if (level == 0) feature.resize(m_dimFine * m_dimFine * m_dimFine);
	else			feature.resize(m_dimFine * m_dimFine * m_dimFine + m_dimCoarse * m_dimCoarse * m_dimCoarse);

	//get cur level
	const DistanceField3f& df = m_pyramid[level];
	for (int rz = -m_radiusFine; rz <= m_radiusFine; rz++) {
		for (int ry = -m_radiusFine; ry <= m_radiusFine; ry++) {
			for (int rx = -m_radiusFine; rx <= m_radiusFine; rx++) {
				const float val = df(rx + loc.x, ry + loc.y, rz + loc.z);
				feature[(rz + m_radiusFine)*m_dimFine*m_dimFine + (rx + m_radiusFine)*m_dimFine + (ry + m_radiusFine)] = val;
				if (val < df.getTruncation()) allTrunc = false;
				//if (val <= occThresh) empty = false;
				if (std::abs(rx) <= 1 && std::abs(ry) <= 1 && std::abs(rz) <= 1 && val <= occThresh) empty = false;
			}
		}
	}
	if (level > 0) { //get coarser level
		const DistanceField3f& dfLow = m_pyramid[level - 1];
		vec3ui locLow = convertLocationFromFineLevelToCoarseLevel(loc, level);
		const unsigned int offset = m_dimFine * m_dimFine * m_dimFine;
		for (int rz = -m_radiusCoarse; rz <= m_radiusCoarse; rz++) {
			for (int ry = -m_radiusCoarse; ry <= m_radiusCoarse; ry++) {
				for (int rx = -m_radiusCoarse; rx <= m_radiusCoarse; rx++) {
					feature[offset + (rz + m_radiusCoarse)*m_dimCoarse*m_dimCoarse + (rx + m_radiusCoarse)*m_dimCoarse + (ry + m_radiusCoarse)] = dfLow(rx + locLow.x, ry + locLow.y, rz + locLow.z);
					//if (std::abs(rx) <= 1 && std::abs(ry) <= 1 && std::abs(rz) <= 1 && dfLow(rx + locLow.x, ry + locLow.y, rz + locLow.z) <= occThresh) empty = false;
				}
			}
		}
	}
#endif
	if (allTrunc)	return FEATURE_STATUS::TRUNCATION_ONLY;
	if (empty)		return FEATURE_STATUS::EMPTY;
	return FEATURE_STATUS::STANDARD;
}

void DistanceFieldPyramid::truncate(float truncationDistance, bool useCoarseTruncationForAll) {
	const float coarseTruncation = truncationDistance * (m_pyramid.front().getDimX() - 1 - 2 * 3);
	for (unsigned int i = 0; i < m_pyramid.size(); i++) {
		float truncation = useCoarseTruncationForAll ? coarseTruncation : truncationDistance * (m_pyramid[i].getDimX() - 1 - 2 * 3);
		m_pyramid[i].setTruncation(truncation);
	}
}

void DistanceFieldPyramid::applyReweighting() {
	for (unsigned int i = 0; i < m_pyramid.size(); i++) {
		PatchHelper::applyReweighting(m_pyramid[i]);
	}
}

void DistanceFieldPyramid::undoReweighting() {
	for (unsigned int i = 0; i < m_pyramid.size(); i++) {
		PatchHelper::undoReweighting(m_pyramid[i]);
	}
}

void DistanceFieldPyramid::saveToFile(const std::string& filename) const
{
	BinaryDataStreamFile s(filename, true);
	s << m_radiusFine << m_radiusCoarse;
	s << m_dimFine << m_dimCoarse;
	s << m_scaleFactor;
	s << m_numLevels;
	for (const DistanceField3f& df : m_pyramid) {
		s << df.getTruncation(); //TODO hack since only stores grid values for s << df
		s << df;
	}
}

void DistanceFieldPyramid::loadFromFile(const std::string& filename)
{
	BinaryDataStreamFile s(filename, false);
	s >> m_radiusFine >> m_radiusCoarse;
	s >> m_dimFine >> m_dimCoarse;
	s >> m_scaleFactor;
	s >> m_numLevels;
	m_pyramid.resize(m_numLevels);
	for (DistanceField3f& df : m_pyramid) {
		float truncation; s >> truncation;
		s >> df;
		df.setTruncation(truncation, false);
	}
}

void DistanceFieldPyramid::saveLevelToFile(const std::string& filename, unsigned int level) const
{
	BinaryDataStreamFile s(filename, true);
	s << m_pyramid[level].getTruncation();
	s << m_pyramid[level];
}

void DistanceFieldPyramid::loadLevelFromFile(const std::string& filename, unsigned int level)
{
	BinaryDataStreamFile s(filename, false);
	float truncation;
	s >> truncation;
	s >> m_pyramid[level];
	m_pyramid[level].setTruncation(truncation, false);
}

void DistanceFieldPyramid::upsampleToNextLevel(unsigned int fromLevel)
{
	vec3ui newDims = math::round(vec3f(m_pyramid[fromLevel].getDimensions()) * m_scaleFactor);
	m_pyramid[fromLevel + 1] = m_pyramid[fromLevel].upsample(newDims);

	//wtf
	for (unsigned int z = 0; z < newDims.z; z++) {
		for (unsigned int y = 0; y < newDims.y; y++) {
			for (unsigned int x = 0; x < newDims.x; x++) {
				float d = m_pyramid[fromLevel + 1](x, y, z);
				if (d > 1.0f) m_pyramid[fromLevel + 1](x, y, z) = d * m_scaleFactor;
			}
		}
	}
}

void DistanceFieldPyramid::getUpsampled(DistanceField3f& df, float scaleFactor)
{
	vec3ui newDims = math::round(vec3f(df.getDimensions()) * scaleFactor);
	df = df.upsample(newDims);

	for (unsigned int z = 0; z < newDims.z; z++) {
		for (unsigned int y = 0; y < newDims.y; y++) {
			for (unsigned int x = 0; x < newDims.x; x++) {
				float d = df(x, y, z);
				if (d > 1.0f) df(x, y, z) = d * scaleFactor;
			}
		}
	}
}

void DistanceFieldPyramid::saveLevelToDFFile(const std::string& filename, unsigned int level) const
{
	BinaryDataStreamFile s(filename, true);
	s << m_pyramid[level];
	s.close();
}
