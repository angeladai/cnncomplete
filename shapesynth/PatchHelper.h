#pragma  once

#include "DistanceFieldPyramid.h"

class PatchHelper {
public:

	//static float reweight(float v) {
	//	return std::log(v + 1);
	//}
	//static float undoReweight(float v) {
	//	return std::exp(v) - 1;
	//}
	//static void applyReweighting(DistanceField3f& df) {
	//	for (unsigned int z = 0; z < df.getDimZ(); z++) {
	//		for (unsigned int y = 0; y < df.getDimY(); y++) {
	//			for (unsigned int x = 0; x < df.getDimX(); x++) {
	//				float v = df(x, y, z);
	//				df(x, y, z) = reweight(v);
	//			}
	//		}
	//	}
	//	const float newTruncation = std::log(df.getTruncation() + 1);
	//	df.setTruncation(newTruncation, false);
	//}
	//static void undoReweighting(DistanceField3f& df) {
	//	for (unsigned int z = 0; z < df.getDimZ(); z++) {
	//		for (unsigned int y = 0; y < df.getDimY(); y++) {
	//			for (unsigned int x = 0; x < df.getDimX(); x++) {
	//				float v = df(x, y, z);
	//				df(x, y, z) = undoReweight(v);
	//			}
	//		}
	//	}
	//	const float newTruncation = std::exp(df.getTruncation()) - 1;
	//	df.setTruncation(newTruncation, false);
	//}

	static float reweight(float v) {
		return v;
	}
	static float undoReweight(float v) {
		return v;
	}
	static void applyReweighting(DistanceField3f& df) {
		
	}
	static void undoReweighting(DistanceField3f& df) {
		
	}

	//assumes square grids, int scaling factor
	static void upsampleFromSourceDF(const Grid3<vec4ui>& curIndices, const DistanceField3f& sourceDF, DistanceField3f& result) {
		unsigned int scaleToNew = (unsigned int)(sourceDF.getDimX() / curIndices.getDimX());
		MLIB_ASSERT(scaleToNew * curIndices.getDimX() == sourceDF.getDimX());
		result.allocate(sourceDF.getDimensions()); result.setValues(sourceDF.getTruncation());
		for (unsigned int z = 0; z < curIndices.getDimZ(); z++) {
			for (unsigned int y = 0; y < curIndices.getDimY(); y++) {
				for (unsigned int x = 0; x < curIndices.getDimX(); x++) {
					vec3ui index = curIndices(x, y, z).getVec3();
					if (index.x == (unsigned int)-1) continue; //valid index (these should be border cases)
					for (unsigned int sz = 0; sz < scaleToNew; sz++) {
						for (unsigned int sy = 0; sy < scaleToNew; sy++) {
							for (unsigned int sx = 0; sx < scaleToNew; sx++) {
								vec3ui indexNew = index * scaleToNew + vec3ui(sx, sy, sz);
								result(x*scaleToNew + sx, y*scaleToNew + sy, z*scaleToNew + sz) = sourceDF(indexNew);
							} //sx
						} //sy
					} //sz
				} //x
			} //y
		} //z
	}

	//vis stuff
	static void visualizePyramid(const DistanceFieldPyramid& pyramid, unsigned int level, const std::string& prefix, float occThreshUnweighted) {
		DistanceField3f curDF = pyramid.getLevel(level);	undoReweighting(curDF);
		BinaryGrid3 bgCur = curDF.computeBinaryGrid(occThreshUnweighted);
		MeshIOf::saveToFile(prefix + ".ply", TriMeshf(bgCur).computeMeshData());
		//DistanceFieldVis::rayCastBinaryGrid(bgCur, prefix + ".png");
	}


	static void getWorldSpacePatchDF(const DistanceField3f& ref, const vec3ui& center, int radius, DistanceField3f& res, float clearValue = 0.0f) {
		res.allocate(ref.getDimensions()); 
		res.setValues(clearValue);
		for (int zz = -radius; zz <= radius; zz++) {
			for (int yy = -radius; yy <= radius; yy++) {
				for (int xx = -radius; xx <= radius; xx++)
					res(center.x + xx, center.y + yy, center.z + zz) = ref(center.x + xx, center.y + yy, center.z + zz);
			}
		}
	}

	static void getPatchDF(const DistanceField3f& ref, const vec3ui& center, int radius, DistanceField3f& res) {
		const unsigned int size = 2 * radius + 1;
		res.allocate(size, size, size);
		for (int zz = -radius; zz <= radius; zz++) {
			for (int yy = -radius; yy <= radius; yy++) {
				for (int xx = -radius; xx <= radius; xx++)
					res(radius + xx, radius + yy, radius + zz) = ref(center.x + xx, center.y + yy, center.z + zz);
			}
		}
	}

	static void visualizeWorldSpacePatchDF(const std::string& prefix, const DistanceFieldPyramid& pyramid, unsigned int level,
		const vec3ui& center, vec4f& color, float clearValue = 0.0f) {
		DistanceField3f patchDF;
		getWorldSpacePatchDF(pyramid.getLevel(level), center, pyramid.getRadiusFine(), patchDF, clearValue);
		undoReweighting(patchDF);
		BinaryGrid3 bg = patchDF.computeBinaryGrid(1.0f);
		MeshIOf::saveToFile(prefix + ".ply", TriMeshf(bg, mat4f::identity(), false, color).computeMeshData());
		//DistanceFieldVis::rayCastBinaryGrid(bg, prefix + ".png");
		if (level > 0) {
			vec3ui centerLo = pyramid.convertLocationFromFineLevelToCoarseLevel(center, level);
			getWorldSpacePatchDF(pyramid.getLevel(level-1), centerLo, pyramid.getRadiusCoarse(), patchDF, clearValue);
			undoReweighting(patchDF);
			bg = patchDF.computeBinaryGrid(1.0f);
			MeshIOf::saveToFile(prefix + "-coarse.ply", TriMeshf(bg, mat4f::identity(), false, color).computeMeshData());
			//DistanceFieldVis::rayCastBinaryGrid(bg, prefix + "-coarse.png");
		}
	}

	static void visualizeIndices(const Grid3<vec4ui>& indices, const DistanceField3f& df, const std::string& filename, float occThresh);
	static void visualizeIndices(const DistanceField3f& df, const std::string& filename, float occThresh);

private:
};

