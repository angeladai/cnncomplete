
#include "stdafx.h"
#include "PatchHelper.h"


void PatchHelper::visualizeIndices(const Grid3<vec4ui>& indices, const DistanceField3f& df, const std::string& filename, float occThresh)
{
	DistanceField3f curDF = df;	undoReweighting(curDF);

	MeshDataf meshData;
	for (unsigned int z = 0; z < indices.getDimZ(); z++) {
		for (unsigned int y = 0; y < indices.getDimY(); y++) {
			for (unsigned int x = 0; x < indices.getDimX(); x++) {
				const vec4ui& idx = indices(x, y, z);
				const float d = curDF(x, y, z);
				if (idx.x != (unsigned int)-1 && d <= occThresh) {
					vec3f color = vec3f(idx.getVec3()) / (float)indices.getDimX();
					vec3f p((float)x, (float)y, (float)z);
					bbox3f bbox(p - 0.5f, p + 0.5f);
					meshData.merge(Shapesf::box(bbox, vec4f(color)).computeMeshData());
				} //valid idx
			} //x
		} //y
	} //z
	MeshIOf::saveToFile(filename, meshData);
}

void PatchHelper::visualizeIndices(const DistanceField3f& df, const std::string& filename, float occThresh)
{
	DistanceField3f curDF = df;	undoReweighting(curDF);

	MeshDataf meshData;
	for (unsigned int z = 0; z < df.getDimZ(); z++) {
		for (unsigned int y = 0; y < df.getDimY(); y++) {
			for (unsigned int x = 0; x < df.getDimX(); x++) {
				const float d = curDF(x, y, z);
				if (d <= occThresh) {
					vec3f p((float)x, (float)y, (float)z);
					bbox3f bbox(p - 0.5f, p + 0.5f);
					vec3f color = p / (float)df.getDimX();
					meshData.merge(Shapesf::box(bbox, vec4f(color)).computeMeshData());
				} //valid idx
			} //x
		} //y
	} //z
	MeshIOf::saveToFile(filename, meshData);
}

