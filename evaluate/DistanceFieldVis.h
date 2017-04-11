
#pragma  once

#include "../VirtualScan/VoxelGrid.h"

class DistanceFieldVis {

public:

	template<typename T>
	static void loadDF(const std::string& filename, DistanceField3f& df);

	static void rayCastDF(const DistanceField3f& df, const std::string& filename)
	{
		BinaryGrid3 bg = df.computeBinaryGrid(1.0f);
		rayCastBinaryGrid(bg, filename);
	}
	static void rayCastSDF(const VoxelGrid& sdf, const std::string& filename)
	{
		BinaryGrid3 bg = sdf.toBinaryGridKnown(1, 1.0f);
		rayCastBinaryGrid(bg, filename);
	}
	static void rayCastBinaryGrid(const BinaryGrid3& bg, const std::string& filename, const mat4f& rotation = mat4f::rotationY(90))
	{
		TriMeshf triMesh(bg);
		if (triMesh.m_vertices.empty()) {
			ColorImageR32G32B32 black(640, 480); black.setPixels(vec3f(0.0f));
			FreeImageWrapper::saveImage(filename, black);
			return;
		}
		vec3f dims = vec3f(bg.getDimensions());
		if (bg.getDimX() != 32) {
			float scaleFactor = 32.0f / (float)bg.getDimX();
			dims *= scaleFactor;
			triMesh.scale(scaleFactor);
		}
		{//if (rotate) 
			vec3f center = triMesh.computeBoundingBox().getCenter();
			triMesh.transform(mat4f::translation(-center));
			triMesh.transform(rotation);
			triMesh.transform(mat4f::translation(center));
		}

		TriMeshAcceleratorBVHf accel(triMesh, false);

		vec3f eye = vec3f(dims.x / 2.0f, dims.y / 2.0f, dims.z / -1.0f);
		eye.x += 3.0f;
		eye.y += 7.0f;
		mat4f rot = mat4f::rotationX(15);
		mat4f camToWorld = Cameraf::viewMatrix(eye, rot*vec3f::eZ, rot*(-vec3f::eY), -vec3f::eX);

		const unsigned int width = 640;
		const unsigned int height = 480;
		const float focalLengthX = 500.0f;
		const float focalLengthY = 500.0f;
		mat4f intrinsics =
			mat4f(
			focalLengthX, 0.0f, (float)width / 2.0f, 0.0f,
			0.0f, focalLengthY, (float)height / 2.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
		mat4f intrinsicsInverse = intrinsics.getInverse();

		ColorImageR32G32B32 image(width, height);

#pragma omp parallel for
		for (int y_ = 0; y_ < (int)height; y_++) {
			unsigned int y = (unsigned int)y_;
			for (unsigned int x = 0; x < width; x++) {
				float depth0 = 0.5f;
				float depth1 = 1.0f;
				vec4f p0 = camToWorld*intrinsicsInverse*vec4f((float)x*depth0, (float)y*depth0, depth0, 1.0f);
				vec4f p1 = camToWorld*intrinsicsInverse*vec4f((float)x*depth1, (float)y*depth1, depth1, 1.0f);

				Rayf r(eye, (p1.getVec3() - p0.getVec3()).getNormalized());

				float t, u, v;
				//const ml::TriMeshf::Trianglef* tri = accel.intersect(r, t, u, v);
				const ml::TriMeshf::Triangle* tri;
				//unsigned int objIdx;
				//TriMeshRayAcceleratorf::Intersection intersect = TriMeshRayAcceleratorf::getFirstIntersection(r, accelVec, objIdx);
				TriMeshRayAcceleratorf::Intersection intersect = accel.intersect(r);
				t = intersect.t;
				u = intersect.u;
				v = intersect.v;
				tri = intersect.triangle;

				if (tri) {
					const vec3f l = vec3f(1.0f, 1.0f, 1.0f).getNormalized();
					vec3f n = (tri->getV1().position - tri->getV0().position) ^ (tri->getV2().position - tri->getV0().position);
					n.normalize();
					float diff = std::abs(l | n);
					//image(x, y) = tri->getSurfaceColor(u, v).getVec3();
					//image(x, y) = vec3f(1.0f, 0.0f, 0.0f) * diff;
					image(x, y) = (n + vec3f(1.0f)) / 2.0f;
				}
				else {
					image(x, y) = 0;
				}

			}
		}

		FreeImageWrapper::saveImage(filename, image);
	}

	static void visualizeSlice(const std::string& filename, const VoxelGrid& grid, unsigned int axis, unsigned int sliceidx, float truncation) {
		const unsigned int dim = (unsigned int)grid.getDimX(); //assumes cubic grid
		ColorImageR32G32B32 image(dim, dim);
		const float minVal = 0.0f, maxVal = 2.0f * truncation;
		for (unsigned int i = 0; i < dim; i++) {
			for (unsigned int j = 0; j < dim; j++) {
				vec3ui coord;
				if (axis == 0) coord = vec3ui(sliceidx, j, i);
				else if (axis == 1) coord = vec3ui(j, sliceidx, i);
				else coord = vec3ui(j, i, sliceidx);
				float sdf = grid(coord).sdf + truncation;
				sdf = math::clamp(sdf, minVal, maxVal);
				vec3f color = BaseImageHelper::convertDepthToRGB(sdf, minVal, maxVal);
				image(j, i) = color;
			}
		}
		FreeImageWrapper::saveImage(filename, image);
	}

private:

	static void visualizePredictedDF(const std::string& predictionFolder, const std::string& filename, unsigned int dim,
		const std::string& outVisFolder, const std::string& gtFolder, const std::string& prefix = "")
	{
		const std::string predFile = predictionFolder + filename + ".bin";
		DistanceField3f predDF;
		loadPredictionDF(predFile, dim, predDF);
		const std::string outVisFile = outVisFolder + prefix + filename;
		MeshIOf::saveToFile(outVisFile + ".ply", TriMeshf(predDF.computeBinaryGrid(1.0f)).computeMeshData());
		rayCastDF(predDF, outVisFile + ".png");
		//load gt
		const std::string gtFile = gtFolder + filename + ".df";
		DistanceField3f gtDF;
		loadDF<unsigned int>(gtFile, gtDF);
		MeshIOf::saveToFile(outVisFile + "gt.ply", TriMeshf(gtDF.computeBinaryGrid(1.0f)).computeMeshData());
		rayCastDF(gtDF, outVisFile + "gt.png");
	}
};

template<typename T>
void DistanceFieldVis::loadDF(const std::string& filename, DistanceField3f& df)
{
	BinaryDataStreamFile s(filename, BinaryDataBuffer::Mode::read_flag);
	vec3f physicalDim;
	s >> physicalDim;
	T dimx, dimy, dimz;
	s >> dimx >> dimy >> dimz;
	df.allocate(dimx, dimy, dimz);
	s.readData((BYTE*)df.getData(), sizeof(float)*df.getNumElements());
	s.close();
}

