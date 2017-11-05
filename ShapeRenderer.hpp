//============================================================================
//
// This file is part of the ShapePFCN project.
//
// Copyright (c) 2016-2017 - Evangelos Kalogerakis, Melinos Averkiou, Siddhartha Chaudhuri, Subhransu Maji
//
// ShapePFCN is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ShapePFCN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ShapePFCN.  If not, see <http://www.gnu.org/licenses/>.
//
//============================================================================

#ifndef __Thea_RenderShape_ShapeRenderer__
#define __Thea_RenderShape_ShapeRenderer__

#include "Common.hpp"
#include "Algorithms/MeshSampler.hpp"
#include "Algorithms/MeshTriangles.hpp"
#include "Algorithms/RayIntersectionTester.hpp"  // SDF/UP change
#include "Graphics/Camera.hpp"
#include "Graphics/DisplayMesh.hpp"
#include "Graphics/MeshCodec.hpp"
#include "Graphics/MeshGroup.hpp"
#include "Plugins/GL/GLHeaders.hpp"
#include "Application.hpp"
#include "Ball3.hpp"
#include "ColorRGBA.hpp"
#include "CoordinateFrame3.hpp"
#include "FilePath.hpp"
#include "FileSystem.hpp"
#include "Math.hpp"
#include "Matrix4.hpp"
#include "Plugin.hpp"
#include "UnorderedMap.hpp"
#include "Vector3.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>
#include <limits>  // SDF/UP change
#include "MeshProcessor.hpp"

using namespace std;
using namespace Thea;
using namespace Algorithms;
using namespace Graphics;


ColorRGBA8 indexToColor(uint32 index, bool is_point, bool use_same_index_for_all_channels = false);

ColorRGBA getPaletteColor(long n);

enum ColorMode
{
	COLOR_SINGLE_RGB = 0x0000,
	COLOR_SINGLE_GRAY = 0x0001,
	COLOR_BY_FACE_IDS = 0x0002,
	COLOR_BY_FACE_POINT_IDS = 0x0004,
	COLOR_BY_FACE_LABELS = 0x0008,
	COLOR_BY_FACE_LABELS_WITH_PALETTE = 0x00010,
  COLOR_BY_AUX = 0x00020, // SDF/UP change
};

struct FaceColorizer
{
	FaceIndexMap const & tri_ids;
	FaceIndexMap const & quad_ids;
	ColorMode color_mode;
	TheaArray<int> const & face_labels;
	Vector3 view_vector;

	FaceColorizer(FaceIndexMap const & tri_ids_, FaceIndexMap const & quad_ids_, TheaArray<int> const & face_labels_, ColorMode color_mode_,Vector3 view_vector_) :
		tri_ids(tri_ids_), quad_ids(quad_ids_), face_labels(face_labels_), color_mode(color_mode_), view_vector(view_vector_)
	{

	}

	bool operator()(DMesh & mesh)
	{
		mesh.isolateFaces();
		mesh.addColors();

		DMesh::IndexArray const & tris = mesh.getTriangleIndices();
		DMesh::VertexArray const & vertices = mesh.getVertices();

		for (array_size_t i = 0; i < tris.size(); i += 3)
		{
			FaceRef face(&mesh, (long)i / 3);
			FaceIndexMap::const_iterator existing = tri_ids.find(face);
			if (existing == tri_ids.end())
				throw Error(format("Could not find index of triangle %ld in mesh '%s'", (long)i / 3, mesh.getName()));
			
			uint32 id = existing->second;

			Vector3 v0 = vertices[tris[i]];
			Vector3 v1 = vertices[tris[i+1]];
			Vector3 v2 = vertices[tris[i+2]];
			Vector3 face_normal = ((v1 - v0).cross(v2 - v0)).unit();

			float normal_dot_view = face_normal.dot(-view_vector);

			ColorRGBA8 color;
			// Paint the face white if its normal dot view vector is smaller than the threshold
			if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS))
			{
        if (normal_dot_view < Settings::point_rejection_angle)
        {
          color.set(255, 255, 255, 255);
        }
        else
        {
          color = indexToColor(id, false);
        }
			}
			else if (color_mode & COLOR_BY_FACE_LABELS)
			{
				if ( (long)id < 0 || (long)id >= face_labels.size() )
				{
					throw Error(format("Face with ID %d is outside the limits of the face labels array size(%lu)", id, face_labels.size()));
				}
				color = indexToColor(face_labels[id], false, true);
			}
			else if (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE)
			{
				if ( (long)id < 0 || (long)id >= face_labels.size() )
				{
					throw Error(format("Face with ID %d is outside the limits of the face labels array size(%lu)", id, face_labels.size()));
				}
				color = getPaletteColor(face_labels[id]);
			}
			else
			{
        color.set(0, 0, 0, 0); // render black
				// throw Error(format("Invalid Color mode"));
			}
			
			mesh.setColor((long)tris[i], color);
			mesh.setColor((long)tris[i + 1], color);
			mesh.setColor((long)tris[i + 2], color);
		}

		DMesh::IndexArray const & quads = mesh.getQuadIndices();
		for (array_size_t i = 0; i < quads.size(); i += 4)
		{
			FaceRef face(&mesh, (long)i / 4);
			FaceIndexMap::const_iterator existing = quad_ids.find(face);
			if (existing == quad_ids.end())
				throw Error(format("Could not find index of quad %ld in mesh '%s'", (long)i / 4, mesh.getName()));

			uint32 id = existing->second;

			Vector3 v0 = vertices[quads[i]];
			Vector3 v1 = vertices[quads[i + 1]];
			Vector3 v2 = vertices[quads[i + 2]];
			Vector3 face_normal = ((v1 - v0).cross(v2 - v0)).unit();

			float normal_dot_view = face_normal.dot(-view_vector);

			ColorRGBA8 color;
			// Paint the face white if its normal dot view vector is smaller than the threshold
			if (normal_dot_view < Settings::point_rejection_angle)
			{
				color.set(255, 255, 255, 255);
			}
			else if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS))
			{
				color = indexToColor(id, false);
			}
			else if (color_mode & COLOR_BY_FACE_LABELS)
			{
        if ((long)id < 0 || (long)id >= face_labels.size())
				{
					throw Error(format("Face with ID %d is outside the limits of the face labels array size(%lu)", id, face_labels.size()));
				}
				color = indexToColor(face_labels[id], false, true);
			}
			else if (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE)
			{
        if ((long)id < 0 || (long)id >= face_labels.size())
				{
					throw Error(format("Face with ID %d is outside the limits of the face labels array size(%lu)", id, face_labels.size()));
				}
				color = getPaletteColor(face_labels[id]);
			}
			else
			{
        color.set(0, 0, 0, 0); // render black
				// throw Error(format("Invalid Color mode"));
			}

			mesh.setColor((long)quads[i], color);
			mesh.setColor((long)quads[i + 1], color);
			mesh.setColor((long)quads[i + 2], color);
			mesh.setColor((long)quads[i + 3], color);
		}

		return false;
	}
};


//Ball3 modelBSphere(Model const & model, Thea::Matrix4 const & transform);

class ShapeRendererImpl
{
private:
	static AtomicInt32 has_render_system;
	static RenderSystem * render_system;
	static Shader * point_shader;
	static Shader * mesh_shader;
	static Shader * face_index_shader;

	string model_path;

	Matrix4 transform;
	View view;
	float zoom;
	int out_width, out_height;
	bool has_up;
	Vector3 view_up;
	float point_size;
	ColorMode color_mode;

	double camera_distance;
	//double mesh_radius;

	ColorRGBA primary_color;
	ColorRGBA background_color;
	int antialiasing_level;
	bool flat;
	Model model;

	Image image_with_color;
	Image image_with_depth;
  Image aux_image; // SDF/UP change
  
	bool loadPlugins();
	bool parseArgs(int argc, char ** argv);
	bool usage();
	bool parseTransform(string const & s, Matrix4 & m);
	bool parseViewDiscrete(string const & s, View & view, bool silent = false);
	bool parseViewContinuous(string const & s, View & view, bool silent = false);
	bool parseViewUp(string const & s, Vector3 & up);
	bool parseColor(string const & s, ColorRGBA & c);
	void resetArgs();
	//bool loadModel(string const & path);
	bool renderModel(ColorRGBA const & color, bool draw_points);
  void renderSDFImage(Camera const & camera, Matrix4 const & transform, int width, int height, Image & image); // SDF/UP change
  void renderCoordImage(Camera const & camera, Matrix4 const & transform, int width, int height, Image & image, bool use_radial_distance, const Thea::Vector3& up_vector); // SDF/UP change

public:
	ShapeRendererImpl(const std::string & _mesh_filepath, const Model & _model);

	int exec(string const & cmdline);
	int exec(int argc, char ** argv);

	Image& get_color_image() { return image_with_color; }
	Image& get_depth_image() { return image_with_depth; }
  Image& get_aux_image() { return aux_image; } // SDF/UP change

}; // class ShapeRendererImpl

class ShapeRenderer
{
public:
	ShapeRenderer(const std::string & _mesh_filepath, const Model & _model);
	~ShapeRenderer();

	int exec(string const & cmdline);
	int exec(int argc, char ** argv);

	Image& get_color_image() { return impl->get_color_image(); }
	Image& get_depth_image() { return impl->get_depth_image(); }
  Image& get_aux_image() { return impl->get_aux_image(); } // SDF/UP change

private:
	ShapeRendererImpl * impl;

}; // class ShapeRenderer

#endif
