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

#include "ShapeRenderer.hpp"


ColorRGBA8 indexToColor(uint32 index, bool is_point, bool use_same_index_for_all_channels)
{
	ColorRGBA8 color;

	if (use_same_index_for_all_channels)
	{
		color = ColorRGBA8((uint8)((index)& 0xFF),
			(uint8)((index)& 0xFF),
			(uint8)((index)& 0xFF),
			255);
	}
	else
	{
		color = ColorRGBA8((uint8)((index)& 0xFF),
			(uint8)((index >> 8) & 0xFF),
			(uint8)((index >> 16) & 0xFF),
			255);
	}

	if (is_point)
	{
		if (color.b() & 0x80)
			THEA_WARNING << "Too many points -- point IDs will overflow and not be one-to-one!";

		color.b() = (color.b() | 0x80);
	}

	return color;
}


ColorRGBA getPaletteColor(long n)
{
	static ColorRGBA PALETTE[] = {
		ColorRGBA::fromARGB(0xFFFF0000),
		ColorRGBA::fromARGB(0xFF00FF00),
		ColorRGBA::fromARGB(0xFF0000FF),
		ColorRGBA::fromARGB(0xFF00FFFF),
		ColorRGBA::fromARGB(0xFFFF00FF),
		ColorRGBA::fromARGB(0xFFFFFF00),
		ColorRGBA::fromARGB(0xFF800000),
		ColorRGBA::fromARGB(0xFF008000),
		ColorRGBA::fromARGB(0xFF000080),
		ColorRGBA::fromARGB(0xFF008080),
		ColorRGBA::fromARGB(0xFF800080),
		ColorRGBA::fromARGB(0xFF808000),
	};

	return PALETTE[n % (sizeof(PALETTE) / sizeof(ColorRGBA))];
}


bool enableWireframe(DMesh & mesh)
{
	mesh.setWireframeEnabled(true);
	return false;
}


bool flattenFaces(DMesh & mesh)
{
	mesh.isolateFaces();
	mesh.computeAveragedVertexNormals();
	return false;
}

bool averageNormals(DMesh & mesh)
{
	//if (!mesh.hasNormals())
	mesh.computeAveragedVertexNormals();

	return false;
}

ShapeRenderer::ShapeRenderer(const std::string & _mesh_filepath, const Model & _model)
{
	try
	{
    impl = new ShapeRendererImpl(_mesh_filepath, _model);
	}
	THEA_STANDARD_CATCH_BLOCKS(throw Error("Could not create ShapeRenderer");, ERROR, "%s", "Could not create shape renderer")
}

ShapeRenderer::~ShapeRenderer()
{
	delete impl;
}

int ShapeRenderer::exec(string const & cmdline)
{
	return impl->exec(cmdline);
}

int ShapeRenderer::exec(int argc, char ** argv)
{
	return impl->exec(argc, argv);
}

AtomicInt32 ShapeRendererImpl::has_render_system(0);
RenderSystem * ShapeRendererImpl::render_system = NULL;
Shader * ShapeRendererImpl::point_shader = NULL;
Shader * ShapeRendererImpl::mesh_shader = NULL;
Shader * ShapeRendererImpl::face_index_shader = NULL;

ShapeRendererImpl::ShapeRendererImpl(const std::string & _mesh_filepath, const Model & _model)
{
	model_path = _mesh_filepath;

	resetArgs();

	if (has_render_system.compareAndSet(0, 1) == 0)
	{
		if (!loadPlugins())
			throw Error("Could not load plugins");
	}

	model = _model;
}

void ShapeRendererImpl::resetArgs()
{
	zoom = 1.0f;
	out_width = out_height = -1;
	point_size = 1.0f;
	has_up = false;
	view_up = Vector3(0, 1, 0);
	color_mode = COLOR_SINGLE_RGB;
	// Gray color for the phong shaded image
	primary_color = ColorRGBA(0.82745098f, 0.82745098f, 0.82745098f, 1.f); //0.0f, 0.0f, 0.0f, 1.0f);  //(1.0f, 0.9f, 0.8f, 1.0f); 
	background_color = ColorRGBA(1, 1, 1, 1);
	antialiasing_level = 1;
	flat = false;
	camera_distance = 0;
	transform = Matrix4::identity();

}

int ShapeRendererImpl::exec(string const & cmdline)  // cmdline should not include program name
{
	TheaArray<string> args;
	stringSplit(cmdline, " \t\n\f\r", args, true);

	TheaArray<char *> argv(args.size());

	for (array_size_t i = 0; i < args.size(); ++i)
	{
		argv[i] = new char[args[i].length() + 1];
		strcpy(argv[i], args[i].c_str());
	}

	int status = exec((int)argv.size(), &argv[0]);

	for (array_size_t i = 0; i < argv.size(); ++i)
		delete[] argv[i];

	return status;
}

//int ShapeRendererImpl::exec(int argc, char ** argv)
//{
//	if (!parseArgs(argc, argv))
//		return -1;
//
//	if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS) | (color_mode & COLOR_BY_FACE_LABELS) | (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE))
//	{
//		FaceColorizer id_colorizer(model.tri_ids, model.quad_ids, model.face_labels, color_mode,view.dir);
//		model.mesh_group.forEachMeshUntil(&id_colorizer);
//	}
//	else
//	{
//		if (flat)
//			//model.mesh_group.forEachMeshUntil(flattenFaces);
//			model.orig_mesh_group.forEachMeshUntil(flattenFaces);
//		else
//		{
//			//model.mesh_group.forEachMeshUntil(averageNormals);
//			model.orig_mesh_group.forEachMeshUntil(averageNormals);
//		}
//	}
//
//	// Set up framebuffer for offscreen drawing
//	int buffer_width = antialiasing_level * out_width;
//	int buffer_height = antialiasing_level * out_height;
//	Texture * color_tex = NULL;
//	Texture * depth_tex = NULL;
//	Framebuffer * fb = NULL;
//	try
//	{
//		Texture::Options tex_opts = Texture::Options::defaults();
//		tex_opts.interpolateMode = Texture::InterpolateMode::NEAREST_NO_MIPMAP;
//		if (color_mode & COLOR_SINGLE_GRAY)
//		{
//			color_tex = render_system->createTexture("Color", buffer_width, buffer_height, 1, Texture::Format::RGB8(),
//				Texture::Dimension::DIM_2D, tex_opts);
//		}
//		else
//		{
//			color_tex = render_system->createTexture("Color", buffer_width, buffer_height, 1, Texture::Format::RGBA8(),
//				Texture::Dimension::DIM_2D, tex_opts);
//		}
//		if (!color_tex)
//		{
//			THEA_ERROR << "Could not create color buffer";
//			return -1;
//		}
//
//		depth_tex = render_system->createTexture("Depth", buffer_width, buffer_height, 1, Texture::Format::DEPTH16(),
//			Texture::Dimension::DIM_2D, tex_opts);
//
//		if (!depth_tex)
//		{
//			THEA_ERROR << "Could not create depth buffer";
//			return -1;
//		}
//
//		fb = render_system->createFramebuffer("Framebuffer");
//		if (!fb)
//		{
//			THEA_ERROR << "Could not create offscreen framebuffer";
//			return -1;
//		}
//
//		fb->attach(Framebuffer::AttachmentPoint::COLOR_0, color_tex);
//		fb->attach(Framebuffer::AttachmentPoint::DEPTH, depth_tex);
//	}
//	THEA_STANDARD_CATCH_BLOCKS(return -1;, ERROR, "%s", "Could not render shape")
//
//	// Do the rendering
//	// We used to do a loop over multiple views, no longer needed, we just have one view
//	try
//	{
//		// Initialize the camera
//		Camera camera = model.fitCamera(transform, view, zoom, buffer_width, buffer_height, camera_distance, model.mesh_radius);
//
//		// Render the mesh to the offscreen framebuffer
//		render_system->pushFramebuffer();
//		render_system->setFramebuffer(fb);
//
//		render_system->pushDepthFlags();
//		render_system->pushColorFlags();
//		render_system->pushShapeFlags();
//		render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
//		render_system->setMatrixMode(RenderSystem::MatrixMode::PROJECTION); render_system->pushMatrix();
//
//		render_system->setCamera(camera);
//		render_system->setDepthTest(RenderSystem::DepthTest::LESS);
//		render_system->setDepthWrite(true);
//		render_system->setColorWrite(true, true, true, true);
//
//		render_system->setColorClearValue(background_color);
//		render_system->clear();
//
//		// Draw model
//		render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
//		render_system->multMatrix(transform);
//		renderModel(primary_color, false);
//		render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();
//
//		if (color_mode & COLOR_BY_FACE_POINT_IDS)
//		{
//			// just get a dummy color. the only case we render points is when we render them using their index as color, so the overlay_color will not be used
//			ColorRGBA overlay_color = getPaletteColor(0);
//
//			render_system->setPolygonOffset(true, -1.0f);  // make sure overlays appear on top of primary shape
//
//			render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
//			render_system->multMatrix(transform);
//			renderModel(overlay_color, true);
//			render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();
//		}
//
//		render_system->setMatrixMode(RenderSystem::MatrixMode::PROJECTION); render_system->popMatrix();
//		render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();
//		render_system->popShapeFlags();
//		render_system->popColorFlags();
//		render_system->popDepthFlags();
//
//		// Grab the rendered color image and depth image		
//		image_with_color.clear();
//		if (color_mode & COLOR_SINGLE_GRAY)
//		{
//			image_with_color = Image(Image::Type::RGB_8U, buffer_width, buffer_height);
//		}
//		else
//		{
//			image_with_color = Image(Image::Type::RGB_8U, buffer_width, buffer_height);
//		}
//		color_tex->getImage(image_with_color);
//
//		if (antialiasing_level > 1 && !image_with_color.rescale(out_width, out_height, Image::Filter::BICUBIC))
//		{
//			THEA_ERROR << "Could not rescale color image to output dimensions";
//			return -1;
//		}
//
//		image_with_depth.clear();
//		image_with_depth = Image(Image::Type::LUMINANCE_16U, buffer_width, buffer_height);
//		depth_tex->getImage(image_with_depth);
//
//		if (antialiasing_level > 1 && !image_with_depth.rescale(out_width, out_height, Image::Filter::BICUBIC))
//		{
//			THEA_ERROR << "Could not rescale depth image to output dimensions";
//			return -1;
//		}
//
//		render_system->popFramebuffer();
//
//    delete color_tex;
//		delete depth_tex;
//		delete fb;
//	}
//  catch (std::exception & e__)
//  {
//    THEA_ERROR << "Could not render view 1 of shape";
//    return -1;
//  }
//	
//  return 0;
//}


int ShapeRendererImpl::exec(int argc, char ** argv) // SDF/UP change
{
  if (!parseArgs(argc, argv))
    return -1;

  if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS) | (color_mode & COLOR_BY_FACE_LABELS) | (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE))
  {
    FaceColorizer id_colorizer(model.tri_ids, model.quad_ids, model.face_labels, color_mode, view.dir);
    model.mesh_group.forEachMeshUntil(&id_colorizer);
  }
  else
  {
    if (flat)
      //model.mesh_group.forEachMeshUntil(flattenFaces);
      model.orig_mesh_group.forEachMeshUntil(flattenFaces);
    else
    {
      //model.mesh_group.forEachMeshUntil(averageNormals);
      model.orig_mesh_group.forEachMeshUntil(averageNormals);
    }
  }

  try
  {
    if (!(color_mode & COLOR_BY_AUX))
    {
      // Set up framebuffer for offscreen drawing
      int buffer_width = antialiasing_level * out_width;
      int buffer_height = antialiasing_level * out_height;

      // Initialize the camera
      Camera camera = model.fitCamera(transform, view, zoom, buffer_width, buffer_height, camera_distance, model.mesh_radius);

      Texture * color_tex = NULL;
      Texture * depth_tex = NULL;
      Framebuffer * fb = NULL;

      Texture::Options tex_opts = Texture::Options::defaults();
      tex_opts.interpolateMode = Texture::InterpolateMode::NEAREST_NO_MIPMAP;
      if (color_mode & COLOR_SINGLE_GRAY)
      {
        color_tex = render_system->createTexture("Color", buffer_width, buffer_height, 1, Texture::Format::RGB8(),
          Texture::Dimension::DIM_2D, tex_opts);
      }
      else
      {
        color_tex = render_system->createTexture("Color", buffer_width, buffer_height, 1, Texture::Format::RGBA8(),
          Texture::Dimension::DIM_2D, tex_opts);
      }
      if (!color_tex)
      {
        THEA_ERROR << "Could not create color buffer";
        return -1;
      }

      depth_tex = render_system->createTexture("Depth", buffer_width, buffer_height, 1, Texture::Format::DEPTH16(),
        Texture::Dimension::DIM_2D, tex_opts);

      if (!depth_tex)
      {
        THEA_ERROR << "Could not create depth buffer";
        return -1;
      }

      fb = render_system->createFramebuffer("Framebuffer");
      if (!fb)
      {
        THEA_ERROR << "Could not create offscreen framebuffer";
        return -1;
      }

      fb->attach(Framebuffer::AttachmentPoint::COLOR_0, color_tex);
      fb->attach(Framebuffer::AttachmentPoint::DEPTH, depth_tex);

      // Do the rendering
      // We used to do a loop over multiple views, no longer needed, we just have one view

      // Render the mesh to the offscreen framebuffer
      render_system->pushFramebuffer();
      render_system->setFramebuffer(fb);

      render_system->pushDepthFlags();
      render_system->pushColorFlags();
      render_system->pushShapeFlags();
      render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
      render_system->setMatrixMode(RenderSystem::MatrixMode::PROJECTION); render_system->pushMatrix();

      render_system->setCamera(camera);
      render_system->setDepthTest(RenderSystem::DepthTest::LESS);
      render_system->setDepthWrite(true);
      render_system->setColorWrite(true, true, true, true);

      render_system->setColorClearValue(background_color);
      render_system->clear();

      // Draw model
      render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
      render_system->multMatrix(transform);
      renderModel(primary_color, false);
      render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();

      if (color_mode & COLOR_BY_FACE_POINT_IDS)
      {
        // just get a dummy color. the only case we render points is when we render them using their index as color, so the overlay_color will not be used
        ColorRGBA overlay_color = getPaletteColor(0);
        render_system->setPolygonOffset(true, -1.0f);  // make sure overlays appear on top of primary shape

        render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->pushMatrix();
        render_system->multMatrix(transform);
        renderModel(overlay_color, true);
        render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();
      }

      render_system->setMatrixMode(RenderSystem::MatrixMode::PROJECTION); render_system->popMatrix();
      render_system->setMatrixMode(RenderSystem::MatrixMode::MODELVIEW); render_system->popMatrix();
      render_system->popShapeFlags();
      render_system->popColorFlags();
      render_system->popDepthFlags();

      // Grab the rendered color image and depth image
      image_with_color.clear();
      if (color_mode & COLOR_SINGLE_GRAY)
      {
        image_with_color = Image(Image::Type::RGB_8U, buffer_width, buffer_height);
      }
      else
      {
        image_with_color = Image(Image::Type::RGB_8U, buffer_width, buffer_height);
      }
      color_tex->getImage(image_with_color);

      if (antialiasing_level > 1 && !image_with_color.rescale(out_width, out_height, Image::Filter::BICUBIC))
      {
        THEA_ERROR << "Could not rescale color image to output dimensions";
        return -1;
      }

      image_with_depth.clear();
      image_with_depth = Image(Image::Type::LUMINANCE_16U, buffer_width, buffer_height);
      depth_tex->getImage(image_with_depth);

      if (antialiasing_level > 1 && !image_with_depth.rescale(out_width, out_height, Image::Filter::BICUBIC))
      {
        THEA_ERROR << "Could not rescale depth image to output dimensions";
        return -1;
      }

      render_system->popFramebuffer();

      delete color_tex;
      delete depth_tex;
      delete fb;
    }
    else // SDF/UP change
    {
      // we were considering SDF, but did not work well for ShapeNet, this is now replaced with x^2 + y^2 + z^2

      //////// Set up framebuffer for offscreen drawing
      //////int buffer_width = out_width / 4; // SDF computation is expensive / do it in lower res
      //////int buffer_height = out_height / 4;

      //////// Initialize the camera
      //////Camera camera = model.fitCamera(transform, view, zoom, buffer_width, buffer_height, camera_distance, model.mesh_radius);

      //////// @Sid add the sdf image computation here
      //////// Save the image in image_sdf; and it will get accessed using get_sdf_image(); (check ShapeRenderer.hpp)
      //////renderSDFImage(camera, transform, buffer_width, buffer_height, image_with_sdf);

      //////if (!image_with_sdf.rescale(out_width, out_height, Image::Filter::BICUBIC))
      //////{
      //////  THEA_ERROR << "Could not rescale SDF image to output dimensions";
      //////  return -1;
      //////}

      // Set up framebuffer for offscreen drawing
      int buffer_width = out_width;
      int buffer_height = out_height;

      // Initialize the camera
      Camera camera = model.fitCamera(transform, view, zoom, buffer_width, buffer_height, camera_distance, model.mesh_radius);

      renderCoordImage(camera, transform, buffer_width, buffer_height, aux_image, !Settings::use_upright_coord, Settings::up_vector);

      if (!aux_image.rescale(out_width, out_height, Image::Filter::BICUBIC))
      {
        THEA_ERROR << "Could not rescale x^2+y^2+z^2 image to output dimensions";
        return -1;
      }
    }
  }
  THEA_STANDARD_CATCH_BLOCKS(return -1; , ERROR, "%s", "Could not render view of shape")

    return 0;
}

// SDF change
void ShapeRendererImpl::renderSDFImage(Camera const & camera, Matrix4 const & transform, int width, int height, Image & image)
{
  if (!model.kdtree)
  {
    THEA_ERROR << "KD-tree for mesh was not created!";
    return;
  }  

  Algorithms::MeshFeatures::Local::ShapeDiameter<DMesh> sdf(model.kdtree, model.mesh_radius); // 2 * model.mesh_radius is max theoretically, but this would only hold for a perfect ball - better scale less aggressively

  image.clear();
  image.resize(Image::Type::LUMINANCE_16U, width, height);
  Matrix4 inv_transform = transform.inverse();

  static uint16 const PMAX = std::numeric_limits<uint16>::max();

#pragma omp parallel for 
  for (int r = 0; r < height; ++r)
  {
    //THEA_CONSOLE << "Rendering SDF row " << r;

    uint16 * pixel = (uint16 *)image.getScanLine(r);

    for (int c = 0; c < width; ++c)
    {
      Vector2 loc(2 * (c + 0.5f) / width - 1, 2 * (r + 0.5f) / height - 1);
      Ray3 ray = camera.computePickRay(loc);
      Vector4 p = inv_transform * Vector4(ray.getOrigin(), 1.0);
      Vector4 u = inv_transform * Vector4(ray.getDirection(), 0.0);
      ray = Ray3(p.xyz() / p.w(), u.xyz().unit());

      KDTree::RayStructureIntersectionT isec = model.kdtree->rayStructureIntersection<Algorithms::RayIntersectionTester>(ray);
      if (isec.isValid())
      {
        alwaysAssertM(isec.hasNormal(), "SDF: Normal at intersection point not found");

        Vector3 surface_point = ray.getPoint(isec.getTime());
        Vector3 surface_normal = isec.getNormal();

        double v0 = sdf.compute(surface_point, surface_normal, true);
        if (v0 < 0)
          v0 = sdf.compute(surface_point, surface_normal, false);

        double v1 = sdf.compute(surface_point, -surface_normal, true);
        if (v1 < 0)
          v1 = sdf.compute(surface_point, -surface_normal, false);

        double vmin = (v1 < 0 || (v0 >= 0 && v0 < v1)) ? v0 : v1;
        double f = Math::clamp( vmin, 0, 1 );

        *pixel = Math::clamp((uint16)std::floor(f * PMAX + 0.5), (uint16)0, PMAX);
      }
      else
      {
        *pixel = PMAX;  // 0xFFFF
      }
      ++pixel;
    }
  }
}


// UP change
void ShapeRendererImpl::renderCoordImage(Camera const & camera, Matrix4 const & transform, int width, int height, Image & image, bool use_radial_distance, const Thea::Vector3& up_vector)
{
  if (!model.kdtree)
  {
    THEA_ERROR << "KD-tree for mesh was not created!";
    return;
  }

  image.clear();
  image.resize(Image::Type::LUMINANCE_16U, width, height);
  Matrix4 inv_transform = transform.inverse();

  static uint16 const PMAX = std::numeric_limits<uint16>::max();

#pragma omp parallel for 
  for (int r = 0; r < height; ++r)
  {
    uint16 * pixel = (uint16 *)image.getScanLine(r);

    for (int c = 0; c < width; ++c)
    {
      Vector2 loc(2 * (c + 0.5f) / width - 1, 2 * (r + 0.5f) / height - 1);
      Ray3 ray = camera.computePickRay(loc);
      Vector4 p = inv_transform * Vector4(ray.getOrigin(), 1.0);
      Vector4 u = inv_transform * Vector4(ray.getDirection(), 0.0);
      ray = Ray3(p.xyz() / p.w(), u.xyz().unit());

      KDTree::RayStructureIntersectionT isec = model.kdtree->rayStructureIntersection<Algorithms::RayIntersectionTester>(ray);
      if (isec.isValid())
      {
        Vector3 surface_point = ray.getPoint(isec.getTime());
        double coord = 0.0;
        if (use_radial_distance)
        {
          coord = sqrt(surface_point.x()*surface_point.x() + surface_point.y()*surface_point.y() + surface_point.z()*surface_point.z()) / (model.mesh_radius + 1e-30);          
        }
        else
        {
          if (abs(up_vector.x()) > .97 )
            coord = surface_point.x() / model.axis_length[0] + 0.5;
          else if (abs(up_vector.y()) > .97 )
            coord = surface_point.y() / model.axis_length[1] + 0.5;
          else if (abs(up_vector.z()) > .97 )
            coord = surface_point.z() / model.axis_length[2] + 0.5;
          else
            coord = sqrt(surface_point.x()*surface_point.x() + surface_point.y()*surface_point.y() + surface_point.z()*surface_point.z()) / (model.mesh_radius + 1e-30);
        }
        coord = Math::clamp(coord, 0.0, 1.0);
        *pixel = Math::clamp((uint16)std::floor(coord * PMAX + 0.5), (uint16)0, PMAX);
      }
      else
      {
        *pixel = PMAX;  // 0xFFFF
      }
      ++pixel;
    }
  }
}

bool ShapeRendererImpl::usage()
{
	//string app_path = FilePath::objectName(Application::programPath());

	THEA_CONSOLE << "";
	THEA_CONSOLE << "Usage: " << " [OPTIONS] <width> <height>";
	THEA_CONSOLE << "";
	THEA_CONSOLE << "Options:";
	THEA_CONSOLE << "  -t <transform>        (row-major comma-separated 3x4 or 4x4 matrix,";
	THEA_CONSOLE << "                         applied to shape)";
	THEA_CONSOLE << "  -z <factor>           (zoom factor, default 1)";
	THEA_CONSOLE << "  -v <arg>              (comma-separated 3-vector (viewing direction);";
	THEA_CONSOLE << "                         or 6-vector (direction + eye position);";
	THEA_CONSOLE << "                         or 9-vector (direction + eye + up);";
	THEA_CONSOLE << "                         or string of 3 chars, one for each coordinate,";
	THEA_CONSOLE << "                           each one of +, - or 0;)";
	THEA_CONSOLE << "  -u <up-dir>           (x, y or z, optionally preceded by + or -)";
	THEA_CONSOLE << "  -c <argb>             (shape color, or 'id' to color faces by face ID and";
	THEA_CONSOLE << "                         points by point ID)";
	THEA_CONSOLE << "  -b <argb>             (background color)";
	THEA_CONSOLE << "  -a N                  (enable NxN antialiasing: 2 is normal, 4 is very";
	THEA_CONSOLE << "                         high quality)";
	THEA_CONSOLE << "  -f                    (flat shading)";
	THEA_CONSOLE << "";

	return false;
}

bool ShapeRendererImpl::parseTransform(string const & s, Matrix4 & m)
{
	TheaArray<string> fields;
	long nfields = stringSplit(trimWhitespace(s), ",;:[({<>})] \t\n\r\f", fields, true);
	if (nfields != 12 && nfields != 16)
	{
		THEA_ERROR << "Could not read row-major comma-separated matrix from '" << s << '\'';
		return false;
	}

	m = Matrix4::identity();
	for (int i = 0; i < nfields; ++i)
	{
		istringstream field_in(fields[i]);
		if (!(field_in >> m(i / 4, i % 4)))
		{
			THEA_ERROR << "Could not parse matrix entry '" << fields[i] << '\'';
			return false;
		}
	}

	return true;
}

bool ShapeRendererImpl::parseViewDiscrete(string const & s, View & view, bool silent)
{
	if (s.length() != 3)
	{
		if (!silent) THEA_ERROR << "Viewing direction string must have exactly 3 characters, one for each coordinate";
		return false;
	}

	if (s == "000")
	{
		if (!silent) THEA_ERROR << "View direction is zero vector";
		return false;
	}

	view = View();

	for (int i = 0; i < 3; ++i)
	{
		switch (s[i])
		{
		case '+': view.dir[i] = 1; break;
		case '-': view.dir[i] = -1; break;
		case '0': view.dir[i] = 0; break;
		default:
			if (!silent) THEA_ERROR << "Invalid view direction string '" << s << '\'';
			return false;
		}
	}

	if (view.dir.squaredLength() <= 1e-10)
	{
		if (!silent) THEA_ERROR << "View direction is zero vector";
		return false;
	}

	view.dir.unitize();

	if (has_up)
		view.up = view_up;
	else if (s == "0-0")
		view.up = -Vector3::unitZ();
	else if (s == "0+0")
		view.up = Vector3::unitZ();
	else
		view.up = Vector3::unitY();

	return true;
}

bool ShapeRendererImpl::parseViewContinuous(string const & s, View & view, bool silent)
{
	double dx, dy, dz;
	double ex, ey, ez;
	double ux, uy, uz;
	double cam_dist;
	char trailing[2];  // to make sure there's nothing after the 9th number
	int num_params = sscanf(s.c_str(), " %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf %1s",
		&dx, &dy, &dz, &cam_dist, &ex, &ey, &ez, &ux, &uy, &uz, trailing);

	if (!(num_params == 10 || num_params == 7 || num_params == 4))
	{
		if (!silent) THEA_ERROR << "Invalid view string '" << s << '\'';
		return false;
	}

	view = View();

	view.dir = Vector3(dx, dy, dz);
	if (view.dir.squaredLength() <= 1e-10)
	{
		if (!silent) THEA_ERROR << "View direction is zero vector";
		return false;
	}
	view.dir.unitize();

	camera_distance = cam_dist;

	if (num_params == 7)
	{
		view.has_eye = true;
		view.eye = Vector3(ex, ey, ez);
	}

	if (num_params == 10)
	{
		view.has_eye = true;
		view.eye = Vector3(ex, ey, ez);

		view.up = Vector3(ux, uy, uz);

		if (view.up.squaredLength() <= 1e-10)
		{
			if (!silent) THEA_ERROR << "View up is zero vector";
			return false;
		}

		view.up.unitize();
	}
	else if (has_up)
		view.up = view_up;
	else
	{
		Real d = view.dir.dot(Vector3::unitY());
		if (Math::fuzzyEq(d, (Real)-1))
			view.up = -Vector3::unitZ();
		else if (Math::fuzzyEq(d, (Real)1))
			view.up = Vector3::unitZ();
		else
			view.up = Vector3::unitY();
	}

	return true;
}


bool ShapeRendererImpl::parseViewUp(string const & s, Vector3 & up)
{
	if (s.length() != 1 && s.length() != 2)
	{
		THEA_ERROR << "Up direction must be 'x', 'y' or 'z', optionally preceded by + or -";
		return false;
	}

	char c = s[0];
	bool neg = false;
	if (c == '+' || c == '-')
	{
		if (s.length() < 2)
		{
			THEA_ERROR << "Up direction must be 'x', 'y' or 'z', optionally preceded by + or -";
			return false;
		}

		if (c == '-')
			neg = true;

		c = s[1];
	}

	if (c == 'x' || c == 'X')
		up = Vector3::unitX();
	else if (c == 'y' || c == 'Y')
		up = Vector3::unitY();
	else if (c == 'z' || c == 'Z')
		up = Vector3::unitZ();
	else
	{
		THEA_ERROR << "Up direction must be 'x', 'y' or 'z', optionally preceded by + or -";
		return false;
	}

	if (neg)
		up = -up;

	return true;
}

bool ShapeRendererImpl::parseColor(string const & s, ColorRGBA & c)
{
	std::stringstream ss;
	ss << std::hex << s;

	uint32 argb;
	if (!(ss >> argb))
	{
		THEA_ERROR << "Could not parse color '" << s << '\'';
		return false;
	}

	c = ColorRGBA::fromARGB(argb);

	if (trimWhitespace(s).length() <= 6)  // alpha channel not specified
		c.a() = 1.0;

	return true;
}

bool ShapeRendererImpl::parseArgs(int argc, char ** argv)
{
	if (argc < 2)
		return usage();

	resetArgs();

	transform = Matrix4::identity();

	argv++;
	argc--;
	int pos = 0;

	while (argc > 0)
	{
		string arg = *argv;
		argv++; argc--;

		if (arg.length() <= 0)
			continue;

		if (arg[0] == '-')
		{
			if (arg.length() != 2)
				return usage();

			switch (arg[1])
			{

			case 't':
			{
				if (argc < 1) { THEA_ERROR << "-t: Transform not specified"; return false; }
				if (!parseTransform(*argv, transform)) return false;
				argv++; argc--; break;
			}

			case 'z':
			{
				if (argc < 1) { THEA_ERROR << "-z: Zoom not specified"; return false; }
				if (sscanf(*argv, " %f", &zoom) != 1)
				{
					THEA_ERROR << "Could not parse zoom '" << *argv << '\'';
					return false;
				}
				if (zoom <= 0)
				{
					THEA_ERROR << "Invalid zoom " << zoom;
					return false;
				}
				argv++; argc--; break;
			}

			case 'v':
			{
				if (argc < 1) { THEA_ERROR << "-v: View parameters not specified"; return false; }
				bool status = false;
				if (strlen(*argv) == 3)
					status = parseViewDiscrete(*argv, view, true);
				else
					status = parseViewContinuous(*argv, view, true);

				if (!status)
				{
					THEA_ERROR << "Could not parse view direction '" << *argv << '\'';
					return false;
				}

				argv++; argc--; break;
			}

			case 'u':
			{
				if (argc < 1) { THEA_ERROR << "-u: Up direction not specified"; return false; }
				if (!parseViewUp(*argv, view_up)) return false;
				has_up = true;
				argv++; argc--; break;
			}

			case 'c':
			{
				if (argc < 1) { THEA_ERROR << "-c: Mesh color not specified"; return false; }

				if (toLower(trimWhitespace(*argv)) == "id")
					color_mode = COLOR_BY_FACE_POINT_IDS;
				else if (toLower(trimWhitespace(*argv)) == "fid")
					color_mode = COLOR_BY_FACE_IDS;
				else if (toLower(trimWhitespace(*argv)) == "lbl")
					color_mode = COLOR_BY_FACE_LABELS;
				else if (toLower(trimWhitespace(*argv)) == "lblp")
					color_mode = COLOR_BY_FACE_LABELS_WITH_PALETTE;
        else if (toLower(trimWhitespace(*argv)) == "aux") // SDF/UP change
          color_mode = COLOR_BY_AUX;
				else if (toLower(trimWhitespace(*argv)) == "gray")
					color_mode = COLOR_SINGLE_GRAY;
				else {
					color_mode = COLOR_SINGLE_RGB;
					if (!parseColor(*argv, primary_color))
						return false;
				}

				argv++; argc--; break;
			}

			case 'b':
			{
				if (argc < 1) { THEA_ERROR << "-b: Background color not specified"; return false; }
				if (!parseColor(*argv, background_color)) return false;
				argv++; argc--; break;
			}

			case 'a':
			{
				if (argc < 1) { THEA_ERROR << "-a: Antialiasing level not specified"; return false; }
				if (sscanf(*argv, " %d", &antialiasing_level) != 1)
				{
					THEA_ERROR << "Could not parse antialiasing level '" << *argv << '\'';
					return false;
				}
				if (antialiasing_level < 1)
				{
					THEA_ERROR << "Invalid antialiasing level " << antialiasing_level;
					return false;
				}
				argv++; argc--; break;
			}

			case 'f':
			{
				flat = true;
				break;
			}
			}
		}
		else
		{
			pos++;

			switch (pos)
			{
			case 1:
			{
				if (argc < 1) { THEA_ERROR << "Width not followed by height"; return false; }

				if (sscanf(arg.c_str(), "%d", &out_width) != 1 || sscanf(*argv, "%d", &out_height) != 1
					|| out_width <= 0 || out_height <= 0)
				{
					THEA_ERROR << "Could not parse output image dimensions: " << arg << " x " << *argv;
					return false;
				}

				argv++; argc--; pos++;
				break;
			}

			default:
			{
				THEA_ERROR << "Too many positional arguments";
				return false;
			}
			}
		}
	}

	if (pos < 2)
	{
		THEA_ERROR << "Too few positional arguments";
		return usage();
	}

	return true;
}

//bool ShapeRendererImpl::loadModel(string const & path)
//{
//  model.mesh_group.clear();
//  //model.face_labels.clear();
//  model.face_labels = ground_truth_labels;
//  model.sample_points.clear();
//  model.tri_ids.clear();
//  model.quad_ids.clear();
//
//  if (endsWith(toLower(path), ".pts"))
//  {
//    THEA_ERROR << "Unsupported file type: " << path;
//    return false;
//  }
//  else
//  {
//    try
//    {
//      MeshReadCallback callback(model.tri_ids, model.quad_ids);
//
//      model.mesh_group.load(path, Codec_AUTO(), &callback);
//    }
//    THEA_STANDARD_CATCH_BLOCKS(return false;, ERROR, "Could not load model from '%s'", path.c_str());
//
//    Ball3 bsphere = modelBSphere(model, transform);
//    mesh_radius = bsphere.getRadius();
//
//    // Write a seg file since we are here
//    std::string seg_file = FilePath::concat(FilePath::parent(path), FilePath::baseName(path) + ".seg");
//    ofstream seg_out(seg_file.c_str());
//
//    if (seg_out)
//    {
//      for (std::vector<int>::iterator fl_it = ground_truth_labels.begin(); fl_it != ground_truth_labels.end(); ++fl_it)
//      {
//        seg_out << *fl_it << std::endl;
//      }
//    }
//  }
//  //}
//
//  return true;
//}

class FarthestPoint
{
public:
	FarthestPoint(Vector3 const & center_, Matrix4 const & transform_)
		: center(center_), transform(transform_), max_sqdist(0) {}

	bool operator()(DMesh const & mesh)
	{
		DMesh::VertexArray const & vertices = mesh.getVertices();
		for (array_size_t i = 0; i < vertices.size(); ++i)
		{
			Real sqdist = (transform * vertices[i] - center).squaredLength();
			if (sqdist > max_sqdist)
				max_sqdist = sqdist;
		}

		return false;
	}

	Real getFarthestDistance() const { return sqrt(max_sqdist); }

private:
	Vector3 center;
	Matrix4 const & transform;
	Real max_sqdist;
};

//Ball3 modelBSphere(Model const & model, Matrix4 const & transform)
//{
//	double sum_x = 0, sum_y = 0, sum_z = 0;
//	double sum_w = 0;
//
//	MeshTriangles<DMesh> tris;
//	tris.add(const_cast<MG &>(model.mesh_group));
//	
//	//Thea::Algorithms::BestFitSphere3 bsphere;
//
//	MeshTriangles<DMesh>::TriangleArray const & tri_array = tris.getTriangles();
//	for (array_size_t i = 0; i < tri_array.size(); ++i)
//	{
//		Vector3 c = tri_array[i].getCentroid();
//		Real area = tri_array[i].getArea();
//		
//		//bsphere.addPoint(tri_array[i].getVertex(0));
//		//bsphere.addPoint(tri_array[i].getVertex(1));
//		//bsphere.addPoint(tri_array[i].getVertex(2));
//
//		sum_x += (area * c[0]);
//		sum_y += (area * c[1]);
//		sum_z += (area * c[2]);
//
//		sum_w += area;
//	}
//
//	Vector3 center(0, 0, 0);
//	if (sum_w > 0)
//	{
//		center[0] = (Real)(sum_x / sum_w);
//		center[1] = (Real)(sum_y / sum_w);
//		center[2] = (Real)(sum_z / sum_w);
//	}
//
//	center = transform * center;
//
//	Real radius = 0;
//
//	FarthestPoint fp(center, transform);
//	model.mesh_group.forEachMeshUntil(&fp);
//	radius = fp.getFarthestDistance();
//
//	return Ball3(center, radius);
//	//return Ball3(bsphere.getCenter(), bsphere.getRadius());
//}

bool initPointShader(Shader & shader)
{
	static string const VERTEX_SHADER =
		"void main()\n"
		"{\n"
		"  gl_Position = ftransform();\n"
		"  gl_FrontColor = gl_Color;\n"
		"  gl_BackColor = gl_Color;\n"
		"}\n";

	static string const FRAGMENT_SHADER =
		"void main()\n"
		"{\n"
		"  gl_FragColor = gl_Color;\n"
		"}\n";

	try
	{
		shader.attachModuleFromString(Shader::ModuleType::VERTEX, VERTEX_SHADER.c_str());
		shader.attachModuleFromString(Shader::ModuleType::FRAGMENT, FRAGMENT_SHADER.c_str());
	}
	THEA_STANDARD_CATCH_BLOCKS(return false; , ERROR, "%s", "Could not attach point shader module")

		return true;
}

bool initMeshShader(Shader & shader)
{
	static string const VERTEX_SHADER =
		"varying vec3 position;  // position in camera space\n"
		"varying vec3 normal;  // normal in camera space\n"
		"\n"
		"void main()\n"
		"{\n"
		"  gl_Position = ftransform();\n"
		"\n"
		"  position = vec3(gl_ModelViewMatrix * gl_Vertex);  // assume rigid transform, so we can drop w\n"
		"  normal = gl_NormalMatrix * gl_Normal;\n"
		"\n"
		"  gl_FrontColor = gl_Color;\n"
		"  gl_BackColor = gl_Color;\n"
		"}\n";

	static string const FRAGMENT_SHADER =
		"uniform vec3 ambient_color;\n"
		//"uniform vec3 light_dir;  // must be specified in camera space, pointing towards object\n"
		"uniform vec3 light_color;\n"
		"uniform vec4 material;  // [ka, kl, <ignored>, <ignored>]\n"
		"uniform float two_sided;\n"
		"\n"
		"varying vec3 position;  // position in camera space\n"
		"varying vec3 normal;  // normal in camera space\n"
		"\n"
		"void main()\n"
		"{\n"
		"  vec3 N = normalize(normal);\n"
		"  vec3 L = normalize(position); //light_dir);\n"
		"\n"
		"  vec3 ambt_color = material[0] * gl_Color.rgb * ambient_color;\n"
		"\n"
		"  float NdL = -dot(N, L);\n"
		"  vec3 lamb_color = (NdL >= -two_sided) ? material[1] * abs(NdL) * gl_Color.rgb * light_color : vec3(0.0, 0.0, 0.0);\n"
		"\n"
		"  gl_FragColor = vec4(ambt_color + lamb_color, gl_Color.a);\n"
		"}\n";

	try
	{
		shader.attachModuleFromString(Shader::ModuleType::VERTEX, VERTEX_SHADER.c_str());
		shader.attachModuleFromString(Shader::ModuleType::FRAGMENT, FRAGMENT_SHADER.c_str());
	}
	THEA_STANDARD_CATCH_BLOCKS(return false; , ERROR, "%s", "Could not attach mesh shader module")

		//shader.setUniform("light_dir", Vector3(-1, -1, -2));
		shader.setUniform("light_color", ColorRGB(1, 1, 1));
	shader.setUniform("ambient_color", ColorRGB(1, 1, 1));
	shader.setUniform("two_sided", 1.0f);
	shader.setUniform("material", Vector4(0.2f, 0.8f, 0.0f, 0)); //(0.2f, 0.6f, 0.2f, 25));

	return true;
}

bool initFaceIndexShader(Shader & shader)
{
	static string const VERTEX_SHADER =
		"\n"
		"void main()\n"
		"{\n"
		"  gl_Position = ftransform();\n"
		"  gl_FrontColor = gl_Color;\n"
		"  gl_BackColor = gl_Color;\n"
		"}\n";

	static string const FRAGMENT_SHADER =
		"void main()\n"
		"{\n"
		"  gl_FragColor = gl_Color;\n"
		"}\n";

	try
	{
		shader.attachModuleFromString(Shader::ModuleType::VERTEX, VERTEX_SHADER.c_str());
		shader.attachModuleFromString(Shader::ModuleType::FRAGMENT, FRAGMENT_SHADER.c_str());
	}
	THEA_STANDARD_CATCH_BLOCKS(return false; , ERROR, "%s", "Could not attach face index shader module")

		return true;
}

bool ShapeRendererImpl::renderModel(ColorRGBA const & color, bool draw_points)
{
	render_system->pushShader();
	render_system->pushShapeFlags();
	render_system->pushColorFlags();

	render_system->setColor(color);

	bool has_transparency = (color.a() < 1);
	if (has_transparency)
	{
		// Enable alpha-blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	if (draw_points)
	{
		// Initialize the shader
		if (!point_shader)
		{
			point_shader = render_system->createShader("Point shader");
			if (!point_shader)
			{
				THEA_ERROR << "Could not create point shader";
				return false;
			}

			if (!initPointShader(*point_shader))
			{
				THEA_ERROR << "Could not initialize point shader";
				return false;
			}
		}

		render_system->setShader(point_shader);

		render_system->setPointSize(point_size * antialiasing_level);
		render_system->beginPrimitive(RenderSystem::Primitive::POINTS);

		for (array_size_t i = 0; i < model.sample_points.size(); ++i)
		{
			if (color_mode & COLOR_BY_FACE_POINT_IDS)
				render_system->setColor(indexToColor((uint32)i, true));

      if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS))
      {
        float normal_dot_view = model.sample_normals[i].dot(-view.dir);
        // Do not count the point as visible if its normal dot view vector is smaller than the threshold
        if (normal_dot_view < Settings::point_rejection_angle)
        {
          continue;
        }
      }
			  render_system->sendVertex(model.sample_points[i]);
		}

		render_system->endPrimitive();
	}
	else
	{
		// Initialize the shader
		if ((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS) | (color_mode & COLOR_BY_FACE_LABELS) | (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE))
		{
			if (!face_index_shader)
			{
				face_index_shader = render_system->createShader("Face index shader");
				if (!face_index_shader)
				{
					THEA_ERROR << "Could not create face index shader";
					return false;
				}

				if (!initFaceIndexShader(*face_index_shader))
				{
					THEA_ERROR << "Could not initialize face index shader";
					return false;
				}
			}

			render_system->setShader(face_index_shader);
		}
		else
		{
			if (!mesh_shader)
			{
				mesh_shader = render_system->createShader("Mesh shader");
				if (!mesh_shader)
				{
					THEA_ERROR << "Could not create mesh shader";
					return false;
				}

				if (!initMeshShader(*mesh_shader))
				{
					THEA_ERROR << "Could not initialize mesh shader";
					return false;
				}
			}

			render_system->setShader(mesh_shader);
		}

		RenderOptions opts = RenderOptions::defaults();
		bool simple_mode = !((color_mode & COLOR_BY_FACE_IDS) | (color_mode & COLOR_BY_FACE_POINT_IDS) | (color_mode & COLOR_BY_FACE_LABELS) | (color_mode & COLOR_BY_FACE_LABELS_WITH_PALETTE));
		if (simple_mode)
		{
			opts.useVertexData() = false;
			opts.sendColors() = false;
			opts.sendNormals() = true;
		}
		else
		{
			opts.useVertexData() = true;
		}


		if (has_transparency && simple_mode)
		{
			// First back faces...
			render_system->setCullFace(RenderSystem::CullFace::FRONT);
			model.orig_mesh_group.draw(*render_system, opts);

			// ... then front faces
			render_system->setCullFace(RenderSystem::CullFace::BACK);
			model.orig_mesh_group.draw(*render_system, opts);
		}
		else
		{
			if (simple_mode)
				model.orig_mesh_group.draw(*render_system, opts);
			else
				model.mesh_group.draw(*render_system, opts);
		}
	}

	render_system->popColorFlags();
	render_system->popShapeFlags();
	render_system->popShader();

	return true;
}

bool ShapeRendererImpl::loadPlugins()
{
	string app_path = FileSystem::resolve(Application::programPath());
	string plugin_dir = FilePath::concat(FilePath::parent(FilePath::parent(app_path)), "lib");

	// Try to load the OpenGL plugin
#ifdef THEA_DEBUG_BUILD

#ifdef THEA_WINDOWS
	string debug_plugin_path = FilePath::concat(plugin_dir, "TheaPluginGLd");
	string release_plugin_path = FilePath::concat(plugin_dir, "TheaPluginGL");
#else
	string debug_plugin_path = FilePath::concat(plugin_dir, "libTheaPluginGLd");
	string release_plugin_path = FilePath::concat(plugin_dir, "libTheaPluginGL");
#endif

#ifdef THEA_WINDOWS
	string debug_plugin_path_ext = debug_plugin_path + ".dll";
#elif THEA_OSX
	string debug_plugin_path_ext = debug_plugin_path + ".dylib";
#else
	string debug_plugin_path_ext = debug_plugin_path + ".so";
#endif

	string plugin_path = FileSystem::exists(debug_plugin_path_ext) ? debug_plugin_path : release_plugin_path;
#else

#ifdef THEA_WINDOWS
	string plugin_path = FilePath::concat(plugin_dir, "TheaPluginGL");
#else
	string plugin_path = FilePath::concat(plugin_dir, "libTheaPluginGL");
#endif

#endif

	Plugin * gl_plugin = Application::getPluginManager().load(plugin_path);
	if (!gl_plugin)
	{
		THEA_ERROR << "Could not load OpenGL plugin: " << plugin_path;
		return false;
	}

	gl_plugin->startup();

	RenderSystemFactory * render_system_factory = Application::getRenderSystemManager().getFactory("OpenGL");
	if (!render_system_factory)
	{
		THEA_ERROR << "Could not get OpenGL rendersystem factory";
		return false;
	}

	render_system = render_system_factory->createRenderSystem("OpenGL RenderSystem");
	if (!render_system)
	{
		THEA_ERROR << "Could not create OpenGL rendersystem";
		return false;
	}

	return true;
}


