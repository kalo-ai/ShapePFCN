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

#include "Common.hpp"
#include "FCNShapes.hpp"

#ifdef _WIN32
#  include "GL/glut.h"
#elif defined(__APPLE__)
#  include "GLUT/glut.h"
#endif

int main(int argc, char * argv[])
{
  // parse input
  if (!parseSettings(argc, argv))
    return -1;

  // Try to open a GLUT window here, see if the rendering code will render into it
#if defined(_WIN32) || defined(__APPLE__)
  if (!Settings::skip_train_rendering || !Settings::skip_test_rendering)
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(Settings::render_size, Settings::render_size);
    glutCreateWindow("Testing if rendering works");
  }
#endif

  MVFCN mvfcn;
  if (Settings::pooling_type == "max")
    mvfcn.setPoolingType(MAX_VIEW_POOLING);
  else if (Settings::pooling_type == "sum")
    mvfcn.setPoolingType(SUM_VIEW_POOLING);
  else
  {
    std::cerr << "Specified Pooling type " << Settings::pooling_type << " is not recognized!\n";
    return -1;
  }


	// train mode
  if (!Settings::train_meshes_path.empty() && !Settings::skip_training)
	{
    mvfcn.train();
	}

  // test mode
  if (!Settings::test_meshes_path.empty() && !Settings::skip_testing)
  {
    mvfcn.test();
  }

	return 0;
}
