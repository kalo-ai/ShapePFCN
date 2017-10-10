# - Searches for an installation of the NanoGUI library
#
# Defines:
#
#   NanoGUI_FOUND         True if NanoGUI was found, else false
#   NanoGUI_LIBRARIES     Libraries to link
#   NanoGUI_INCLUDE_DIRS  The directories containing the header files
#
# To specify an additional directory to search, set NanoGUI_ROOT.
#
# Author: Siddhartha Chaudhuri, 2016
#

SET(NanoGUI_FOUND FALSE)

# Look for the NanoGUI header, first in the user-specified location and then in the system locations
SET(NanoGUI_INCLUDE_DOC "The directory containing the NanoGUI include file nanogui/nanogui.h")
FIND_PATH(NanoGUI_INCLUDE_DIRS NAMES nanogui/nanogui.h PATHS ${NanoGUI_ROOT} ${NanoGUI_ROOT}/include
          DOC ${NanoGUI_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT NanoGUI_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(NanoGUI_INCLUDE_DIRS NAMES nanogui/nanogui.h DOC ${NanoGUI_INCLUDE_DOC})
ENDIF(NOT NanoGUI_INCLUDE_DIRS)

# Look for the NanoVG header, first in the user-specified location and then in the system locations
SET(NanoGUI_NVG_INCLUDE_DIRS)
IF(NanoGUI_INCLUDE_DIRS)
  SET(NanoGUI_NVG_INCLUDE_DOC "The directory containing the NanoVG include file nanovg.h")
  FIND_PATH(NanoGUI_NVG_INCLUDE_DIRS NAMES nanovg.h PATHS ${NanoGUI_ROOT} ${NanoGUI_ROOT}/include
            DOC ${NanoGUI_NVG_INCLUDE_DOC} NO_DEFAULT_PATH)
  IF(NOT NanoGUI_NVG_INCLUDE_DIRS)  # now look in system locations
    FIND_PATH(NanoGUI_NVG_INCLUDE_DIRS NAMES nanovg.h DOC ${NanoGUI_NVG_INCLUDE_DOC})
  ENDIF(NOT NanoGUI_NVG_INCLUDE_DIRS)

  IF(NanoGUI_NVG_INCLUDE_DIRS)
    SET(NanoGUI_INCLUDE_DIRS ${NanoGUI_INCLUDE_DIRS} ${NanoGUI_NVG_INCLUDE_DIRS})
    LIST(REMOVE_DUPLICATES NanoGUI_INCLUDE_DIRS)
  ELSE(NanoGUI_NVG_INCLUDE_DIRS)
    SET(NanoGUI_INCLUDE_DIRS)
  ENDIF(NanoGUI_NVG_INCLUDE_DIRS)
ENDIF(NanoGUI_INCLUDE_DIRS)

# Look for the Eigen header, first in the user-specified location and then in the system locations
IF(NanoGUI_INCLUDE_DIRS)
  SET(NanoGUI_Eigen_INCLUDE_DOC "The directory containing the Eigen include file Eigen/Core")
  FIND_PATH(NanoGUI_Eigen_INCLUDE_DIRS NAMES Eigen/Core eigen/Core Eigen/core eigen/core PATHS
            ${NanoGUI_ROOT} ${NanoGUI_ROOT}/include DOC ${NanoGUI_Eigen_INCLUDE_DOC} NO_DEFAULT_PATH)
  IF(NOT NanoGUI_Eigen_INCLUDE_DIRS)  # now look in system locations
    FIND_PATH(NanoGUI_Eigen_INCLUDE_DIRS NAMES Eigen/Core eigen/Core Eigen/core eigen/core DOC ${NanoGUI_Eigen_INCLUDE_DOC})
  ENDIF(NOT NanoGUI_Eigen_INCLUDE_DIRS)

  IF(NanoGUI_NVG_INCLUDE_DIRS)
    SET(NanoGUI_INCLUDE_DIRS ${NanoGUI_INCLUDE_DIRS} ${NanoGUI_Eigen_INCLUDE_DIRS})
    LIST(REMOVE_DUPLICATES NanoGUI_INCLUDE_DIRS)
  ELSE(NanoGUI_NVG_INCLUDE_DIRS)
    SET(NanoGUI_INCLUDE_DIRS)
  ENDIF(NanoGUI_NVG_INCLUDE_DIRS)
ENDIF(NanoGUI_INCLUDE_DIRS)

# Only look for the library file in the immediate neighbourhood of the include directory
IF(NanoGUI_INCLUDE_DIRS)
  LIST(GET NanoGUI_INCLUDE_DIRS 0 NanoGUI_LIBRARY_DIRS)
  IF("${NanoGUI_LIBRARY_DIRS}" MATCHES "/include$")
    # Strip off the trailing "/include" or "/Source" from the path
    GET_FILENAME_COMPONENT(NanoGUI_LIBRARY_DIRS ${NanoGUI_LIBRARY_DIRS} PATH)
  ENDIF("${NanoGUI_LIBRARY_DIRS}" MATCHES "/include$")

  FIND_LIBRARY(NanoGUI_DEBUG_LIBRARY
               NAMES nanoguid NanoGUId
               PATH_SUFFIXES "" Debug
               PATHS ${NanoGUI_LIBRARY_DIRS} ${NanoGUI_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)
  IF(NOT NanoGUI_DEBUG_LIBRARY)
    FIND_LIBRARY(NanoGUI_DEBUG_LIBRARY
                 NAMES nanogui NanoGUI
                 PATHS ${NanoGUI_LIBRARY_DIRS}/Debug ${NanoGUI_LIBRARY_DIRS}/lib/Debug NO_DEFAULT_PATH)
  ENDIF(NOT NanoGUI_DEBUG_LIBRARY)

  FIND_LIBRARY(NanoGUI_RELEASE_LIBRARY
               NAMES nanogui NanoGUI
               PATH_SUFFIXES "" Release
               PATHS ${NanoGUI_LIBRARY_DIRS} ${NanoGUI_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)

  SET(NanoGUI_LIBRARIES)
  IF(NanoGUI_DEBUG_LIBRARY AND NanoGUI_RELEASE_LIBRARY)
    SET(NanoGUI_LIBRARIES debug ${NanoGUI_DEBUG_LIBRARY} optimized ${NanoGUI_RELEASE_LIBRARY})
  ELSEIF(NanoGUI_DEBUG_LIBRARY)
    SET(NanoGUI_LIBRARIES ${NanoGUI_DEBUG_LIBRARY})
  ELSEIF(NanoGUI_RELEASE_LIBRARY)
    SET(NanoGUI_LIBRARIES ${NanoGUI_RELEASE_LIBRARY})
  ENDIF(NanoGUI_DEBUG_LIBRARY AND NanoGUI_RELEASE_LIBRARY)

  IF(NanoGUI_LIBRARIES)
    SET(NanoGUI_FOUND TRUE)
  ENDIF(NanoGUI_LIBRARIES)
ENDIF(NanoGUI_INCLUDE_DIRS)

IF(NanoGUI_FOUND)
  IF(NOT NanoGUI_FIND_QUIETLY)
    MESSAGE(STATUS "Found NanoGUI: headers at ${NanoGUI_INCLUDE_DIRS}, libraries at ${NanoGUI_LIBRARIES}")
  ENDIF(NOT NanoGUI_FIND_QUIETLY)
ELSE(NanoGUI_FOUND)
  IF(NanoGUI_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "NanoGUI not found")
  ENDIF(NanoGUI_FIND_REQUIRED)
ENDIF(NanoGUI_FOUND)
