# Searches for an installation of the GLFW 3.x library. On success, it sets the following variables:
#
#   GLFW3_FOUND              Set to true to indicate the glfw library was found
#   GLFW3_INCLUDE_DIRS       The directory containing the header file GL/glfw.h
#   GLFW3_LIBRARIES          The libraries needed to use the glfw library
#
# To specify an additional directory to search, set GLFW3_ROOT.
#
# Author: Siddhartha Chaudhuri, 2016
#

# Look for the header, first in the user-specified location and then in the system locations
SET(GLFW3_INCLUDE_DOC "The directory containing the header file glfw3.h")
FIND_PATH(GLFW3_INCLUDE_DIRS NAMES GLFW/glfw3.h GL/glfw3.h PATHS ${GLFW3_ROOT} ${GLFW3_ROOT}/include DOC ${GLFW3_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT GLFW3_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(GLFW3_INCLUDE_DIRS NAMES GLFW/glfw3.h GL/glfw3.h DOC ${GLFW3_INCLUDE_DOC})
ENDIF(NOT GLFW3_INCLUDE_DIRS)

SET(GLFW3_FOUND FALSE)

IF(GLFW3_INCLUDE_DIRS)
  SET(GLFW3_LIBRARY_DIRS ${GLFW3_INCLUDE_DIRS})

  IF("${GLFW3_LIBRARY_DIRS}" MATCHES "/include$")
    # Strip off the trailing "/include" in the path.
    GET_FILENAME_COMPONENT(GLFW3_LIBRARY_DIRS ${GLFW3_LIBRARY_DIRS} PATH)
  ENDIF("${GLFW3_LIBRARY_DIRS}" MATCHES "/include$")

  IF(EXISTS "${GLFW3_LIBRARY_DIRS}/lib")
    SET(GLFW3_LIBRARY_DIRS ${GLFW3_LIBRARY_DIRS}/lib)
  ENDIF(EXISTS "${GLFW3_LIBRARY_DIRS}/lib")

  # Find GLFW libraries
  FIND_LIBRARY(GLFW3_LIBRARIES NAMES glfw3 GLFW3 PATHS ${GLFW3_LIBRARY_DIRS} NO_DEFAULT_PATH)

  IF(GLFW3_LIBRARIES)
    SET(GLFW3_FOUND TRUE)
  ENDIF(GLFW3_LIBRARIES)
ENDIF(GLFW3_INCLUDE_DIRS)

IF(GLFW3_FOUND)
  IF(NOT GLFW3_FIND_QUIETLY)
    MESSAGE(STATUS "Found GLFW 3.x: headers at ${GLFW3_INCLUDE_DIRS}, libraries at ${GLFW3_LIBRARIES}")
  ENDIF(NOT GLFW3_FIND_QUIETLY)
ELSE(GLFW3_FOUND)
  IF(GLFW3_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "GLFW library not found")
  ENDIF(GLFW3_FIND_REQUIRED)
ENDIF(GLFW3_FOUND)
