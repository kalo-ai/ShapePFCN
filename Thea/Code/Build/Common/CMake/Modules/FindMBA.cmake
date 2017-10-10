# - Searches for an installation of the MBA library
#
# Defines:
#
#   MBA_FOUND           True if MBA was found, else false
#   MBA_LIBRARIES       Libraries to link
#   MBA_INCLUDE_DIRS    The directories containing the header files
#
# To specify an additional directory to search, set MBA_ROOT.
#
# Author: Siddhartha Chaudhuri, 2015
#

SET(MBA_FOUND FALSE)

# Look for the MBA header, first in the user-specified location and then in the system locations
SET(MBA_INCLUDE_DOC "The directory containing the MBA include file MBA/DatabaseManager.hpp")
FIND_PATH(MBA_INCLUDE_DIRS NAMES MBA/DatabaseManager.hpp PATHS ${MBA_ROOT} ${MBA_ROOT}/include ${MBA_ROOT}/Source
          DOC ${MBA_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT MBA_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(MBA_INCLUDE_DIRS NAMES MBA/DatabaseManager.hpp DOC ${MBA_INCLUDE_DOC})
ENDIF(NOT MBA_INCLUDE_DIRS)

# Only look for the library file in the immediate neighbourhood of the include directory
IF(MBA_INCLUDE_DIRS)
  SET(MBA_LIBRARY_DIRS ${MBA_INCLUDE_DIRS})
  IF("${MBA_LIBRARY_DIRS}" MATCHES "/include$" OR "${MBA_LIBRARY_DIRS}" MATCHES "/Source$")
    # Strip off the trailing "/include" or "/Source" from the path
    GET_FILENAME_COMPONENT(MBA_LIBRARY_DIRS ${MBA_LIBRARY_DIRS} PATH)
  ENDIF("${MBA_LIBRARY_DIRS}" MATCHES "/include$" OR "${MBA_LIBRARY_DIRS}" MATCHES "/Source$")

  FIND_LIBRARY(MBA_DEBUG_LIBRARY
               NAMES MBA_d MBAd
               PATH_SUFFIXES "" Debug
               PATHS ${MBA_LIBRARY_DIRS} ${MBA_LIBRARY_DIRS}/lib ${MBA_LIBRARY_DIRS}/Build/lib NO_DEFAULT_PATH)

  FIND_LIBRARY(MBA_RELEASE_LIBRARY
               NAMES MBA
               PATH_SUFFIXES "" Release
               PATHS ${MBA_LIBRARY_DIRS} ${MBA_LIBRARY_DIRS}/lib ${MBA_LIBRARY_DIRS}/Build/lib NO_DEFAULT_PATH)

  UNSET(MBA_LIBRARIES)
  IF(MBA_DEBUG_LIBRARY AND MBA_RELEASE_LIBRARY)
    SET(MBA_LIBRARIES debug ${MBA_DEBUG_LIBRARY} optimized ${MBA_RELEASE_LIBRARY})
  ELSEIF(MBA_DEBUG_LIBRARY)
    SET(MBA_LIBRARIES ${MBA_DEBUG_LIBRARY})
  ELSEIF(MBA_RELEASE_LIBRARY)
    SET(MBA_LIBRARIES ${MBA_RELEASE_LIBRARY})
  ENDIF(MBA_DEBUG_LIBRARY AND MBA_RELEASE_LIBRARY)

  IF(MBA_LIBRARIES)
    SET(MBA_FOUND TRUE)
  ENDIF(MBA_LIBRARIES)
ENDIF(MBA_INCLUDE_DIRS)

IF(MBA_FOUND)
  IF(NOT MBA_FIND_QUIETLY)
    MESSAGE(STATUS "Found MBA: headers at ${MBA_INCLUDE_DIRS}, libraries at ${MBA_LIBRARIES}")
  ENDIF(NOT MBA_FIND_QUIETLY)
ELSE(MBA_FOUND)
  IF(MBA_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "MBA not found")
  ENDIF(MBA_FIND_REQUIRED)
ENDIF(MBA_FOUND)
