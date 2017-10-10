# - Searches for an installation of the Eigen library
#
# Defines:
#
#   Eigen_FOUND         True if Eigen was found, else false
#   Eigen_INCLUDE_DIRS  The directories containing the header files (Eigen is header-only)
#
# To specify an additional directory to search, set Eigen_ROOT.
#
# Author: Siddhartha Chaudhuri, 2016
#

SET(Eigen_FOUND FALSE)

# Look for the Eigen header, first in the user-specified location and then in the system locations
SET(Eigen_INCLUDE_DOC "The directory containing the Eigen include file Eigen/Core")
FIND_PATH(Eigen_INCLUDE_DIRS NAMES Eigen/Core eigen/Core Eigen/core eigen/core PATHS ${Eigen_ROOT} ${Eigen_ROOT}/include
          DOC ${Eigen_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT Eigen_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(Eigen_INCLUDE_DIRS NAMES Eigen/Core eigen/Core Eigen/core eigen/core DOC ${Eigen_INCLUDE_DOC})
ENDIF(NOT Eigen_INCLUDE_DIRS)

IF(Eigen_INCLUDE_DIRS)
  SET(Eigen_FOUND TRUE)
ENDIF(Eigen_INCLUDE_DIRS)

IF(Eigen_FOUND)
  IF(NOT Eigen_FIND_QUIETLY)
    MESSAGE(STATUS "Found Eigen: headers at ${Eigen_INCLUDE_DIRS}")
  ENDIF(NOT Eigen_FIND_QUIETLY)
ELSE(Eigen_FOUND)
  IF(Eigen_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Eigen not found")
  ENDIF(Eigen_FIND_REQUIRED)
ENDIF(Eigen_FOUND)
