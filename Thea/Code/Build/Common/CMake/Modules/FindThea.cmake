# - Searches for an installation of the Thea library
#
# Defines:
#
#   Thea_FOUND           True if Thea was found, else false
#   Thea_LIBRARIES       Libraries to link
#   Thea_LIBRARY_DIRS    Additional directories for libraries. These do not necessarily correspond to Thea_LIBRARIES, and both
#                        variables must be passed to the linker.
#   Thea_INCLUDE_DIRS    The directories containing the header files
#   Thea_CFLAGS          Extra compiler flags
#   Thea_DEBUG_CFLAGS    Extra compiler flags to be used only in debug builds
#   Thea_RELEASE_CFLAGS  Extra compiler flags to be used only in release builds
#   Thea_LDFLAGS         Extra linker flags
#
# To specify an additional directory to search, set Thea_ROOT.
# To prevent automatically searching for all dependencies, set Thea_NO_DEPENDENCIES to true.
#
# Author: Siddhartha Chaudhuri, 2009
#
# Revisions:
#   - 2011-04-12: Locate dependencies automatically, without requiring the caller to do so separately. [SC]
#

SET(Thea_FOUND FALSE)
UNSET(Thea_LIBRARY_DIRS)
UNSET(Thea_LIBRARY_DIRS CACHE)
UNSET(Thea_CFLAGS)
UNSET(Thea_CFLAGS CACHE)
UNSET(Thea_LDFLAGS)
UNSET(Thea_LDFLAGS CACHE)

# Look for the Thea header, first in the user-specified location and then in the system locations
SET(Thea_INCLUDE_DOC "The directory containing the Thea include file Thea/Thea.hpp")
FIND_PATH(Thea_INCLUDE_DIRS NAMES Thea/Common.hpp PATHS ${Thea_ROOT} ${Thea_ROOT}/include ${Thea_ROOT}/Source
          DOC ${Thea_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT Thea_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(Thea_INCLUDE_DIRS NAMES Thea/Common.hpp DOC ${Thea_INCLUDE_DOC})
ENDIF(NOT Thea_INCLUDE_DIRS)

# Only look for the library file in the immediate neighbourhood of the include directory
IF(Thea_INCLUDE_DIRS)
  SET(Thea_LIBRARY_DIRS ${Thea_INCLUDE_DIRS})
  IF("${Thea_LIBRARY_DIRS}" MATCHES "/include$" OR "${Thea_LIBRARY_DIRS}" MATCHES "/Source$")
    # Strip off the trailing "/include" or "/Source" from the path
    GET_FILENAME_COMPONENT(Thea_LIBRARY_DIRS ${Thea_LIBRARY_DIRS} PATH)
  ENDIF("${Thea_LIBRARY_DIRS}" MATCHES "/include$" OR "${Thea_LIBRARY_DIRS}" MATCHES "/Source$")

  FIND_LIBRARY(Thea_DEBUG_LIBRARY
               NAMES Thea_d Thead
               PATH_SUFFIXES "" Debug
               PATHS ${Thea_LIBRARY_DIRS} ${Thea_LIBRARY_DIRS}/lib ${Thea_LIBRARY_DIRS}/Build/lib NO_DEFAULT_PATH)

  FIND_LIBRARY(Thea_RELEASE_LIBRARY
               NAMES Thea
               PATH_SUFFIXES "" Release
               PATHS ${Thea_LIBRARY_DIRS} ${Thea_LIBRARY_DIRS}/lib ${Thea_LIBRARY_DIRS}/Build/lib NO_DEFAULT_PATH)

  UNSET(Thea_LIBRARIES)
  IF(Thea_DEBUG_LIBRARY AND Thea_RELEASE_LIBRARY)
    SET(Thea_LIBRARIES debug ${Thea_DEBUG_LIBRARY} optimized ${Thea_RELEASE_LIBRARY})
  ELSEIF(Thea_DEBUG_LIBRARY)
    SET(Thea_LIBRARIES ${Thea_DEBUG_LIBRARY})
  ELSEIF(Thea_RELEASE_LIBRARY)
    SET(Thea_LIBRARIES ${Thea_RELEASE_LIBRARY})
  ENDIF(Thea_DEBUG_LIBRARY AND Thea_RELEASE_LIBRARY)

  IF(Thea_LIBRARIES)
    SET(Thea_FOUND TRUE)

    # Update the library directories based on the actual library locations
    UNSET(Thea_LIBRARY_DIRS)
    UNSET(Thea_LIBRARY_DIRS CACHE)
    IF(Thea_DEBUG_LIBRARY)
      GET_FILENAME_COMPONENT(Thea_LIBDIR ${Thea_DEBUG_LIBRARY} PATH)
      SET(Thea_LIBRARY_DIRS ${Thea_LIBRARY_DIRS} ${Thea_LIBDIR})
    ENDIF(Thea_DEBUG_LIBRARY)
    IF(Thea_RELEASE_LIBRARY)
      GET_FILENAME_COMPONENT(Thea_LIBDIR ${Thea_RELEASE_LIBRARY} PATH)
      SET(Thea_LIBRARY_DIRS ${Thea_LIBRARY_DIRS} ${Thea_LIBDIR})
    ENDIF(Thea_RELEASE_LIBRARY)

    # Flags for importing symbols from dynamically linked libraries
    IF(WIN32)
      # What's a good way of testing whether the .lib is static, or merely exports symbols from a DLL? For now, let's assume
      # it always exports (or hope that __declspec(dllimport) is a noop for static libraries)
      SET(Thea_CFLAGS "-DTHEA_DLL -DTHEA_DLL_IMPORTS")
    ELSE(WIN32)
      IF("${Thea_LIBRARIES}" MATCHES ".dylib$" OR "${Thea_LIBRARIES}" MATCHES ".so$")
        SET(Thea_CFLAGS "-DTHEA_DLL -DTHEA_DLL_IMPORTS")
      ENDIF("${Thea_LIBRARIES}" MATCHES ".dylib$" OR "${Thea_LIBRARIES}" MATCHES ".so$")
    ENDIF(WIN32)

    # Read extra flags to be used to build Thea
    SET(Thea_BUILD_FLAGS_FILE "${Thea_INCLUDE_DIRS}/Thea/BuildFlags.txt")
    IF(EXISTS "${Thea_BUILD_FLAGS_FILE}")
      FILE(READ "${Thea_BUILD_FLAGS_FILE}" Thea_BUILD_FLAGS)
      STRING(REGEX REPLACE "\n" " " Thea_BUILD_FLAGS "${Thea_BUILD_FLAGS}")
      SET(Thea_CFLAGS "${Thea_CFLAGS} ${Thea_BUILD_FLAGS}")
    ENDIF(EXISTS "${Thea_BUILD_FLAGS_FILE}")

  ENDIF(Thea_LIBRARIES)
ENDIF(Thea_INCLUDE_DIRS)

IF(NOT Thea_NO_DEPENDENCIES)

# Dependency: Boost
IF(Thea_FOUND)
  SET(Boost_USE_MULTITHREADED    ON)
  # SET(Boost_USE_STATIC_LIBS      ON)
  # SET(Boost_USE_STATIC_RUNTIME  OFF)
  INCLUDE(BoostAdditionalVersions)
  IF(EXISTS ${Thea_ROOT}/installed-boost)
    SET(BOOST_ROOT ${Thea_ROOT}/installed-boost)
  ELSE(EXISTS ${Thea_ROOT}/installed-boost)
    SET(BOOST_ROOT ${Thea_ROOT})
  ENDIF(EXISTS ${Thea_ROOT}/installed-boost)
  FIND_PACKAGE(Boost COMPONENTS filesystem thread system)
  IF(Boost_FOUND)
    SET(Thea_LIBRARIES ${Thea_LIBRARIES} ${Boost_LIBRARIES})
    SET(Thea_INCLUDE_DIRS ${Thea_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  ELSE(Boost_FOUND)
    MESSAGE(STATUS "Thea: Boost not found")
    SET(Thea_FOUND FALSE)
  ENDIF(Boost_FOUND)
ENDIF(Thea_FOUND)

# Dependency: Lib3ds
IF(Thea_FOUND)
  IF(EXISTS ${Thea_ROOT}/installed-lib3ds)
    SET(Lib3ds_ROOT ${Thea_ROOT}/installed-lib3ds)
  ELSE(EXISTS ${Thea_ROOT}/installed-lib3ds)
    SET(Lib3ds_ROOT ${Thea_ROOT})
  ENDIF(EXISTS ${Thea_ROOT}/installed-lib3ds)
  FIND_PACKAGE(Lib3ds)
  IF(Lib3ds_FOUND)
    SET(Thea_LIBRARIES ${Thea_LIBRARIES} ${Lib3ds_LIBRARIES})
    SET(Thea_INCLUDE_DIRS ${Thea_INCLUDE_DIRS} ${Lib3ds_INCLUDE_DIRS})
    SET(Thea_CFLAGS ${Thea_CFLAGS} -DTHEA_LIB3DS_VERSION_MAJOR=${Lib3ds_VERSION_MAJOR})
  ELSE(Lib3ds_FOUND)
    MESSAGE(STATUS "Thea: lib3ds not found")
    SET(Thea_FOUND FALSE)
  ENDIF(Lib3ds_FOUND)
ENDIF(Thea_FOUND)

# Dependency: CLUTO
IF(Thea_FOUND)
  IF(APPLE)
    IF(CMAKE_SIZEOF_VOID_P EQUAL 4)  # We don't have Cluto built for 64-bit OS X, i.e. Snow Leopard (10.6)
      SET(_Thea_FIND_CLUTO TRUE)
    ELSE(CMAKE_SIZEOF_VOID_P EQUAL 4)
      SET(_Thea_FIND_CLUTO FALSE)
    ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
  ELSE(APPLE)
    SET(_Thea_FIND_CLUTO TRUE)
  ENDIF(APPLE)

  IF(_Thea_FIND_CLUTO)
    IF(EXISTS ${Thea_ROOT}/installed-cluto)
      SET(CLUTO_ROOT ${Thea_ROOT}/installed-cluto)
    ELSE(EXISTS ${Thea_ROOT}/installed-cluto)
      SET(CLUTO_ROOT ${Thea_ROOT})
    ENDIF(EXISTS ${Thea_ROOT}/installed-cluto)
    FIND_PACKAGE(CLUTO)
    IF(CLUTO_FOUND)
      SET(Thea_LIBRARIES ${Thea_LIBRARIES} ${CLUTO_LIBRARIES})
      SET(Thea_INCLUDE_DIRS ${Thea_INCLUDE_DIRS} ${CLUTO_INCLUDE_DIRS})
      SET(Thea_CFLAGS ${Thea_CFLAGS} -DTHEA_ENABLE_CLUTO)
    ELSE(CLUTO_FOUND)
      MESSAGE(STATUS "Thea: CLUTO not found")  # this is not a fatal error
    ENDIF(CLUTO_FOUND)
  ELSE(_Thea_FIND_CLUTO)
    MESSAGE(STATUS "NOTE: CLUTO not available for this system: Thea will not be able to use CLUTO for clustering")
  ENDIF(_Thea_FIND_CLUTO)
ENDIF(Thea_FOUND)

# Dependency: FreeImage
IF(Thea_FOUND)
  IF(EXISTS ${Thea_ROOT}/installed-freeimage)
    SET(FreeImage_ROOT ${Thea_ROOT}/installed-freeimage)
  ELSE(EXISTS ${Thea_ROOT}/installed-freeimage)
    SET(FreeImage_ROOT ${Thea_ROOT})
  ENDIF(EXISTS ${Thea_ROOT}/installed-freeimage)
  SET(FreeImage_LANGUAGE "C++")
  FIND_PACKAGE(FreeImage REQUIRED)
  IF(FreeImage_FOUND)
    SET(Thea_LIBRARIES ${Thea_LIBRARIES} ${FreeImage_LIBRARIES})
    SET(Thea_INCLUDE_DIRS ${Thea_INCLUDE_DIRS} ${FreeImage_INCLUDE_DIRS})
  ELSE(FreeImage_FOUND)
    MESSAGE(STATUS "Thea: FreeImage not found")
    SET(Thea_FOUND FALSE)
  ENDIF(FreeImage_FOUND)
ENDIF(Thea_FOUND)

# Platform libs
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET(Thea_LIBRARIES ${Thea_LIBRARIES} "-framework Carbon")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

SET(Thea_LIBRARIES ${Thea_LIBRARIES} ${CMAKE_DL_LIBS})  # for loading plugins with DynLib

ENDIF(NOT Thea_NO_DEPENDENCIES)

# Remove duplicate entries from lists, else the same dirs and flags can repeat many times

# Don't remove duplicates from Thea_LIBRARIES -- the list includes repetitions of "debug" and "optimized"

IF(Thea_LIBRARY_DIRS)
  LIST(REMOVE_DUPLICATES Thea_LIBRARY_DIRS)
ENDIF(Thea_LIBRARY_DIRS)

IF(Thea_INCLUDE_DIRS)
  LIST(REMOVE_DUPLICATES Thea_INCLUDE_DIRS)
ENDIF(Thea_INCLUDE_DIRS)

IF(Thea_CFLAGS)
  LIST(REMOVE_DUPLICATES Thea_CFLAGS)
ENDIF(Thea_CFLAGS)

IF(Thea_DEBUG_CFLAGS)
  LIST(REMOVE_DUPLICATES Thea_DEBUG_CFLAGS)
ENDIF(Thea_DEBUG_CFLAGS)

IF(Thea_RELEASE_CFLAGS)
  LIST(REMOVE_DUPLICATES Thea_RELEASE_CFLAGS)
ENDIF(Thea_RELEASE_CFLAGS)

IF(Thea_LDFLAGS)
  LIST(REMOVE_DUPLICATES Thea_LDFLAGS)
ENDIF(Thea_LDFLAGS)

SET(Thea_LIBRARY_DIRS ${Thea_LIBRARY_DIRS} CACHE STRING "Additional directories for libraries required by Thea" FORCE)
SET(Thea_CFLAGS ${Thea_CFLAGS} CACHE STRING "Extra compiler flags required by Thea" FORCE)
SET(Thea_LDFLAGS ${Thea_LDFLAGS} CACHE STRING "Extra linker flags required by Thea" FORCE)

IF(Thea_FOUND)
  IF(NOT Thea_FIND_QUIETLY)
    MESSAGE(STATUS "Found Thea: headers at ${Thea_INCLUDE_DIRS}, libraries at ${Thea_LIBRARIES}")
  ENDIF(NOT Thea_FIND_QUIETLY)
ELSE(Thea_FOUND)
  IF(Thea_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Thea not found")
  ENDIF(Thea_FIND_REQUIRED)
ENDIF(Thea_FOUND)
