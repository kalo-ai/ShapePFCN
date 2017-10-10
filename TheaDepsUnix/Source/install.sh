#!/bin/bash

################################################################################################################################
# Initialize variables
################################################################################################################################

USE_ROOT=false
USER=`whoami`
SAME_USER=true
NUM_PROCS=-j2
SCRIPT_DIR=`dirname "$0"`
BASE_DIR="`perl -e 'use Cwd "abs_path";print abs_path(shift)' "$SCRIPT_DIR"`"
INSTALL_DIR="${BASE_DIR}/Installations"
HOST_OS=`uname -s`
APT_GET="apt-get -y"
WITH_BOOST=false
WITH_G3D=false
WITH_FREEIMAGEPLUS=false
WITH_CLUTO=false
WITH_CGAL=false
WITH_EIGEN=false
WITH_OPENMESH=false
WITH_OSMESA=false
WITH_SIRIKATA=false
WITH_S3FS=false
WITH_LAPACK=false
WITH_ARPACK=false
WITH_ARPACKPP=false
WITH_PETSC=false
WITH_SLEPC=false
WITH_LPSOLVE=false
WITH_OPTPP=false
WITH_FFTW=false
WITH_LIB3DS=false
WITH_GLFW=false
WITH_ANN=false
ANN_FLOAT=false
WITH_PNL=false
WITH_QT4=false
WITH_WXWIDGETS=false
WITH_NANOGUI=false
WITH_NANOGUI_LATEST=false
WITH_THEA=false
WITH_SNL=false
SNL_INSTALL_DEPS=false
WITH_ICONS_TANGO=false
WITH_ICONS_TANGO_ART_LIBRE=false
WITH_ANY=false


################################################################################################################################
# The help message
################################################################################################################################

function show_help {
    echo "
Usage: $0 [--use-root] [--user <username>] [-j<num-procs>]
           [--prefix <prefix-path>]
           [--with-<package>] ...
           [--installed-<package> <path>] ...
           [--ann-float]
           [--thea-install-deps]
           [--snl-install-deps]

Each <package> must be one of:
  boost, g3d, freeimageplus, cluto, cgal, eigen, openmesh, osmesa, sirikata,
  s3fs, lapack, arpack, arpack++, petsc, slepc, lp_solve, fftw, opt++, lib3ds,
  glfw, ann, pnl, qt4, nanogui, wxwidgets, thea, snl, icons-tango,
  icons-tango-art-libre

(For s3fs <path> should be the location of the s3fs executable.)

The --installed... option is ignored if the corresponding --with... option is
not present.

--ann-float: Build ANN for single, rather than double-precision computations.
--thea-install-deps: Install all dependencies required by Thea.

You must run the script with root privileges (sudo) if you specify --use-root.
$0 --help displays this help message.
"
}


################################################################################################################################
# Parse command-line arguments
################################################################################################################################

until [ -z "$1" ]  # until all parameters used up
do
  case "$1" in
      --use-root ) USE_ROOT=true ;;
      --user )
          shift
          USER="$1"
          ;;
      -j* ) NUM_PROCS="$1" ;;
      --prefix )
          shift
          INSTALL_DIR="$1"
          ;;
      --with-boost )
          WITH_BOOST=true
          WITH_ANY=true
          ;;
      --installed-boost )
          shift
          INSTALLED_BOOST="$1"
          ;;
      --with-g3d )
          WITH_G3D=true
          WITH_ANY=true
          ;;
      --installed-g3d )
          shift
          INSTALLED_G3D="$1"
          ;;
      --with-freeimageplus )
          WITH_FREEIMAGEPLUS=true
          WITH_ANY=true
          ;;
      --installed-freeimageplus )
          shift
          INSTALLED_FREEIMAGEPLUS="$1"
          ;;
      --with-cluto )
          WITH_CLUTO=true
          WITH_ANY=true
          ;;
      --installed-cluto )
          shift
          INSTALLED_CLUTO="$1"
          ;;
      --with-cgal )
          WITH_CGAL=true
          WITH_ANY=true
          ;;
      --installed-cgal )
          shift
          INSTALLED_CGAL="$1"
          ;;
      --with-eigen )
          WITH_EIGEN=true
          WITH_ANY=true
          ;;
      --installed-eigen )
          shift
          INSTALLED_EIGEN="$1"
          ;;
      --with-openmesh )
          WITH_OPENMESH=true
          WITH_ANY=true
          ;;
      --installed-openmesh )
          shift
          INSTALLED_OPENMESH="$1"
          ;;
      --with-osmesa )
          WITH_OSMESA=true
          WITH_ANY=true
          ;;
      --installed-osmesa )
          shift
          INSTALLED_OSMESA="$1"
          ;;
      --with-sirikata )
          WITH_SIRIKATA=true
          WITH_ANY=true
          ;;
      --installed-sirikata )
          shift
          INSTALLED_SIRIKATA="$1"
          ;;
      --with-s3fs )
          WITH_S3FS=true
          WITH_ANY=true
          ;;
      --installed-s3fs )
          shift
          INSTALLED_S3FS="$1"
          ;;
      --with-lapack )
          WITH_LAPACK=true
          WITH_ANY=true
          ;;
      --installed-lapack )
          shift
          INSTALLED_LAPACK="$1"
          ;;
      --with-arpack )
          WITH_ARPACK=true
          WITH_ANY=true
          ;;
      --installed-arpack )
          shift
          INSTALLED_ARPACK="$1"
          ;;
      --with-arpack++ )
          WITH_ARPACKPP=true
          WITH_ANY=true
          ;;
      --installed-arpack++ )
          shift
          INSTALLED_ARPACKPP="$1"
          ;;
      --with-petsc )
          WITH_PETSC=true
          WITH_ANY=true
          ;;
      --installed-petsc )
          shift
          INSTALLED_PETSC="$1"
          ;;
      --with-slepc )
          WITH_SLEPC=true
          WITH_ANY=true
          ;;
      --installed-slepc )
          shift
          INSTALLED_SLEPC="$1"
          ;;
      --with-lp_solve )
          WITH_LPSOLVE=true
          WITH_ANY=true
          ;;
      --installed-lp_solve )
          shift
          INSTALLED_LPSOLVE="$1"
          ;;
      --with-fftw )
          WITH_FFTW=true
          WITH_ANY=true
          ;;
      --installed-fftw )
          shift
          INSTALLED_FFTW="$1"
          ;;
      --with-opt++ )
          WITH_OPTPP=true
          WITH_ANY=true
          ;;
      --installed-opt++ )
          shift
          INSTALLED_OPTPP="$1"
          ;;
      --with-lib3ds )
          WITH_LIB3DS=true
          WITH_ANY=true
          ;;
      --installed-lib3ds )
          shift
          INSTALLED_LIB3DS="$1"
          ;;
      --with-glfw )
          WITH_GLFW=true
          WITH_ANY=true
          ;;
      --installed-glfw )
          shift
          INSTALLED_GLFW="$1"
          ;;
      --with-ann )
          WITH_ANN=true
          WITH_ANY=true
          ;;
      --installed-ann )
          shift
          INSTALLED_ANN="$1"
          ;;
      --ann-float )
          ANN_FLOAT=true
          ;;
      --with-qt4 )
          WITH_QT4=true
          WITH_ANY=true
          ;;
      --installed-qt4 )
          shift
          INSTALLED_QT4="$1"
          ;;
      --with-wxwidgets )
          WITH_WXWIDGETS=true
          WITH_ANY=true
          ;;
      --installed-wxwidgets )
          shift
          INSTALLED_WXWIDGETS="$1"
          ;;
      --with-nanogui )
          WITH_NANOGUI=true
          WITH_ANY=true
          ;;
      --with-nanogui-latest )
          WITH_NANOGUI_LATEST=true
          WITH_ANY=true
          ;;
      --installed-nanogui )
          shift
          INSTALLED_NANOGUI="$1"
          ;;
      --with-pnl )
          WITH_PNL=true
          WITH_ANY=true
          ;;
      --installed-pnl )
          shift
          INSTALLED_PNL="$1"
          ;;
      --with-thea )
          WITH_THEA=true
          WITH_ANY=true
          ;;
      --installed-thea )
          shift
          INSTALLED_THEA="$1"
          ;;
      --with-snl )
          WITH_SNL=true
          WITH_ANY=true
          ;;
      --installed-snl )
          shift
          INSTALLED_SNL="$1"
          ;;
      --snl-install-deps )
          SNL_INSTALL_DEPS=true
          ;;
      --with-icons-tango )
          WITH_ICONS_TANGO=true
          WITH_ANY=true
          ;;
      --installed-icons-tango )
          shift
          INSTALLED_ICONS_TANGO="$1"
          ;;
      --with-icons-tango-art-libre )
          WITH_ICONS_TANGO_ART_LIBRE=true
          WITH_ANY=true
          ;;
      --installed-icons-tango-art-libre )
          shift
          INSTALLED_ICONS_TANGO_ART_LIBRE="$1"
          ;;
      --help )
          show_help
          exit
          ;;
  esac
  shift
done


################################################################################################################################
# See if we're supposed to do any work
################################################################################################################################

if ! $WITH_ANY ; then
  show_help
  exit
fi


################################################################################################################################
# Notify the user about the installation directory
################################################################################################################################

echo "Installing packages to $INSTALL_DIR"


################################################################################################################################
# Check for superuser
################################################################################################################################

if $USE_ROOT ; then
    while test $UID != "0" ; do
        echo "
Please run this script with root privileges when you specify --use-root.
Type $0 --help to see a list of all options.
"
        exit -1;
    done
fi


################################################################################################################################
# Check for distro
################################################################################################################################

if $USE_ROOT ; then
    if [[ $HOST_OS =~ .*inux.* && -n `which apt-get` ]] ; then
        DISTRO=debian
    else
        # TODO: Replace with message listing packages that should be installed as root, as per Patrick's suggestion
        echo "Installing as root is not supported on your distribution."
        exit
    fi
fi


################################################################################################################################
# Install support packages
################################################################################################################################

if $USE_ROOT ; then
     case "$DISTRO" in
         debian )
             ${APT_GET} install tar unzip zip automake jam g++ autoconf libtool patch git-core cmake pkg-config fort77
             if $WITH_S3FS ; then
                 ${APT_GET} install libfuse2 libxml2 libcurl4-openssl-dev libssl-dev
             fi
             ;;
     esac
fi


################################################################################################################################
# Check if the user runnning the script is the the same as the one specified on the command line
################################################################################################################################

if [[ $USER != `whoami` ]] ; then
    SAME_USER=false
fi


################################################################################################################################
# Define a function to execute a command as a different user if necessary
################################################################################################################################

function user_eval {
    if $SAME_USER ; then
        eval $1
    else
        su -c "$1" - $USER
    fi
}


################################################################################################################################
# Define a function to make a symlink to an existing installation
################################################################################################################################

function make_symlink {
    TARGET=$1
    LINK_NAME=$2

    user_eval "cd '${BASE_DIR}'; rm -f '$LINK_NAME'; ln -s '$TARGET' '$LINK_NAME'"
}


################################################################################################################################
# Define a function to check if a library exists in any one of the standard directories: check_library <prefix>
################################################################################################################################

function check_library_in_location {
    LIB_PREFIX=$1
    LIB_LOCATION=$2

    return $(expr $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.a 2>/dev/null | wc -l) \
                + $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.a.* 2>/dev/null | wc -l) \
                + $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.so 2>/dev/null | wc -l) \
                + $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.so.* 2>/dev/null | wc -l) \
                + $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.dylib 2>/dev/null | wc -l) \
                + $(ls ${LIB_LOCATION}/${LIB_PREFIX}*.dylib.* 2>/dev/null | wc -l))
}

function check_library {
    LIB_PREFIX=$1

    check_library_in_location ${LIB_PREFIX} /usr/lib;                         LIB_FOUND_1=$?
    check_library_in_location ${LIB_PREFIX} /usr/local/lib;                   LIB_FOUND_2=$?
    check_library_in_location ${LIB_PREFIX} /opt/lib;                         LIB_FOUND_3=$?
    check_library_in_location ${LIB_PREFIX} /opt/local/lib;                   LIB_FOUND_4=$?
    check_library_in_location ${LIB_PREFIX} /sw/lib;                          LIB_FOUND_5=$?
    check_library_in_location ${LIB_PREFIX} /sw/local/lib;                    LIB_FOUND_6=$?

    return $(expr ${LIB_FOUND_1} + ${LIB_FOUND_2} + ${LIB_FOUND_3} + ${LIB_FOUND_4} + ${LIB_FOUND_5} + ${LIB_FOUND_6})
}


################################################################################################################################
# Ensure target directories exist
################################################################################################################################

user_eval "mkdir -p '${INSTALL_DIR}/bin' '${INSTALL_DIR}/include' '${INSTALL_DIR}/lib' '${INSTALL_DIR}/share' '${INSTALL_DIR}/man'"


################################################################################################################################
# Install Boost
################################################################################################################################

if $WITH_BOOST ; then
    echo "Installing Boost..."
    BOOST_INSTALLED=false
    if [[ -n "$INSTALLED_BOOST" ]] ; then
        make_symlink $INSTALLED_BOOST "${INSTALL_DIR}/installed-boost"
        BOOST_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libboost-all-dev) ; then
                    BOOST_INSTALLED=true
                elif (${APT_GET} install libboost-dev || ${APT_GET} install libboost1.43-dev || ${APT_GET} install libboost1.42-dev || ${APT_GET} install libboost1.41-dev || ${APT_GET} install libboost1.40-dev || ${APT_GET} install libboost1.39-dev || ${APT_GET} install libboost1.38-dev) ; then
                    ${APT_GET} install libboost-filesystem-dev || ${APT_GET} install libboost-filesystem1.43-dev || ${APT_GET} install libboost-filesystem1.42-dev || ${APT_GET} install libboost-filesystem1.41-dev || ${APT_GET} install libboost-filesystem1.40-dev || ${APT_GET} install libboost-filesystem1.39-dev || ${APT_GET} install libboost-filesystem1.38-dev
                    ${APT_GET} install libboost-program-options-dev || ${APT_GET} install libboost-program-options1.43-dev || ${APT_GET} install libboost-program-options1.42-dev || ${APT_GET} install libboost-program-options1.41-dev || ${APT_GET} install libboost-program-options1.40-dev || ${APT_GET} install libboost-program-options1.39-dev || ${APT_GET} install libboost-program-options1.38-dev
                    ${APT_GET} install libboost-regex-dev || ${APT_GET} install libboost-regex1.43-dev || ${APT_GET} install libboost-regex1.42-dev || ${APT_GET} install libboost-regex1.41-dev || ${APT_GET} install libboost-regex1.40-dev || ${APT_GET} install libboost-regex1.39-dev || ${APT_GET} install libboost-regex1.38-dev
                    ${APT_GET} install libboost-system-dev || ${APT_GET} install libboost-system1.43-dev || ${APT_GET} install libboost-system1.42-dev || ${APT_GET} install libboost-system1.41-dev || ${APT_GET} install libboost-system1.40-dev || ${APT_GET} install libboost-system1.39-dev || ${APT_GET} install libboost-system1.38-dev
                    ${APT_GET} install libboost-thread-dev || ${APT_GET} install libboost-thread1.43-dev || ${APT_GET} install libboost-thread1.42-dev || ${APT_GET} install libboost-thread1.41-dev || ${APT_GET} install libboost-thread1.40-dev || ${APT_GET} install libboost-thread1.39-dev || ${APT_GET} install libboost-thread1.38-dev
                    BOOST_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $BOOST_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/Boost';
                   echo 'Decompressing archive...';
                   tar -xkzf boost_1_53_0.tar.gz;
                   cd boost_1_53_0;
                   ./bootstrap.sh --prefix='${INSTALL_DIR}';"
        BOOST_TOOLSET=`grep -o -m1 'using.\+' "${BASE_DIR}/Boost/boost_1_53_0/project-config.jam" | sed 's/using[ \t]\+\([^ \t]\+\).*/\1/'`
        user_eval "cd '${BASE_DIR}/Boost/boost_1_53_0';
                   if [[ '$BOOST_TOOLSET' = darwin ]] ; then
                       ./bjam $NUM_PROCS darwin/cxxflags=-fno-strict-aliasing install;
                   elif [[ '$BOOST_TOOLSET' = gcc ]] ; then
                       ./bjam $NUM_PROCS gcc/cxxflags='-fno-strict-aliasing -fPIC' install;
                   else
                       ./bjam $NUM_PROCS install;
                   fi;
                   cd '${BASE_DIR}'"
        # Boost bakes in non-absolute install names on OS X, we must replace them with absolute paths since we're installing to
        # a non-standard location
        if [[ $HOST_OS =~ .*arwin.* ]] ; then
            user_eval "cd '${INSTALL_DIR}/lib';
                       for f in libboost*.dylib;
                       do
                         if [[ ! -h \$f ]] ; then
                           echo 'Fixing install names of '\$f;
                           otool -L \$f | grep -m 1 'libboost.*[.]dylib[^:]' | grep -o 'libboost.*[.]dylib' | xargs -I filename install_name_tool -id '${INSTALL_DIR}/lib/'filename \$f;
                           otool -L \$f | grep 'libboost.*[.]dylib[^:]' | grep -o 'libboost.*[.]dylib' | xargs -I filename install_name_tool -change filename '${INSTALL_DIR}/lib/'filename \$f;
                         fi;
                       done;
                       cd '${BASE_DIR}'"
        fi
    fi
fi


################################################################################################################################
# Install G3D
################################################################################################################################

if $WITH_G3D ; then
    echo "Installing G3D..."
    if [[ -n "$INSTALLED_G3D" ]] ; then
        make_symlink $INSTALLED_G3D "${INSTALL_DIR}/installed-g3d"
    else
        user_eval "cd '${BASE_DIR}/G3D';
                   echo 'Decompressing archive...';
                   unzip -unq G3D-8.00-src.zip;
                   cd Patches/G3D-8.00;
                   FILES=\$(find . -name '*.patch');
                   for f in \$FILES;
                   do
                     patch -N -F 0 '${BASE_DIR}/G3D/G3D/'\${f%.patch} \$f;
                   done;
                   cd '${BASE_DIR}/G3D/G3D';
                   ./buildg3d --install '${INSTALL_DIR}' lib;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install FreeImagePlus
################################################################################################################################

if $WITH_FREEIMAGEPLUS ; then
    echo "Installing FreeImagePlus..."
    FREEIMAGEPLUS_INSTALLED=false
    if [[ -n "$INSTALLED_FREEIMAGEPLUS" ]] ; then
        make_symlink $INSTALLED_FREEIMAGEPLUS "${INSTALL_DIR}/installed-freeimageplus"
        FREEIMAGEPLUS_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libfreeimage-dev) ; then
                    ${APT_GET} install libfreeimageplus-dev || true  # needed since Ubuntu 16.04
                    FREEIMAGEPLUS_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $FREEIMAGEPLUS_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/FreeImage';
                   echo 'Decompressing archive...';
                   unzip -unq FreeImage3120.zip;
                   cd FreeImage;
                   export INCDIR='${INSTALL_DIR}/include';
                   export INSTALLDIR='${INSTALL_DIR}/lib';
                   cat Makefile.fip | tr -d '\r' \
                       | sed -e 's/-o root -g root//g' \
                             -e 's/\(install -m 755.*\)/& \&\& ranlib \$(INSTALLDIR)\/\$(STATICLIB) \&\& ln -sf \$(SHAREDLIB) \$(INSTALLDIR)\/\$(VERLIBNAME)/' \
                       > Makefile_noroot.fip;
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                       cat Makefile_noroot.fip | sed -e 's/[.]so/.dylib/g' -e 's/-shared/-dynamiclib/g' -e 's/-Wl[^ ]*//g' > Makefile_noroot_osx.fip;
                       make $NUM_PROCS -f Makefile_noroot_osx.fip;
                       make -f Makefile_noroot_osx.fip install;
                   else
                       make $NUM_PROCS -f Makefile_noroot.fip;
                       make -f Makefile_noroot.fip install;
                   fi;
                   cd '${BASE_DIR}'"
        # FreeImage bakes in non-absolute install names on OS X, we must replace them with absolute paths since we're installing
        # to a non-standard location
        if [[ $HOST_OS =~ .*arwin.* ]] ; then
            user_eval "cd '${INSTALL_DIR}/lib';
                       for f in libfreeimage*.dylib;
                       do
                         if [[ ! -h \$f ]] ; then
                           echo 'Fixing install names of '\$f;
                           otool -L \$f | grep -m 1 'libfreeimage.*[.]dylib[^:]' | grep -o 'libfreeimage.*[.]dylib' | xargs -I filename install_name_tool -id '${INSTALL_DIR}/lib/'filename \$f;
                           otool -L \$f | grep 'libfreeimage.*[.]dylib[^:]' | grep -o 'libfreeimage.*[.]dylib' | xargs -I filename install_name_tool -change filename '${INSTALL_DIR}/lib/'filename \$f;
                         fi;
                       done;
                       cd '${BASE_DIR}'"
        fi
    fi
fi


################################################################################################################################
# Install CLUTO
################################################################################################################################

if $WITH_CLUTO ; then
    echo "Installing CLUTO..."
    if [[ -n "$INSTALLED_CLUTO" ]] ; then
        make_symlink $INSTALLED_CLUTO "${INSTALL_DIR}/installed-cluto"
    else
        FULL_NAME=`uname -a`
        user_eval "cd '${BASE_DIR}/CLUTO';
                   echo 'Decompressing archive...';
                   unzip -unq cluto-2.1.2a.zip;
                   cd cluto-2.1.2;
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                       if [[ '$FULL_NAME' =~ .*i386.* ]] ; then
                         cp Darwin-i386/libcluto.a '${INSTALL_DIR}/lib/' ;
                       else
                         cp Darwin-ppc/libcluto.a '${INSTALL_DIR}/lib/' ;
                       fi;
                   elif [[ '$HOST_OS' =~ .*inux.* ]] ; then
                       if [[ '$FULL_NAME' =~ .*x86_64.* ]] ; then
                         cp Linux-x86_64/libcluto.a '${INSTALL_DIR}/lib/' ;
                       else
                         cp Linux-i686/libcluto.a '${INSTALL_DIR}/lib/' ;
                       fi;
                   else
                       echo 'Error: Cannot install CLUTO on this OS. Please try manual installation.';
                       exit;
                   fi;
                   cp cluto.h '${INSTALL_DIR}/include/' ;
                   chmod 644 '${INSTALL_DIR}/lib/libcluto.a' ;
                   chmod 644 '${INSTALL_DIR}/include/cluto.h' ;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install CGAL
################################################################################################################################

if $WITH_CGAL ; then
    echo "Installing CGAL..."
    if [[ -n "$INSTALLED_CGAL" ]] ; then
        make_symlink $INSTALLED_CGAL "${INSTALL_DIR}/installed-cgal"
    else
        if $USE_ROOT ; then
            case "$DISTRO" in
                debian )
                    ${APT_GET} install libgmp-dev libmpfr-dev
                    ;;
            esac
        fi

        user_eval "cd '${BASE_DIR}/CGAL';
                   echo 'Decompressing archive...';
                   unzip -unq CGAL-4.6.2.zip;
                   cd CGAL-4.6.2;
                   cmake -DCGAL_CXX_FLAGS='-DCGAL_CFG_NO_CPP0X_ARRAY -DCGAL_CFG_NO_TR1_ARRAY' -DCMAKE_PREFIX_PATH='${INSTALL_DIR}' -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' .;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'"
        # CGAL bakes in non-absolute install names on OS X, we must replace them with absolute paths since we're installing to a
        # non-standard location
        if [[ $HOST_OS =~ .*arwin.* ]] ; then
            user_eval "cd '${INSTALL_DIR}/lib';
                       for f in libCGAL*.dylib;
                       do
                         if [[ ! -h \$f ]] ; then
                           echo 'Fixing install names of '\$f;
                           otool -L \$f | grep -m 1 'libCGAL.*[.]dylib[^:]' | grep -o 'libCGAL.*[.]dylib' | xargs -I filename install_name_tool -id '${INSTALL_DIR}/lib/'filename \$f;
                           otool -L \$f | grep 'libCGAL.*[.]dylib[^:]' | grep -o 'libCGAL.*[.]dylib' | xargs -I filename install_name_tool -change filename '${INSTALL_DIR}/lib/'filename \$f;
                         fi;
                       done;
                       cd '${BASE_DIR}'"
        fi
    fi
fi


################################################################################################################################
# Install Eigen
################################################################################################################################

if $WITH_EIGEN ; then
    echo "Installing Eigen..."
    if [[ -n "$INSTALLED_EIGEN" ]] ; then
        make_symlink $INSTALLED_EIGEN "${INSTALL_DIR}/installed-eigen"
    else
        user_eval "cd '${BASE_DIR}/Eigen';
                   echo 'Decompressing archive...';
                   unzip -unq eigen-eigen-dc6cfdf9bcec.zip;
                   cd eigen-eigen-dc6cfdf9bcec;
                   echo 'Copying header files to ${INSTALL_DIR}/include/Eigen';
                   cp -R Eigen '${INSTALL_DIR}/include/';
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install OpenMesh
################################################################################################################################

if $WITH_OPENMESH ; then
    echo "Installing OpenMesh..."
    if [[ -n "$INSTALLED_OPENMESH" ]] ; then
        make_symlink $INSTALLED_OPENMESH "${INSTALL_DIR}/installed-openmesh"
    else
        user_eval "cd '${BASE_DIR}/OpenMesh';
                   echo 'Decompressing archive...';
                   unzip -unq OpenMesh-2.0-RC3.zip;
                   cd OpenMesh-2.0-RC3;
                   mkdir build;
                   cd build;
                   cmake -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' ..;
                   make $NUM_PROCS install;
                   cp -R ../src/OpenMesh '${INSTALL_DIR}/include/';
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install OSMesa
################################################################################################################################

if $WITH_OSMESA ; then
    echo "Installing OSMesa..."
    if [[ -n "$INSTALLED_OSMESA" ]] ; then
        make_symlink $INSTALLED_OSMESA "${INSTALL_DIR}/installed-osmesa"
    else
        user_eval "cd '${BASE_DIR}/Mesa';
                   echo 'Decompressing Mesa archive...';
                   tar -xkzf mesa-11.0.7.tar.gz;
                   cd mesa-11.0.7;
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                       ./configure --prefix='${INSTALL_DIR}';
                   else
                       ./configure --disable-driglx-direct --disable-dri --disable-glx --disable-glw --disable-egl \
                                   --disable-gallium-osmesa --disable-gallium-llvm --with-gallium-drivers=''
                                   --disable-gles1 --disable-gles2 --with-pic --enable-osmesa --prefix='${INSTALL_DIR}';
                   fi;
                   make $NUM_PROCS;
                   make install;
                   cd '${BASE_DIR}/Mesa';
                   echo 'Decompressing GLU archive...';
                   tar -xkzf glu-9.0.0.tar.gz;
                   cd glu-9.0.0;
                   PKG_CONFIG_PATH='${INSTALL_DIR}/lib/pkgconfig' ./configure --enable-osmesa --enable-shared --disable-static \
                                                                              --prefix='${INSTALL_DIR}';
                   make $NUM_PROCS;
                   make install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install Sirikata
################################################################################################################################

if $WITH_SIRIKATA ; then
    echo "Installing Sirikata..."
    if [[ -n "$INSTALLED_SIRIKATA" ]] ; then
        make_symlink $INSTALLED_SIRIKATA "${INSTALL_DIR}/installed-sirikata"
    else
        user_eval "cd '${BASE_DIR}/Sirikata';
                   if [[ -d sirikata ]] ; then
                       cd sirikata;
                       git pull;
                   else
                       git clone git://github.com/sirikata/sirikata.git;
                       cd sirikata;
                   fi;
                   make depends;
                   cd build/cmake;
                   cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' .;
                   make $NUM_PROCS;
                   make install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install s3fs
################################################################################################################################

if $WITH_S3FS ; then
    echo "Installing s3fs..."
    if [[ -n "$INSTALLED_S3FS" ]] ; then
        make_symlink $INSTALLED_S3FS "${INSTALL_DIR}/installed-s3fs"
    else
        user_eval "cd '${BASE_DIR}/s3fs';
                   echo 'Decompressing archive...';
                   tar -xkzf s3fs-r177-source.tar.gz;
                   cd s3fs;
                   cat Makefile | sed -e 's:/usr/bin:${INSTALL_DIR}/bin/': > Makefile_localinstall;
                   make $NUM_PROCS -f Makefile_localinstall install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install LAPACK
################################################################################################################################

if $WITH_LAPACK ; then
    echo "Installing LAPACK..."
    LAPACK_INSTALLED=false
    if [[ -n "$INSTALLED_LAPACK" ]] ; then
        make_symlink $INSTALLED_LAPACK "${INSTALL_DIR}/installed-lapack"
        LAPACK_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install liblapack-dev) ; then
                    LAPACK_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $LAPACK_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/LAPACK';
                   echo 'Decompressing archive...';
                   tar -xkzf lapack-3.6.1.tgz;
                   cd lapack-3.6.1;
                   cmake -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' -DBUILD_TESTING=OFF .;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install ARPACK
################################################################################################################################

if $WITH_ARPACK ; then
    echo "Installing ARPACK..."
    ARPACK_INSTALLED=false
    if [[ -n "$INSTALLED_ARPACK" ]] ; then
        make_symlink $INSTALLED_ARPACK "${INSTALL_DIR}/installed-arpack"
        ARPACK_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libarpack2-dev) ; then
                    ARPACK_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $ARPACK_INSTALLED ; then
        # The ARPACK patch file needs to overwrite the original files, so we don't use the -k option to tar
        user_eval "cd '${BASE_DIR}/ARPACK';
                   tar -xkvzf arpack96.tar.gz;
                   tar -xvzf patch.tar.gz;
                   patch -N -p 1 -d ARPACK < arpack96.patch.diff;
                   cd ARPACK;
                   cat ARmake.inc | sed -e 's:\$(HOME):${BASE_DIR}/ARPACK:' \
                                        -e 's:PLAT = SUN4:PLAT = ${HOST_OS}:' \
                                        -e 's:-cg89:-fPIC:' \
                                        -e 's:/bin/make:make:'
                       > ARmake.inc.tmp;
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                       cat ARmake.inc.tmp | sed -e 's/f77/gfortran/' > ARmake.inc;
                   else
                       cp ARmake.inc.tmp ARmake.inc;
                   fi;
                   make lib;
                   cp -v 'libarpack_${HOST_OS}.a' '${INSTALL_DIR}/lib/libarpack.a';
                   cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install ARPACK++
################################################################################################################################

if $WITH_ARPACKPP ; then
    echo "Installing ARPACK++..."
    if $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                ${APT_GET} install libsuperlu-dev || ${APT_GET} install libsuperlu3-dev
                ${APT_GET} install libsuitesparse-dev
                ;;
        esac
    fi

    if [[ -n "$INSTALLED_ARPACKPP" ]] ; then
        make_symlink $INSTALLED_ARPACKPP "${INSTALL_DIR}/installed-arpack++"
    else
        user_eval "cd '${BASE_DIR}/ARPACK++';
                   tar -xkvzf arpack++.tar.gz;
                   patch -N -p 1 -d arpack++ < arpack++1.2.patch.diff;
                   cd arpack++;
                   mkdir -p '${INSTALL_DIR}/include/arpack++';
                   cd include;
                   ls -1 . | grep '[.]h$' | xargs -I filename chmod 644 filename;
                   ls -1 . | grep '[.]h$' | xargs -I filename cp filename '${INSTALL_DIR}/include/arpack++/';
                   chmod 644 '${INSTALL_DIR}'/include/arpack++/*.h;
                   cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install PETSc
################################################################################################################################

if $WITH_PETSC ; then
    echo "Installing PETSc..."
    if [[ -n "$INSTALLED_PETSC" ]] ; then
        make_symlink $INSTALLED_PETSC "${INSTALL_DIR}/installed-petsc"
    else
        check_library "libblas"; BLAS_FOUND=$?
        check_library "libmpi";  MPI_FOUND=$?
        if [[ "${BLAS_FOUND}" != "0" ]] ; then
          echo "Detected an existing installation of BLAS on your system"
          BLAS_FLAGS="--with-blas=1"
        else
          BLAS_FLAGS="--download-c-blas-lapack=1"
        fi
        if [[ "${MPI_FOUND}" != "0" ]] ; then
          echo "Detected an existing installation of MPI on your system"
          MPI_FLAGS="--with-mpi=1"
        else
          MPI_FLAGS="--download-mpich=1"
        fi
        user_eval "cd '${BASE_DIR}/PETSc';
                   echo 'Decompressing archive...';
                   tar -xkzf petsc-3.0.0-p8.tar.gz;
                   cd petsc-3.0.0-p8;
                   export PETSC_DIR='${BASE_DIR}/PETSc/petsc-3.0.0-p8';
                   ./configure --with-pic --with-fortran=0 ${BLAS_FLAGS} ${MPI_FLAGS} --prefix='${INSTALL_DIR}';
                   cd '${BASE_DIR}'"
        # For the time being assume permissions are not going to be so screwy that the following commands will fail. Getting
        # command evaluation and variables working correctly (correct execution order, correct syntax etc) within user_eval is
        # tough.
        PETSC_CONFLOG_PATH=$(readlink -f "${BASE_DIR}/PETSc/petsc-3.0.0-p8/configure.log")
        PETSC_CONF_DIR=$(dirname ${PETSC_CONFLOG_PATH})
        PETSC_ARCH_DIR=$(dirname ${PETSC_CONF_DIR})
        PETSC_ARCH_NAME=$(basename ${PETSC_ARCH_DIR})
        user_eval "cd '${BASE_DIR}/PETSc/petsc-3.0.0-p8';
                   export PETSC_DIR='${BASE_DIR}/PETSc/petsc-3.0.0-p8';
                   export PETSC_ARCH='${PETSC_ARCH_NAME}';
                   make $NUM_PROCS all;
                   make install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install SLEPc
################################################################################################################################

if $WITH_SLEPC ; then
    echo "Installing SLEPc..."
    if [[ -n "$INSTALLED_SLEPC" ]] ; then
        make_symlink $INSTALLED_SLEPC "${INSTALL_DIR}/installed-slepc"
    else
        if [[ -e "${INSTALL_DIR}/include/petscconf.h" ]] ; then
          echo "Detected existing PETSc installation in ${INSTALL_DIR}"
          PETSC_DIR="${INSTALL_DIR}"
          PETSC_ARCH_LINE=$(grep 'PETSC_ARCH_NAME *=' "${INSTALL_DIR}/conf/petscvariables")
          PETSC_ARCH=${PETSC_ARCH_LINE/PETSC_ARCH_NAME *=/}  # remove variable assignment
          PETSC_ARCH=${PETSC_ARCH//[ \t]/}  # remove any remaining spaces
        fi
        if [[ -z ${PETSC_DIR:-} ]] ; then
          echo "Before installing SLEPc, please set PETSC_DIR to the installation prefix of PETSc"
          exit
        fi
        if [[ -z ${PETSC_ARCH:-} ]] ; then
          echo "Before installing SLEPc, please set PETSC_ARCH to the installation architecture of PETSc (e.g. linux-gnu-c-debug)"
          exit
        fi
        PETSC_ARCH_BACKUP=${PETSC_ARCH}  # SLEPc configure should not have PETSC_ARCH set, just like PETSC configure. It will be
                                         # restored for the actual make step
        user_eval "cd '${BASE_DIR}/SLEPc';
                   echo 'Decompressing archive...';
                   tar -xkzf slepc-3.0.0-p6.tgz;
                   cd slepc-3.0.0-p6;
                   export PETSC_DIR;
                   unset PETSC_ARCH;
                   export SLEPC_DIR='${BASE_DIR}/SLEPc/slepc-3.0.0-p6';
                   ./configure --prefix='${INSTALL_DIR}';
                   export PETSC_ARCH='${PETSC_ARCH_BACKUP}';
                   CFLAGS='-fPIC' make $NUM_PROCS all;
                   make install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install lp_solve
################################################################################################################################

if $WITH_LPSOLVE ; then
    echo "Installing lp_solve..."
    LPSOLVE_INSTALLED=false
    if [[ -n "$INSTALLED_LPSOLVE" ]] ; then
        make_symlink $INSTALLED_LPSOLVE "${INSTALL_DIR}/installed-lp_solve"
        LPSOLVE_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install liblpsolve55-dev) ; then
                    LPSOLVE_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $LPSOLVE_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/lp_solve';
                  tar -xkzf lp_solve_5.5.0.15_source.tar.gz;
                  cd lp_solve_5.5/lpsolve55;
                  if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                      sh ccc.osx;
                  else
                      sh ccc;
                  fi;
                  cd bin;
                  if [[ -d ux64 ]] ; then
                      cd ux64;
                  else
                      cd ux32;
                  fi;
                  ls -1 | grep liblpsolve55 | xargs -I filename cp filename '${INSTALL_DIR}/lib/';
                  cd ../../..;
                  mkdir -p '${INSTALL_DIR}/include/lpsolve';
                  ls -1 | grep [.]h$ | xargs -I filename cp filename '${INSTALL_DIR}/include/lpsolve/';
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install FFTW
################################################################################################################################

if $WITH_FFTW ; then
    echo "Installing fftw..."
    if [[ -n "$INSTALLED_FFTW" ]] ; then
        make_symlink $INSTALLED_FFTW "${INSTALL_DIR}/installed-fftw"
    else
        user_eval "cd '${BASE_DIR}/FFTW';
                  tar -xkzf fftw-3.2.2.tar.gz;
                  cd fftw-3.2.2;
                  ./configure --prefix='${INSTALL_DIR}';
                  make $NUM_PROCS install;
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install OPT++
################################################################################################################################

if $WITH_OPTPP ; then
    echo "Installing OPT++..."
    if [[ -n "$INSTALLED_OPTPP" ]] ; then
        make_symlink $INSTALLED_OPTPP "${INSTALL_DIR}/installed-opt++"
    else
        user_eval "cd '${BASE_DIR}/OPT++';
                  tar -xkzf optpp-2.4.tar.gz;
                  cd optpp-2.4;
                  patch -N -F 0 newmat11/precisio.h ../Patches/newmat11/precisio.h.patch;
                  ./configure --prefix='${INSTALL_DIR}' --includedir='${INSTALL_DIR}'/include/OPT++ --with-pic;
                  make $NUM_PROCS install;
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install lib3ds
################################################################################################################################

if $WITH_LIB3DS ; then
    echo "Installing lib3ds..."
    LIB3DS_INSTALLED=false
    if [[ -n "$INSTALLED_LIB3DS" ]] ; then
        make_symlink $INSTALLED_LIB3DS "${INSTALL_DIR}/installed-lib3ds"
        LIB3DS_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install lib3ds-dev) ; then
                    LIB3DS_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $LIB3DS_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/lib3ds';
                  unzip -unq lib3ds-1.3.0.zip;
                  cd lib3ds-1.3.0;
                  ./configure --prefix='${INSTALL_DIR}';
                  make $NUM_PROCS install;
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install GLFW
################################################################################################################################

if $WITH_GLFW ; then
    echo "Installing GLFW..."
    GLFW_INSTALLED=false
    if [[ -n "$INSTALLED_GLFW" ]] ; then
        make_symlink $INSTALLED_GLFW "${INSTALL_DIR}/installed-glfw"
        GLFW_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libglfw-dev) ; then
                    GLFW_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $GLFW_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/GLFW';
                   echo 'Decompressing archive...';
                   unzip -unq glfw-3.2.zip;
                   cd glfw-3.2;
                   cmake -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install ANN
################################################################################################################################

if $WITH_ANN ; then
    echo "Installing ANN..."
    if [[ -n "$INSTALLED_ANN" ]] ; then
        make_symlink $INSTALLED_ANN "${INSTALL_DIR}/installed-ann"
    else
        user_eval "cd '${BASE_DIR}/ANN';
                  unzip -unq ann_1.1.2.zip;
                  cd ann_1.1.2;
                  if $ANN_FLOAT; then
                     patch -N -F 0 include/ANN/ANN.h ../Patches/ann_1.1.2/ANN.h.float.patch;
                     patch -N -F 0 test/rand.cpp ../Patches/ann_1.1.2/rand.cpp.float.patch;
                  fi;
                  if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                    make macosx-g++ $NUM_PROCS;
                  else
                    make linux-g++ $NUM_PROCS;
                  fi;
                  cp -R include/* '${INSTALL_DIR}/include/';
                  cp lib/* '${INSTALL_DIR}/lib/';
                  cp bin/* '${INSTALL_DIR}/bin/';
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install PNL
################################################################################################################################

if $WITH_PNL ; then
    echo "Installing PNL..."
    if [[ -n "$INSTALLED_PNL" ]] ; then
        make_symlink $INSTALLED_PNL "${INSTALL_DIR}/installed-pnl"
    else
        # Because configure.gcc unpacks its own set of Makefile.in's from conf.tbz which overwrite the original Makefile.in's,
        # the two Makefile.in's must be patched AFTER the configure step.
        user_eval "cd '${BASE_DIR}/PNL';
                   echo 'Decompressing archive...';
                   unzip -unq PNL_Release_1_0_linux.zip;
                   cd PNL;
                   patch -N -F 0 high/include/ModelEngine.hpp ../Patches/high/include/ModelEngine.hpp.patch;
                   patch -N -F 0 high/include/WInner.hpp ../Patches/high/include/WInner.hpp.patch;
                   patch -N -F 0 cxcore/cxcore/src/cxswitcher.cpp ../Patches/cxcore/cxcore/src/cxswitcher.cpp.patch;
                   patch -N -F 0 c_pgmtk/src/include/cart/datadefs.h ../Patches/c_pgmtk/src/include/cart/datadefs.h.patch;
                   patch -N -F 0 c_pgmtk/src/include/cart/inlines.h ../Patches/c_pgmtk/src/include/cart/inlines.h.patch;
                   patch -N -F 0 c_pgmtk/src/include/pnlCondSoftMaxDistribFun.hpp ../Patches/c_pgmtk/src/include/pnlCondSoftMaxDistribFun.hpp.patch;
                   patch -N -F 0 c_pgmtk/src/include/pnlGaussianDistribFun.hpp ../Patches/c_pgmtk/src/include/pnlGaussianDistribFun.hpp.patch;
                   patch -N -F 0 c_pgmtk/src/include/pnlTabularDistribFun.hpp ../Patches/c_pgmtk/src/include/pnlTabularDistribFun.hpp.patch;
                   patch -N -F 0 c_pgmtk/src/include/pnlTreeDistribFun.hpp ../Patches/c_pgmtk/src/include/pnlTreeDistribFun.hpp.patch;
                   patch -N -F 0 c_pgmtk/src/cvcart.cpp ../Patches/c_pgmtk/src/cvcart.cpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlAllocator.hpp ../Patches/c_pgmtk/include/pnlAllocator.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlBKInferenceEngine.hpp ../Patches/c_pgmtk/include/pnlBKInferenceEngine.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlEmLearningEngineDBN.hpp ../Patches/c_pgmtk/include/pnlEmLearningEngineDBN.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlGraph.hpp ../Patches/c_pgmtk/include/pnlGraph.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlSamplingInferenceEngine.hpp ../Patches/c_pgmtk/include/pnlSamplingInferenceEngine.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlSoftMaxCPD.hpp ../Patches/c_pgmtk/include/pnlSoftMaxCPD.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlStackMemProv.hpp ../Patches/c_pgmtk/include/pnlStackMemProv.hpp.patch;
                   patch -N -F 0 c_pgmtk/include/pnlString.hpp ../Patches/c_pgmtk/include/pnlString.hpp.patch;
                   patch -N -F 0 configure.gcc ../Patches/configure.gcc.patch;
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                     patch -N -F 0 high/source/Makefile.am ../Patches/high/source/Makefile.am.patch.osx;
                     patch -N -F 0 c_pgmtk/src/Makefile.am ../Patches/c_pgmtk/src/Makefile.am.patch.osx;
                   fi;
                   chmod 755 configure.gcc;
                   chmod 755 install-sh;
                   ./configure.gcc --prefix='${INSTALL_DIR}';
                   if [[ '$HOST_OS' =~ .*arwin.* ]] ; then
                     patch -N -F 0 high/source/Makefile.in ../Patches/high/source/Makefile.in.patch.osx;
                     patch -N -F 0 c_pgmtk/src/Makefile.in ../Patches/c_pgmtk/src/Makefile.in.patch.osx;
                   fi;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install Qt4
################################################################################################################################

if $WITH_QT4 ; then
    echo "Installing Qt4..."
    QT4_INSTALLED=false
    if [[ -n "$INSTALLED_QT4" ]] ; then
        make_symlink $INSTALLED_QT4 "${INSTALL_DIR}/installed-qt4"
        QT4_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libqt4-dev) ; then
                    QT4_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $QT4_INSTALLED ; then
        echo "Qt4 installation failed"
        exit
    fi
fi


################################################################################################################################
# Install wxWidgets
################################################################################################################################

if $WITH_WXWIDGETS ; then
    echo "Installing wxWidgets..."
    WXWIDGETS_INSTALLED=false
    if [[ -n "$INSTALLED_WXWIDGETS" ]] ; then
        make_symlink $INSTALLED_WXWIDGETS "${INSTALL_DIR}/installed-wxwidgets"
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install libwxgtk3.0-dev) ; then
                    WXWIDGETS_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $WXWIDGETS_INSTALLED ; then
        WXWIDGETS_CONFIG_ARGS="--without-subdirs --disable-debug"

        if [[ "$HOST_OS" =~ .*arwin.* ]] ; then
          WXWIDGETS_OSX_VERSION=`sysctl -n kern.osrelease | grep -oE '^[0-9]+'`

					# wxWidgets has OS X 10.7 compatibility by default, forcing linking to stdc++ instead of libc++ (the default from 10.9
					# onwards). This causes problems when linking to other code built using default options on 10.9 or later.
          if [[ "WXWIDGETS_OSX_VERSION" -ge 13 ]] ; then
            WXWIDGETS_CONFIG_ARGS="${WXWIDGETS_CONFIG_ARGS} --with-macosx-version-min=10.9 --disable-xrc"
          fi

          # On macOS Sierra, wxMediaCtrl is broken since it tries to include the absent QTKit header
          if [[ "WXWIDGETS_OSX_VERSION" -ge 16 ]] ; then
            WXWIDGETS_CONFIG_ARGS="${WXWIDGETS_CONFIG_ARGS} --disable-mediactrl"
          fi
        fi

        user_eval "cd '${BASE_DIR}/wxWidgets';
                  tar -xkjf wxWidgets-3.1.0.tar.bz2;
                  cd wxWidgets-3.1.0;
                  ./configure --prefix='${INSTALL_DIR}' ${WXWIDGETS_CONFIG_ARGS};
                  make $NUM_PROCS install;
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install NanoGUI (from local snapshot)
################################################################################################################################

if $WITH_NANOGUI ; then
    echo "Installing NanoGUI (from local snapshot)..."
    if [[ -n "$INSTALLED_NANOGUI" ]] ; then
        make_symlink $INSTALLED_NANOGUI "${INSTALL_DIR}/installed-nanogui"
    else
        if $USE_ROOT ; then
            case "$DISTRO" in
                debian )
                    ${APT_GET} install xorg-dev libglu1-mesa-dev python-dev
                    ;;
            esac
        fi

        user_eval "cd '${BASE_DIR}/NanoGUI';
                   echo 'Decompressing archive...';
                   unzip -unq nanogui_2016_08_11.zip;
                   cd nanogui;
                   cmake -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' -DNANOGUI_BUILD_EXAMPLE=OFF -DNANOGUI_BUILD_PYTHON=OFF -DNANOGUI_USE_GLAD=OFF .;
                   make $NUM_PROCS install;
                   cp ext/nanovg/src/nanovg.h '${INSTALL_DIR}/include/';
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install NanoGUI (latest version from GitHub)
################################################################################################################################

if $WITH_NANOGUI_LATEST ; then
    echo "Installing NanoGUI (latest version)..."

    if $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                ${APT_GET} install xorg-dev libglu1-mesa-dev python-dev
                ;;
        esac
    fi

    user_eval "cd '${BASE_DIR}/NanoGUI';
               echo 'Fetching latest version from GitHub...';
               mkdir -p 'latest';
               cd latest;
               git clone --recursive https://github.com/wjakob/nanogui;
               cd nanogui;
               cmake -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' -DNANOGUI_BUILD_EXAMPLE=OFF -DNANOGUI_BUILD_PYTHON=OFF -DNANOGUI_USE_GLAD=OFF .;
               make $NUM_PROCS install;
               cp ext/nanovg/src/nanovg.h '${INSTALL_DIR}/include/';
               cd '${BASE_DIR}'"
fi


################################################################################################################################
# Install Thea
################################################################################################################################

if $WITH_THEA ; then
    echo "Installing Thea..."
    if [[ -n "$INSTALLED_THEA" ]] ; then
        make_symlink $INSTALLED_THEA "${INSTALL_DIR}/installed-thea"
    else
        user_eval "cd '${BASE_DIR}/Thea';
                   cd Code/Build;
                   cmake -DTHEA_INSTALLATIONS_ROOT='${INSTALL_DIR}' -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' .;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install SNL
################################################################################################################################

if $WITH_SNL ; then
    echo "Installing SNL..."
    if [[ -n "$INSTALLED_SNL" ]] ; then
        make_symlink $INSTALLED_SNL "${INSTALL_DIR}/installed-snl"
    else
        if $SNL_INSTALL_DEPS ; then
            echo "Installing SNL dependencies to $INSTALL_DIR"
            cd "$BASE_DIR/SNL/Code/Dependencies"
            if $USE_ROOT ; then
                ./install-defaults --prefix "$INSTALL_DIR" $NUM_PROCS --use-root --user $USER
            else
                ./install-defaults --prefix "$INSTALL_DIR" $NUM_PROCS --user $USER
            fi
            cd "$BASE_DIR"
        fi

        user_eval "cd '${BASE_DIR}/SNL';
                   cd Code/Build;
                   cmake -DSNL_INSTALLATIONS_ROOT='${INSTALL_DIR}' -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' .;
                   make $NUM_PROCS install;
                   cd '${BASE_DIR}'"
    fi
fi


################################################################################################################################
# Install Tango icons
################################################################################################################################

if $WITH_ICONS_TANGO ; then
    echo "Installing Tango icons..."
    ICONS_TANGO_INSTALLED=false
    if [[ -n "$INSTALLED_ICONS_TANGO" ]] ; then
        make_symlink $INSTALLED_ICONS_TANGO "${INSTALL_DIR}/installed-icons-tango"
        ICONS_TANGO_INSTALLED=true
    elif $USE_ROOT ; then
        case "$DISTRO" in
            debian )
                if (${APT_GET} install tango-icon-theme) ; then
                    ICONS_TANGO_INSTALLED=true
                fi
                ;;
        esac
    fi

    if ! $ICONS_TANGO_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/Icons/Tango';
                  tar -xkzf tango-icon-theme-0.8.90.tar.gz;
                  cd tango-icon-theme-0.8.90;
                  mkdir -p '${INSTALL_DIR}/share/icons/Tango';
                  cp -R 16x16 '${INSTALL_DIR}/share/icons/Tango/';
                  cp -R 22x22 '${INSTALL_DIR}/share/icons/Tango/';
                  cp -R 32x32 '${INSTALL_DIR}/share/icons/Tango/';
                  cp -R scalable '${INSTALL_DIR}/share/icons/Tango/';
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Install Tango Art Libre icons
################################################################################################################################

if $WITH_ICONS_TANGO_ART_LIBRE ; then
    echo "Installing Tango Art Libre icons..."
    ICONS_TANGO_ART_LIBRE_INSTALLED=false
    if [[ -n "$INSTALLED_ICONS_TANGO_ART_LIBRE" ]] ; then
        make_symlink $INSTALLED_ICONS_TANGO_ART_LIBRE "${INSTALL_DIR}/installed-icons-tango-art-libre"
        ICONS_TANGO_ART_LIBRE_INSTALLED=true
    fi

    if ! $ICONS_TANGO_ART_LIBRE_INSTALLED ; then
        user_eval "cd '${BASE_DIR}/Icons/TangoArtLibre';
                  unzip -unq tango-art-libre.zip;
                  cd tango-art-libre;
                  mkdir -p '${INSTALL_DIR}/share/icons/TangoArtLibre';
                  cp -R 16x16 '${INSTALL_DIR}/share/icons/TangoArtLibre/';
                  cp -R 22x22 '${INSTALL_DIR}/share/icons/TangoArtLibre/';
                  cp -R scalable '${INSTALL_DIR}/share/icons/TangoArtLibre/';
                  cd '${BASE_DIR}'";
    fi
fi


################################################################################################################################
# Finish
################################################################################################################################

echo "Installation complete"
