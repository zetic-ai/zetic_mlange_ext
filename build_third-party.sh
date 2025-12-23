#!/bin/bash

set -e

OPENCV_VERSION="4.12.0"
THIRD_PARTY_DIR="${THIRD_PARTY_PATH:-$(pwd)/third-party}"
OPENCV_ROOT="$THIRD_PARTY_DIR/opencv"
PLATFORM=${1:-"all"}  # all, android, ios, ios-simulator

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Function to check system resources
check_system_resources() {
    log_info "System resources:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Memory: $(vm_stat | head -5)"
        echo "  CPU cores: $(sysctl -n hw.ncpu)"
        echo "  Disk space: $(df -h . | tail -1)"
    else
        echo "  Memory: $(free -h | head -2)"
        echo "  CPU cores: $(nproc)"
        echo "  Disk space: $(df -h . | tail -1)"
    fi
}

# Function to monitor build progress
monitor_build() {
    local build_pid=$1
    local build_dir=$2
    local log_file="$build_dir/build.log"

    while kill -0 "$build_pid" 2>/dev/null; do
        if [[ -f "$log_file" ]]; then
            tail -n 1 "$log_file" 2>/dev/null | grep -o '\[[0-9]*%\]' | tail -1
        fi
        sleep 10
    done
}

clone_opencv() {
    log_info "Cloning OpenCV $OPENCV_VERSION..."

    if [ -d "$OPENCV_ROOT" ]; then
        log_warn "OpenCV directory exists. Removing..."
        rm -rf "$OPENCV_ROOT"
    fi

    mkdir -p "$THIRD_PARTY_DIR"
    cd "$THIRD_PARTY_DIR"

    # Clone with progress and retry logic
    local retry_count=0
    local max_retries=3

    while [ $retry_count -lt $max_retries ]; do
        if git clone --depth 1 --branch "$OPENCV_VERSION" --progress https://github.com/opencv/opencv.git; then
            break
        else
            retry_count=$((retry_count + 1))
            log_warn "Clone attempt $retry_count failed. Retrying..."
            sleep 5
        fi
    done

    if [ $retry_count -eq $max_retries ]; then
        log_error "Failed to clone OpenCV after $max_retries attempts"
        exit 1
    fi

    cd opencv
    log_info "OpenCV cloned successfully"
}

build_ios() {
    local target_platform=$1  # OS64 or SIMULATOR64
    local build_suffix=$2     # arm64 or x86_64

    log_info "Building OpenCV for iOS ($build_suffix)..."
    check_system_resources

    cd "$OPENCV_ROOT"

    BUILD_DIR="ios_build_$build_suffix"
    INSTALL_DIR="$OPENCV_ROOT/ios_install_$build_suffix"
    LOG_FILE="$BUILD_DIR/build.log"

    rm -rf $BUILD_DIR $INSTALL_DIR
    mkdir -p $BUILD_DIR

    # Download iOS toolchain with retry
    mkdir -p platforms/ios
    if [ ! -f "platforms/ios/ios.toolchain.cmake" ]; then
        log_info "Downloading iOS toolchain..."
        local retry_count=0
        while [ $retry_count -lt 3 ]; do
            if curl -L --connect-timeout 30 --max-time 300 "https://raw.githubusercontent.com/leetal/ios-cmake/master/ios.toolchain.cmake" -o "platforms/ios/ios.toolchain.cmake"; then
                break
            else
                retry_count=$((retry_count + 1))
                log_warn "Download attempt $retry_count failed. Retrying..."
                sleep 5
            fi
        done
    fi

    cd $BUILD_DIR

    # Configure with better error handling
    log_info "Configuring CMake for iOS ($build_suffix)..."
    cmake "$OPENCV_ROOT" \
        -G Xcode \
        -DCMAKE_TOOLCHAIN_FILE="$OPENCV_ROOT/platforms/ios/ios.toolchain.cmake" \
        -DPLATFORM=$target_platform \
        -DARCHS=$build_suffix \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DDEPLOYMENT_TARGET=12.0 \
        -DENABLE_BITCODE=OFF \
        -DENABLE_ARC=OFF \
        -DENABLE_VISIBILITY=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_C_FLAGS="-Os" \
        -DCMAKE_CXX_FLAGS="-Os -fvisibility=default -fno-lto" \
        -DCMAKE_CXX_VISIBILITY_PRESET=default \
        -DCMAKE_VISIBILITY_INLINES_HIDDEN=0 \
        -DBUILD_opencv_world=ON \
        -DBUILD_ZLIB=ON \
        -DOPENCV_MODULE_TYPE=STATIC \
        -DBUILD_opencv_core=ON \
        -DBUILD_opencv_imgproc=ON \
        -DBUILD_opencv_imgcodecs=OFF \
        -DWITH_ITT=OFF \
        -DWITH_IPP=OFF \
        -DWITH_IPP_A=OFF \
        -DBUILD_ITT=OFF \
        -DWITH_1394=OFF \
        -DWITH_AVFOUNDATION=OFF \
        -DWITH_CAP_IOS=OFF \
        -DWITH_CAROTENE=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_CUFFT=OFF \
        -DWITH_EIGEN=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_JPEG=ON \
        -DWITH_LAPACK=OFF \
        -DWITH_MATLAB=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCLAMDBLAS=OFF \
        -DWITH_OPENCLAMDFFT=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_PNG=ON \
        -DWITH_PROTOBUF=OFF \
        -DWITH_PTHREADS_PF=OFF \
        -DWITH_QUICKTIME=OFF \
        -DWITH_TBB=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_V4L=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_WIN32UI=OFF \
        -DBUILD_ANDROID_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_PACKAGE=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_WITH_DEBUG_INFO=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_APPS=OFF \
        -DINSTALL_TESTS=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_ANDROID_EXAMPLES=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_python=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DAPPLE_FRAMEWORK=OFF \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
        -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO

    log_info "Building OpenCV for iOS ($build_suffix)... This may take a while."

    xcodebuild -project 'OpenCV.xcodeproj' -configuration Release -jobs 8
    xcodebuild -project 'OpenCV.xcodeproj' -configuration Release -target install

    # Verify build success
    if [ -d "$INSTALL_DIR" ] && [ -n "$(find "$INSTALL_DIR" -name "*.a" 2>/dev/null)" ]; then
        log_info "iOS ($build_suffix) build completed successfully"
        log_info "Libraries created: $(find "$INSTALL_DIR" -name "*.a" | wc -l)"
    else
        log_error "iOS ($build_suffix) build failed - no libraries found"
        log_error "Check build log: $LOG_FILE"
        exit 1
    fi
}

build_android() {
    local abi=$1  # arm64-v8a, armeabi-v7a, x86, x86_64

    log_info "Building OpenCV for Android ($abi)..."
    check_system_resources

    cd "$OPENCV_ROOT"

    if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$ANDROID_NDK" ]; then
        log_error "ANDROID_NDK_HOME or ANDROID_NDK environment variable is not set"
        exit 1
    fi

    NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"

    if [ ! -d "$NDK_PATH" ]; then
        log_error "Android NDK not found at: $NDK_PATH"
        exit 1
    fi

    BUILD_DIR="android_build_$abi"
    INSTALL_DIR="$OPENCV_ROOT/android_install_$abi"
    LOG_FILE="$BUILD_DIR/build.log"

    rm -rf $BUILD_DIR $INSTALL_DIR
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR

    log_info "Configuring CMake for Android ($abi)..."
    cmake "$OPENCV_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=$abi \
        -DANDROID_PLATFORM=android-21 \
        -DANDROID_STL=c++_shared \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_opencv_world=OFF \
        -DBUILD_opencv_java=ON \
        -DBUILD_opencv_core=ON \
        -DBUILD_opencv_imgproc=ON \
        -DBUILD_opencv_imgcodecs=ON \
        -DBUILD_opencv_videoio=ON \
        -DBUILD_opencv_video=ON \
        -DBUILD_opencv_highgui=ON \
        -DINSTALL_CREATE_DISTRIB=ON \
        -DWITH_QT=OFF \
        -DWITH_OPENEXR=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_ANDROID_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_ITT=OFF \
        -DWITH_EIGEN=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_JPEG=ON \
        -DWITH_PNG=ON \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF

    make -j8
    make install

    # Verify build success
    if [ -d "$INSTALL_DIR" ] && [ -n "$(find "$INSTALL_DIR" -name "*.so" -o -name "*.a" 2>/dev/null)" ]; then
        log_info "Android ($abi) build completed successfully"
    else
        log_error "Android ($abi) build failed"
        exit 1
    fi
}

create_android_sdk() {
    log_info "Creating Android SDK structure..."

    cd "$OPENCV_ROOT"

    SDK_DIR="android/sdk"
    rm -rf android
    mkdir -p "$SDK_DIR/native/jni"
    mkdir -p "$SDK_DIR/native/libs"

    for abi in arm64-v8a armeabi-v7a x86 x86_64 ; do
        if [ -d "android_build_$abi" ]; then
            mkdir -p "$SDK_DIR/native/libs/$abi"
            find "android_install_$abi/sdk/native/libs/$abi" -name "*.so" -exec cp {} "$SDK_DIR/native/libs/$abi/" \; 2>/dev/null || true
            find "android_install_$abi/sdk/native/libs/$abi" -name "*.a" -exec cp {} "$SDK_DIR/native/libs/$abi/" \; 2>/dev/null || true

            if [ ! -d "$SDK_DIR/native/jni/include" ]; then
                cp -r "android_install_$abi/sdk/native/jni/include" "$SDK_DIR/native/jni/"
            fi
        fi
    done

    cat > "$SDK_DIR/native/jni/OpenCVConfig.cmake" << EOF
# OpenCV Android Configuration
set(OpenCV_VERSION "$OPENCV_VERSION")
set(OpenCV_ANDROID_SDK_ROOT "\${CMAKE_CURRENT_LIST_DIR}/../..")

# Include directories
set(OpenCV_INCLUDE_DIRS
    "\${CMAKE_CURRENT_LIST_DIR}/include"
    "\${CMAKE_CURRENT_LIST_DIR}/include/opencv4"
)

# Find libraries based on Android ABI
set(OpenCV_LIBS_DIR "\${OpenCV_ANDROID_SDK_ROOT}/native/libs/\${ANDROID_ABI}")

# Define libraries
set(OpenCV_LIBS
    "\${OpenCV_LIBS_DIR}/libopencv_java4.so"
)

# Filter existing libraries
set(OpenCV_LIBS_FILTERED)
foreach(lib \${OpenCV_LIBS})
    if(EXISTS "\${lib}")
        list(APPEND OpenCV_LIBS_FILTERED "\${lib}")
    endif()
endforeach()
set(OpenCV_LIBS \${OpenCV_LIBS_FILTERED})
EOF

    log_info "Android SDK created successfully"
}

create_ios_universal() {
    log_info "Creating iOS universal libraries..."

    cd "$OPENCV_ROOT"

    UNIVERSAL_DIR="ios_install"
    rm -rf "$UNIVERSAL_DIR"
    mkdir -p "$UNIVERSAL_DIR/lib"

    if [ -d "ios_install_arm64/include" ]; then
        cp -r "ios_install_arm64/include" "$UNIVERSAL_DIR/"
    fi

    if [ -d "ios_install_arm64/lib" ]; then
        ZLIB_LIB="ios_install_arm64/lib/opencv4/3rdparty/libzlib.a"
        if [ -f "$ZLIB_LIB" ]; then
            mkdir -p "$UNIVERSAL_DIR/lib/opencv4/3rdparty"
            cp "$ZLIB_LIB" "$UNIVERSAL_DIR/lib/opencv4/3rdparty/"
        fi

        for lib in ios_install_arm64/lib/*.a; do
            if [ -f "$lib" ]; then
                lib_name=$(basename "$lib")

                lipo_args=()

                if [ -f "ios_install_arm64/lib/$lib_name" ]; then
                    lipo_args+=("-arch" "arm64" "ios_install_arm64/lib/$lib_name")
                fi

                if [ -f "ios_install_x86_64/lib/$lib_name" ]; then
                    lipo_args+=("-arch" "x86_64" "ios_install_x86_64/lib/$lib_name")
                fi

                if [ ${#lipo_args[@]} -gt 0 ]; then
                    log_info "Creating universal library: $lib_name"
                    lipo "${lipo_args[@]}" -create -output "$UNIVERSAL_DIR/lib/$lib_name"
                fi
            fi
        done
    fi

    mkdir -p "$UNIVERSAL_DIR/lib/cmake/opencv4"
    cat > "$UNIVERSAL_DIR/lib/cmake/opencv4/OpenCVConfig.cmake" << EOF
# OpenCV iOS Configuration
set(OpenCV_VERSION "$OPENCV_VERSION")
set(OpenCV_INSTALL_PATH "\${CMAKE_CURRENT_LIST_DIR}/../../..")

# Include directories
set(OpenCV_INCLUDE_DIRS
    "\${OpenCV_INSTALL_PATH}/include"
    "\${OpenCV_INSTALL_PATH}/include/opencv4"
)

# Libraries
file(GLOB OpenCV_LIBS "\${OpenCV_INSTALL_PATH}/lib/*.a")

# Required frameworks
set(OpenCV_FRAMEWORKS
    "-framework Foundation"
    "-framework Accelerate"
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework CoreImage"
    "-framework QuartzCore"
)
EOF

    log_info "iOS universal libraries created successfully"
    log_info "Universal libraries: $(find "$UNIVERSAL_DIR/lib" -name "*.a" | wc -l)"
}

cleanup_on_error() {
    log_error "Build interrupted or failed. Cleaning up..."
    pkill -f xcodebuild 2>/dev/null || true
    pkill -f make 2>/dev/null || true
    exit 1
}

# Set up signal handlers
trap cleanup_on_error INT TERM

main() {
    log_info "Starting OpenCV $OPENCV_VERSION build process..."
    log_info "Platform: $PLATFORM"
    log_info "Third party directory: $THIRD_PARTY_DIR"

    check_system_resources

    clone_opencv

    case $PLATFORM in
        "ios")
            build_ios "OS64" "arm64"
            create_ios_universal
            ;;
        "ios-simulator")
            build_ios "SIMULATOR64" "x86_64"
            ;;
        "ios-all")
            build_ios "OS64" "arm64"
            build_ios "SIMULATOR64" "x86_64"
            create_ios_universal
            ;;
        "android")
            build_android "arm64-v8a"
            build_android "armeabi-v7a"
            create_android_sdk
            ;;
        "all")
            if command -v ndk-build &> /dev/null || [ -n "$ANDROID_NDK_HOME" ] || [ -n "$ANDROID_NDK" ]; then
                build_android "arm64-v8a"
                build_android "armeabi-v7a"
                build_android "x86"
                build_android "x86_64"
                create_android_sdk
            else
                log_warn "Android NDK not found, skipping Android build"
            fi

            if [[ "$OSTYPE" == "darwin"* ]]; then
                build_ios "OS64" "arm64"
                build_ios "SIMULATOR64" "x86_64"
                create_ios_universal
            else
                log_warn "Not on macOS, skipping iOS build"
            fi
            ;;
        *)
            log_error "Unknown platform: $PLATFORM"
            log_info "Usage: $0 [all|android|ios|ios-simulator|ios-all]"
            exit 1
            ;;
    esac

    log_info "OpenCV build process completed successfully!"
    log_info "Installation directories:"
    if [ -d "$OPENCV_ROOT/android" ]; then
        log_info "  Android SDK: $OPENCV_ROOT/android/sdk"
    fi
    if [ -d "$OPENCV_ROOT/ios_install" ]; then
        log_info "  iOS: $OPENCV_ROOT/ios_install"
    fi
}

main "$@"
