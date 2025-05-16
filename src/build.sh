#!/bin/bash

set -e
this_script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd ${this_script_dir}

linux=1
android=0
clean_build=0

function help_message() {
    echo "$me --all : Build for all linux x86, aarch64-android"
    echo "$me --linux : (Default) Build for linux x86"
    echo "$me --android : Build for Android (aarch64-android). By default"
    echo "$me --clean : clean build directory and build"
    exit 1
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --all)
        linux=1
        android=1
        shift
        ;;
        --linux)
        linux=1
        android=0
        shift
        ;;
        --android)
        linux=0
        android=1
        shift
        ;;
        --clean)
        clean_build=1
        shift
        ;;
        -h|--help)
        help_message
        ;;
        --)
        shift
        break
        ;;
        -*)
        echo "Error: unknown option: $1"
        help_message
        ;;
        *)
        break
        ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ ${linux} -eq 1 ]; then
    build_dir="${this_script_dir}/build/linux_x86"
    if [ ${clean_build} -eq 1 ]; then
        [ -d ${build_dir} ] && rm -r ${build_dir}
    fi
    target="x86_64-linux-clang"
    cmake_args="-DZETIC_MLANGE_TARGET=${target}"

    mkdir -p ${build_dir}
    pushd ${build_dir}
    cmake ${cmake_args} ../.. && make -j8
    popd
fi

if [ ${android} -eq 1 ]; then
    build_dir="${this_script_dir}/build/aarch64-android"
    if [ ${clean_build} -eq 1 ]; then
        [ -d ${build_dir} ] && rm -r ${build_dir}
    fi
    target="aarch64-android"
    cmake_args="-DZETIC_MLANGE_TARGET=${target}"

    mkdir -p ${build_dir}
    pushd ${build_dir}
    cmake ${cmake_args} ../.. && make -j8
    popd
fi


popd
