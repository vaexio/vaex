#!/bin/bash
# extract from https://github.com/matthew-brett/multibuild/blob/a72a17b7a6150c74c927539564ab5badc4034321/common_utils.sh etc
set -e
set -x # echo on
PCRE_VERSION=${PCRE_VERSION:-8.38}
BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"


function rm_mkdir {
    # Remove directory if present, then make directory
    local path=$1
    if [ -z "$path" ]; then echo "Need not-empty path"; exit 1; fi
    if [ -d "$path" ]; then rm -rf $path; fi
    mkdir $path
}
function fetch_unpack {
    echo Fetch
    # Fetch input archive name from input URL
    # Parameters
    #    url - URL from which to fetch archive
    #    archive_fname (optional) archive name
    #
    # Echos unpacked directory and file names.
    #
    # If `archive_fname` not specified then use basename from `url`
    # If `archive_fname` already present at download location, use that instead.
    local url=$1
    if [ -z "$url" ];then echo "url not defined"; exit 1; fi
    local archive_fname=${2:-$(basename $url)}
    local arch_sdir="${ARCHIVE_SDIR:-archives}"
    # Make the archive directory in case it doesn't exist
    mkdir -p $arch_sdir
    local out_archive="${arch_sdir}/${archive_fname}"
    # If the archive is not already in the archives directory, get it.
    if [ ! -f "$out_archive" ]; then
        # Source it from multibuild archives if available.
        local our_archive="${MULTIBUILD_DIR}/archives/${archive_fname}"
        if [ -f "$our_archive" ]; then
            ln -s $our_archive $out_archive
        else
            # Otherwise download it.
            curl -L $url > $out_archive
        fi
    fi
    # Unpack archive, refreshing contents, echoing dir and file
    # names.
    tar xfv $out_archive
    # rm_mkdir arch_tmp
    # install_rsync
    # (cd arch_tmp && \
    #     untar ../$out_archive && \
    #     ls -1d * &&
    #     rsync --delete -ah * ..)
}


function build_simple {
    # Example: build_simple libpng $LIBPNG_VERSION \
    #               https://download.sourceforge.net/libpng tar.gz \
    #               --additional --configure --arguments
    local name=$1
    local version=$2
    local url=$3
    local ext=${4:-tar.gz}
    local configure_args=${@:5}
    # if [ -e "${name}-stamp" ]; then
    #     return
    # fi
    local name_version="${name}-${version}"
    local archive=${name_version}.${ext}
    local cflags=""
    fetch_unpack $url/$archive
    if [ "$ARCHFLAGS" = "-arch arm64" ]; then
        (cd $name_version \
            && CPPFLAGS="-arch arm64" CFLAGS="-arch arm64" LDFLAGS="-arch arm64"  ./configure --prefix=$BUILD_PREFIX $configure_args $ --host=arm-apple-darwin --build=x86_64-apple-darwin11.0.0 \
            && make uninstall \
            && make -j4 \
            && make install)
    else
        (cd $name_version \
            && ./configure --prefix=$BUILD_PREFIX $configure_args $cflags\
            && make uninstall \
            && make -j4 \
            && make install)
    fi
    # touch "${name}-stamp"
}


function build_pcre {
    echo "Build pcre"
    echo $ARCHFLAGS
    build_simple pcre $PCRE_VERSION http://ftp.exim.org/pub/pcre/
}

function install_precompiled() {
    # Mac https://formulae.brew.sh/formula/pcre
    # DebianUbuntu https://packages.ubuntu.com/libpcre3-dev
    # Alpine https://pkgs.alpinelinux.org/package/edge/main/x86_64/pcre
    # RHEL https://git.almalinux.org/rpms/pcre
    if [ -n "$(which brew)" ]; then
        brew install pcre
    elif [ -n "$(which apt)" ]; then
        apt update
        apt install -y libpcre3-dev
    elif [ -n "$(which apk)" ]; then
        apk add --update pcre pcre-dev
    elif [ -n "$(which dnf)" ]; then
        dnf --setopt install_weak_deps=false -y install pcre
    else
        false
    fi
}

echo "Install pcre"
install_precompiled || build_pcre
