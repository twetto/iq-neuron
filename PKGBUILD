# This is an example PKGBUILD file. Use this as a start to creating your own,
# and remove these comments. For more information, see 'man PKGBUILD'.
# NOTE: Please fill out the license field for your package! If it is unknown,
# then please put 'unknown'.

# The following guidelines are specific to BZR, GIT, HG and SVN packages.
# Other VCS sources are not natively supported by makepkg yet.

# Maintainer: twetto <franky85912@gmail.com>
pkgname=iq-neuron
pkgver=0.0.2.r0.g268f982
pkgrel=1
pkgdesc="A library for IQIF"
arch=('any')
url="https://github.com/twetto/iq-neuron"
license=('MIT')
depends=('openmp')
makedepends=('git' 'gcc' 'cmake') # 'bzr', 'git', 'mercurial' or 'subversion'
source=('git+https://github.com/twetto/iq-neuron.git')
sha256sums=('SKIP')

# Please refer to the 'USING VCS SOURCES' section of the PKGBUILD man page for
# a description of each element in the source array.

pkgver() {
	cd "$srcdir/${pkgname}"

# Git, tags available
    printf "%s" "$(git describe --long --tags | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g')"

}

build() {
	cd "$srcdir/${pkgname}"
    cmake -B build -S "$srcdir/${pkgname}" \
        -DCMAKE_INSTALL_PREFIX='/usr'
    make -C build
}

package() {
	cd "$srcdir/${pkgname}/build"
	make DESTDIR="$pkgdir/" install
}
