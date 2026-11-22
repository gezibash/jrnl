#!/bin/sh
# Builds the release archives into dist/: ./release.sh 3.5.0
#
# GOEXPERIMENT=simd: the built-in embedding model (pure Go, GoMLX) answers a
# query about 2x and indexes about 6x faster with Go's SIMD support; a build
# without it (go install) works, only slower.
set -eu
v=${1:?give the version, e.g. 3.5.0}
rm -rf dist
mkdir dist
for t in darwin/amd64 darwin/arm64 linux/amd64 linux/arm64; do
	os=${t%/*}
	arch=${t#*/}
	d=$(mktemp -d)
	GOEXPERIMENT=simd CGO_ENABLED=0 GOOS=$os GOARCH=$arch \
		go build -trimpath -ldflags "-s -w -X main.version=$v" -o "$d/jrnl" .
	cp README.md LICENSE "$d/"
	tar -czf "dist/jrnl_${v}_${os}_${arch}.tar.gz" -C "$d" jrnl README.md LICENSE
	rm -rf "$d"
done
(cd dist && shasum -a 256 ./*.tar.gz | sed 's| \./| |' >checksums.txt)
cat dist/checksums.txt
