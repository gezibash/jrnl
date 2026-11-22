#!/bin/sh
# Installs the jrnl binary from GitHub releases.
#
#   curl -fsSL https://raw.githubusercontent.com/gezibash/jrnl/main/install.sh | sh
#
# JRNL_VERSION=3.2.0 picks a release (default: the latest).
# JRNL_INSTALL_DIR picks the folder (default: ~/.local/bin).
set -eu

repo=gezibash/jrnl
dir=${JRNL_INSTALL_DIR:-$HOME/.local/bin}

fail() {
	echo "jrnl install: $*" >&2
	exit 1
}

case $(uname -s) in
Darwin) os=darwin ;;
Linux) os=linux ;;
*) fail "no build for $(uname -s); try: go install github.com/$repo@main" ;;
esac
case $(uname -m) in
x86_64 | amd64) arch=amd64 ;;
arm64 | aarch64) arch=arm64 ;;
*) fail "no build for $(uname -m); try: go install github.com/$repo@main" ;;
esac

version=${JRNL_VERSION:-}
if [ -z "$version" ]; then
	# /releases/latest redirects to /releases/tag/vX.Y.Z.
	url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$repo/releases/latest") ||
		fail "could not reach github.com"
	version=${url##*/v}
fi
version=${version#v}

asset="jrnl_${version}_${os}_${arch}.tar.gz"
base="https://github.com/$repo/releases/download/v$version"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "jrnl install: downloading $asset"
curl -fsSL -o "$tmp/$asset" "$base/$asset" || fail "no $asset in release v$version"
curl -fsSL -o "$tmp/checksums.txt" "$base/checksums.txt" || fail "no checksums.txt in release v$version"

want=$(awk -v f="$asset" '$2 == f { print $1 }' "$tmp/checksums.txt")
[ -n "$want" ] || fail "$asset is not listed in checksums.txt"
if command -v sha256sum >/dev/null 2>&1; then
	got=$(sha256sum "$tmp/$asset" | awk '{ print $1 }')
else
	got=$(shasum -a 256 "$tmp/$asset" | awk '{ print $1 }')
fi
[ "$got" = "$want" ] || fail "checksum mismatch for $asset; nothing installed"

tar -xzf "$tmp/$asset" -C "$tmp" jrnl
mkdir -p "$dir"
mv "$tmp/jrnl" "$dir/jrnl"
chmod 755 "$dir/jrnl"
echo "jrnl install: installed $("$dir/jrnl" version) to $dir/jrnl"

found=$(command -v jrnl 2>/dev/null || true)
case :$PATH: in
*:"$dir":*)
	if [ -n "$found" ] && [ "$found" != "$dir/jrnl" ]; then
		echo "jrnl install: note: $found comes first on PATH; it may be another program named jrnl" >&2
	fi
	;;
*) echo "jrnl install: add $dir to PATH" >&2 ;;
esac
