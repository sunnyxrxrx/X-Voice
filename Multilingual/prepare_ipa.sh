#!/bin/bash
set -e

pip install phonemizer

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
  echo "请以 root 身份运行此脚本 (sudo ./install.sh)"
  exit 1
fi

# 临时编译目录
BUILD_DIR="/tmp/espeak_build_tmp"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 并行核心数
CORES=$(nproc)

echo "安装基础编译工具"
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y build-essential autoconf automake libtool pkg-config git wget
elif command -v yum &> /dev/null; then
    yum groupinstall -y "Development Tools"
    yum install -y autoconf automake libtool pkgconfig git wget
fi

echo "安装 PCAudioLib"
if [ ! -d "pcaudiolib" ]; then
    git clone https://github.com/espeak-ng/pcaudiolib.git
fi
cd pcaudiolib
./autogen.sh
# 安装到 /usr
./configure --prefix=/usr --sysconfdir=/etc
make -j$CORES
make install
cd ..

echo "安装Espeak-NG"
if [ ! -d "espeak-ng" ]; then
    git clone https://github.com/espeak-ng/espeak-ng.git
fi
cd espeak-ng
./autogen.sh
# 安装到 /usr
./configure --prefix=/usr --sysconfdir=/etc
make -j$CORES
make install

# 刷新动态库缓存，让系统立即识别新安装的 .so 文件
ldconfig

echo "清理"
rm -rf "$BUILD_DIR"
echo "验证："
espeak-ng --version  
# 预期: Data at: /usr/share/espeak-ng-data
# python
# from phonemizer.backend import EspeakBackend
# print(EspeakBackend('en-us').phonemize(['Hello, world!']))
# 预期: ['həloʊ wɜːld ']