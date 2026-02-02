#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-all-dev libmkl-full-dev libcpprest-dev python3.10

# Copy the local project files into the container
WORKDIR /app
COPY . .

# Build the project
RUN mkdir build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
RUN cmake --build build -- -j
