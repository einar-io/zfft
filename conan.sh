#!/usr/bin/env

#mkdir -p build/
conan install . --build missing -pr:b=default -if build/ -of build/
