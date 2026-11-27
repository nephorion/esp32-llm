cmake_minimum_required(VERSION 3.16)

# Shared code and settings live in ../../common (see the repository README).
set(EXTRA_COMPONENT_DIRS ../../common/components)
set(SDKCONFIG_DEFAULTS "../../common/sdkconfig.common;../../common/sdkconfig.small-app;sdkconfig.defaults")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(matbench)
