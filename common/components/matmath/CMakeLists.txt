idf_component_register(SRCS "matmath.c" "matmath_s3.S" INCLUDE_DIRS "include")
# The C versions are the yardstick the assembly is measured against, so build them with
# -O2 whatever the project's own optimisation setting is.
set_source_files_properties(matmath.c PROPERTIES COMPILE_OPTIONS "-O2")
