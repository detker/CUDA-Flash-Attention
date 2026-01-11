#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include "error_utils.h"
#include "enum_types.h"

bool file_exists(const char* filename);
void load_binary_file(const char*, float*, size_t);
void save_binary_file(const char*, const float*, size_t);
void parse_config_string(const char*, int*, int*, int*, int*);
void parse_args(int, char**, ComputeType* compute_method, ModeType* mode, char** data_path);