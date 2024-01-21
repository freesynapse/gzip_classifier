#pragma once

#include <stdio.h>

//
#define LOG_INFO(...) { printf("[INFO] " __VA_ARGS__); }
#define LOG_ERROR(...) { printf("[ERROR] " __VA_ARGS__); }

