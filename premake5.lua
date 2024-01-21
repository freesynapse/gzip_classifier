workspace "syn_app"
    -- location of generated solution/make and build files
    location "build"

    -- architecture: 'x86' or 'x86_64'
    architecture "x86_64"
    staticruntime "on"
  
    -- language to be compiled and c++ flavor
    language "C++"
    cppdialect "C++17"

    -- set flags for the compiler
    flags { "MultiProcessorCompile" }

    -- for using OpenMP
    --buildoptions { "-fopenmp" }

    -- used for storing compiler / linker settings togehter
    configurations { "Debug", "Release" }

    -- enable symbols for Debug builds
    filter "configurations:Debug"
        defines { "DEBUG", "DEBUG_PROFILING" }
        optimize "Debug"
        symbols "On"
        linkoptions { "-rdynamic" }

    -- enable optimization for Release builds
    filter "configurations:Release"
        defines "NDEBUG"
        --optimize "On" --> -O2
        optimize "Speed" -- --> -O3

    -- reset filter
    filter { }


-----------------------------------------------------------------------------------------
project "gzip_classifier"

    kind "ConsoleApp"

    targetdir ("%{wks.location}")
	objdir ("%{wks.location}/obj")

    -- show preprocessor output -- didn't work
    --buildoptions { "-E" }

    files
    {
        "src/**.c",
        "src/**.cpp",
        "src/**.h",
        "src/**.hpp",
    }

    defines
    {
        -- "_CRT_SECURE_NO_WARNINGS",
        -- "GLFW_INCLUDE_NONE",
    }

    includedirs
    {
        ".",    -- to allow orderly structured includes in files
                -- in the style #include "src/.../file.h" instead
                -- of #include "./file.h" or #include "../../file.h".
    }

    libdirs
    {
    }

    links
    {
        "pthread",
        "m",
        "z",
        "synapse",
    }

    filter { "configurations.Debug" }
        runtime "Debug"

    filter { "configurations.Release" }
        runtime "Release"

