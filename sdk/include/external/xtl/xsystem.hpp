/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XSYSTEM_HPP
#define XTL_XSYSTEM_HPP

#if defined(__linux__)
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  if defined(NOMINMAX)
#    include <windows.h>
#  else
#    define NOMINMAX
#    include <windows.h>
#    undef NOMINMAX
#  endif
#endif
#ifdef __APPLE__
#  include <cstdint>
#  include <mach-o/dyld.h>
#endif
#if defined(__sun)
#  include <stdlib.h>
#endif
#ifdef __FreeBSD__
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif

#include <cstring>
#include <string>

namespace xtl
{
    std::string executable_path();
    std::string prefix_path();

    /******************
     * implementation *
     ******************/
    
    inline std::string executable_path()
    {
        std::string path;
#if defined(UNICODE)
    wchar_t buffer[1024];
#else
    char buffer[1024];
#endif
        std::memset(buffer, '\0', sizeof(buffer));
#if defined(__linux__)
        if (readlink("/proc/self/exe", buffer, sizeof(buffer)) != -1)
        {
            path = buffer;
        }
        else
        {
            // failed to determine run path
        }
#elif defined (_WIN32)
    #if defined(UNICODE)
        if (GetModuleFileNameW(nullptr, buffer, sizeof(buffer)) != 0)
        {
            // Convert wchar_t to std::string
            std::wstring wideString(buffer);
            std::string narrowString(wideString.begin(), wideString.end());
            path = narrowString;
        }
    #else
        if (GetModuleFileNameA(nullptr, buffer, sizeof(buffer)) != 0)
        {
            path = buffer;
        }
    #endif
        // failed to determine run path
#elif defined (__APPLE__)
        std::uint32_t size = sizeof(buffer);
        if(_NSGetExecutablePath(buffer, &size) == 0)
        {
            path = buffer;
        }
        else
        {
            // failed to determine run path
        }
#elif defined (__FreeBSD__)
        int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
        size_t buffer_size = sizeof(buffer);
        if (sysctl(mib, 4, buffer, &buffer_size, NULL, 0) != -1)
        {
            path = buffer;
        }
        else
        {
            // failed to determine run path
        }
#elif defined(__sun)
        path = getexecname();
#endif
        return path;
    }

    inline std::string prefix_path()
    {
        std::string path = executable_path();
#if defined (_WIN32)
        char separator = '\\';
#else
        char separator = '/';
#endif
        std::string bin_folder = path.substr(0, path.find_last_of(separator));
        std::string prefix = bin_folder.substr(0, bin_folder.find_last_of(separator)) + separator;
        return prefix;
    }
}

#endif

