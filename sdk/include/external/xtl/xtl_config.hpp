/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_CONFIG_HPP
#define XTL_CONFIG_HPP

#define XTL_VERSION_MAJOR 0
#define XTL_VERSION_MINOR 8
#define XTL_VERSION_PATCH 0

#ifndef __has_feature
#define __has_feature(x) 0
#endif

// Attempt to discover whether we're being compiled with exception support
#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && !defined(XTL_NO_EXCEPTIONS)
// Exceptions are enabled.
#else
#if !defined(XTL_NO_EXCEPTIONS)
// Exceptions are disabled.
#define XTL_NO_EXCEPTIONS
#endif
#endif

#if defined(XTL_NO_EXCEPTIONS)

#include <iostream>

#define XTL_THROW(_, msg)              \
    {                                  \
        std::cerr << msg << std::endl; \
        std::abort();                  \
    }
#else
#define XTL_THROW(exception, msg) throw exception(msg)
#endif

#endif
