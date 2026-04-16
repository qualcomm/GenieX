/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_CONFIG_HPP
#define XTENSOR_CONFIG_HPP

#define XTENSOR_VERSION_MAJOR 0
#define XTENSOR_VERSION_MINOR 26
#define XTENSOR_VERSION_PATCH 0


// Define if the library is going to be using exceptions.
#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
#undef XTENSOR_DISABLE_EXCEPTIONS
#define XTENSOR_DISABLE_EXCEPTIONS
#endif

// Exception support.
#if defined(XTENSOR_DISABLE_EXCEPTIONS)
#include <iostream>
#define XTENSOR_THROW(_, msg)          \
    {                                  \
        std::cerr << msg << std::endl; \
        std::abort();                  \
    }
#else
#define XTENSOR_THROW(exception, msg) throw exception(msg)
#endif

// Workaround for some missing constexpr functionality in MSVC 2015 and MSVC 2017 x86
#if defined(_MSC_VER)
#define XTENSOR_CONSTEXPR_ENHANCED const
// The following must not be defined to const, otherwise
// it prevents generation of copy operators of classes
// containing XTENSOR_CONSTEXPR_ENHANCED_STATIC members
#define XTENSOR_CONSTEXPR_ENHANCED_STATIC
#define XTENSOR_CONSTEXPR_RETURN inline
#else
#define XTENSOR_CONSTEXPR_ENHANCED constexpr
#define XTENSOR_CONSTEXPR_RETURN constexpr
#define XTENSOR_CONSTEXPR_ENHANCED_STATIC constexpr static
#define XTENSOR_HAS_CONSTEXPR_ENHANCED
#endif

#ifndef XTENSOR_DEFAULT_DATA_CONTAINER
#define XTENSOR_DEFAULT_DATA_CONTAINER(T, A) uvector<T, A>
#endif

#ifndef XTENSOR_DEFAULT_SHAPE_CONTAINER
#define XTENSOR_DEFAULT_SHAPE_CONTAINER(T, EA, SA) \
    xt::svector<typename XTENSOR_DEFAULT_DATA_CONTAINER(T, EA)::size_type, 4, SA, true>
#endif

#ifdef XTENSOR_USE_XSIMD
#include <xsimd/xsimd.hpp>
#define XSIMD_DEFAULT_ALIGNMENT xsimd::default_arch::alignment()
#endif


#ifndef XTENSOR_DEFAULT_ALLOCATOR
#ifdef XTENSOR_ALLOC_TRACKING
#ifndef XTENSOR_ALLOC_TRACKING_POLICY
#define XTENSOR_ALLOC_TRACKING_POLICY xt::alloc_tracking::policy::print
#endif
#ifdef XTENSOR_USE_XSIMD
#include <xsimd/xsimd.hpp>
#define XTENSOR_DEFAULT_ALLOCATOR(T) \
    xt::tracking_allocator<T, xsimd::aligned_allocator<T, XSIMD_DEFAULT_ALIGNMENT>, XTENSOR_ALLOC_TRACKING_POLICY>
#else
#define XTENSOR_DEFAULT_ALLOCATOR(T) \
    xt::tracking_allocator<T, std::allocator<T>, XTENSOR_ALLOC_TRACKING_POLICY>
#endif
#else
#ifdef XTENSOR_USE_XSIMD

#define XTENSOR_DEFAULT_ALLOCATOR(T) xsimd::aligned_allocator<T, XTENSOR_DEFAULT_ALIGNMENT>
#else
#define XTENSOR_DEFAULT_ALLOCATOR(T) std::allocator<T>
#endif
#endif
#endif

#ifndef XTENSOR_DEFAULT_ALIGNMENT
#ifdef XTENSOR_USE_XSIMD
#define XTENSOR_DEFAULT_ALIGNMENT XSIMD_DEFAULT_ALIGNMENT
#else
#define XTENSOR_DEFAULT_ALIGNMENT 0
#endif
#endif

#ifndef XTENSOR_DEFAULT_LAYOUT
#define XTENSOR_DEFAULT_LAYOUT ::xt::layout_type::row_major
#endif

#ifndef XTENSOR_DEFAULT_TRAVERSAL
#define XTENSOR_DEFAULT_TRAVERSAL ::xt::layout_type::row_major
#endif

#ifndef XTENSOR_OPENMP_TRESHOLD
#define XTENSOR_OPENMP_TRESHOLD 0
#endif

#ifndef XTENSOR_TBB_THRESHOLD
#define XTENSOR_TBB_THRESHOLD 0
#endif

#ifndef XTENSOR_SELECT_ALIGN
#define XTENSOR_SELECT_ALIGN(T) (XTENSOR_DEFAULT_ALIGNMENT != 0 ? XTENSOR_DEFAULT_ALIGNMENT : alignof(T))
#endif

#ifndef XTENSOR_FIXED_ALIGN
#define XTENSOR_FIXED_ALIGN XTENSOR_SELECT_ALIGN(void*)
#endif

#ifdef IN_DOXYGEN
namespace xtl
{
    template <class... T>
    struct conjunction
    {
        constexpr bool value = true;
    };

    template <class... C>
    using check_concept = std::enable_if_t<conjunction<C...>::value, int>;

#define XTL_REQUIRES(...) xtl::check_concept<__VA_ARGS__> = 0
}
#endif

#endif
