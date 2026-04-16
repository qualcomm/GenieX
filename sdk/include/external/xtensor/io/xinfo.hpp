/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_INFO_HPP
#define XTENSOR_INFO_HPP

#include <string>

#include "../core/xlayout.hpp"

#ifndef _MSC_VER
#if __cplusplus < 201103
#define CONSTEXPR11_TN
#define CONSTEXPR14_TN
#define NOEXCEPT_TN
#elif __cplusplus < 201402
#define CONSTEXPR11_TN constexpr
#define CONSTEXPR14_TN
#define NOEXCEPT_TN noexcept
#else
#define CONSTEXPR11_TN constexpr
#define CONSTEXPR14_TN constexpr
#define NOEXCEPT_TN noexcept
#endif
#else  // _MSC_VER
#if _MSC_VER < 1900
#define CONSTEXPR11_TN
#define CONSTEXPR14_TN
#define NOEXCEPT_TN
#elif _MSC_VER < 2000
#define CONSTEXPR11_TN constexpr
#define CONSTEXPR14_TN
#define NOEXCEPT_TN noexcept
#else
#define CONSTEXPR11_TN constexpr
#define CONSTEXPR14_TN constexpr
#define NOEXCEPT_TN noexcept
#endif
#endif

namespace xt
{
    // see http://stackoverflow.com/a/20170989
    struct static_string
    {
        template <std::size_t N>
        explicit CONSTEXPR11_TN static_string(const char (&a)[N]) NOEXCEPT_TN : data(a),
                                                                                size(N - 1)
        {
        }

        CONSTEXPR11_TN static_string(const char* a, const std::size_t sz) NOEXCEPT_TN : data(a),
                                                                                        size(sz)
        {
        }

        const char* const data;
        const std::size_t size;
    };

    template <class T>
    CONSTEXPR14_TN static_string type_name()
    {
#ifdef __clang__
        static_string p(__PRETTY_FUNCTION__);
        return static_string(p.data + 39, p.size - 39 - 1);
#elif defined(__GNUC__)
        static_string p(__PRETTY_FUNCTION__);
#if __cplusplus < 201402
        return static_string(p.data + 36, p.size - 36 - 1);
#else
        return static_string(p.data + 54, p.size - 54 - 1);
#endif
#elif defined(_MSC_VER)
        static const static_string p(__FUNCSIG__);
        return static_string(p.data + 47, p.size - 47 - 7);
#endif
    }

    template <class T>
    std::string type_to_string()
    {
        static_string static_name = type_name<T>();
        return std::string(static_name.data, static_name.size);
    }

    template <class T>
    std::string info(const T& t)
    {
        std::string s;
        s += "\nValue type: " + type_to_string<typename T::value_type>();
        s += "\nLayout: ";
        if (t.layout() == layout_type::row_major)
        {
            s += "row_major";
        }
        else if (t.layout() == layout_type::column_major)
        {
            s += "column_major";
        }
        else if (t.layout() == layout_type::dynamic)
        {
            s += "dynamic";
        }
        else
        {
            s += "any";
        }
        s += "\nShape: (";
        bool first = true;
        for (const auto& el : t.shape())
        {
            if (!first)
            {
                s += ", ";
            }
            first = false;
            s += std::to_string(el);
        }
        s += ")\nStrides: (";
        first = true;
        for (const auto& el : t.strides())
        {
            if (!first)
            {
                s += ", ";
            }
            first = false;
            s += std::to_string(el);
        }
        s += ")\nSize: " + std::to_string(t.size()) + "\n";
        return s;
    }
}

#endif
