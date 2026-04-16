/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EXCEPTION_HPP
#define XTENSOR_EXCEPTION_HPP

#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <xtl/xcompare.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xspan_impl.hpp>

#include "../core/xtensor_config.hpp"

#ifdef __GNUC__
#define XTENSOR_UNUSED_VARIABLE __attribute__((unused))
#else
#define XTENSOR_UNUSED_VARIABLE
#endif

namespace xt
{
    struct missing_type
    {
    };

    namespace
    {
        missing_type XTENSOR_UNUSED_VARIABLE missing;
    }

    namespace detail
    {
        template <class... Args>
        struct last_type_is_missing_impl
            : std::is_same<missing_type, xtl::mpl::back_t<xtl::mpl::vector<Args...>>>
        {
        };

        template <>
        struct last_type_is_missing_impl<> : std::false_type
        {
        };

        template <class... Args>
        constexpr bool last_type_is_missing = last_type_is_missing_impl<Args...>::value;
    }

    /*******************
     * broadcast_error *
     *******************/

    class broadcast_error : public std::runtime_error
    {
    public:

        explicit broadcast_error(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1& lhs, const S2& rhs);

    /*********************
     * concatenate_error *
     *********************/

    class concatenate_error : public std::runtime_error
    {
    public:

        explicit concatenate_error(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    template <class S1, class S2>
    [[noreturn]] void throw_concatenate_error(const S1& lhs, const S2& rhs);

    /**********************************
     * broadcast_error implementation *
     **********************************/

    namespace detail
    {
        template <class S1, class S2>
        inline std::string shape_error_message(const S1& lhs, const S2& rhs)
        {
            std::ostringstream buf("Incompatible dimension of arrays:", std::ios_base::ate);

            buf << "\n LHS shape = (";
            using size_type1 = typename S1::value_type;
            std::ostream_iterator<size_type1> iter1(buf, ", ");
            std::copy(lhs.cbegin(), lhs.cend(), iter1);

            buf << ")\n RHS shape = (";
            using size_type2 = typename S2::value_type;
            std::ostream_iterator<size_type2> iter2(buf, ", ");
            std::copy(rhs.cbegin(), rhs.cend(), iter2);
            buf << ")";

            return buf.str();
        }
    }

#ifdef NDEBUG
    // Do not inline this function
    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1&, const S2&)
    {
        XTENSOR_THROW(broadcast_error, "Incompatible dimension of arrays, compile in DEBUG for more info");
    }
#else
    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1& lhs, const S2& rhs)
    {
        std::string msg = detail::shape_error_message(lhs, rhs);
        XTENSOR_THROW(broadcast_error, msg.c_str());
    }
#endif

    /************************************
     * concatenate_error implementation *
     ************************************/

#ifdef NDEBUG
    // Do not inline this function
    template <class S1, class S2>
    [[noreturn]] void throw_concatenate_error(const S1&, const S2&)
    {
        XTENSOR_THROW(concatenate_error, "Incompatible dimension of arrays, compile in DEBUG for more info");
    }
#else
    template <class S1, class S2>
    [[noreturn]] void throw_concatenate_error(const S1& lhs, const S2& rhs)
    {
        std::string msg = detail::shape_error_message(lhs, rhs);
        XTENSOR_THROW(concatenate_error, msg.c_str());
    }
#endif

    /*******************
     * transpose_error *
     *******************/

    class transpose_error : public std::runtime_error
    {
    public:

        explicit transpose_error(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    /***************
     * check_index *
     ***************/

    template <class S, class... Args>
    void check_index(const S& shape, Args... args);

    template <class S, class It>
    void check_element_index(const S& shape, It first, It last);

    namespace detail
    {
        template <class S, std::size_t dim>
        inline void check_index_impl(const S&)
        {
        }

        template <class S, std::size_t dim>
        inline void check_index_impl(const S&, missing_type)
        {
        }

        template <class S, std::size_t dim, class T, class... Args>
        inline void check_index_impl(const S& shape, T arg, Args... args)
        {
            if (std::size_t(arg) >= std::size_t(shape[dim]) && shape[dim] != 1)
            {
                XTENSOR_THROW(
                    std::out_of_range,
                    "index " + std::to_string(arg) + " is out of bounds for axis " + std::to_string(dim)
                        + " with size " + std::to_string(shape[dim])
                );
            }
            check_index_impl<S, dim + 1>(shape, args...);
        }
    }

    template <class S>
    inline void check_index(const S&)
    {
    }

    template <class S>
    inline void check_index(const S&, missing_type)
    {
    }

    template <class S, class Arg, class... Args>
    inline void check_index(const S& shape, Arg arg, Args... args)
    {
        constexpr std::size_t nargs = sizeof...(Args) + 1;
        if (nargs == shape.size())
        {
            detail::check_index_impl<S, 0>(shape, arg, args...);
        }
        else if (nargs > shape.size())
        {
            // Too many arguments: drop the first
            check_index(shape, args...);
        }
        else if (detail::last_type_is_missing<Args...>)
        {
            // Too few arguments & last argument xt::missing: postfix index with zeros
            detail::check_index_impl<S, 0>(shape, arg, args...);
        }
        else
        {
            // Too few arguments: ignore the beginning of the shape
            auto it = shape.end() - nargs;
            detail::check_index_impl<decltype(it), 0>(it, arg, args...);
        }
    }

    template <class S, class It>
    inline void check_element_index(const S& shape, It first, It last)
    {
        using value_type = typename std::iterator_traits<It>::value_type;
        using size_type = typename S::size_type;
        auto dst = static_cast<size_type>(last - first);
        It efirst = last - static_cast<std::ptrdiff_t>((std::min)(shape.size(), dst));
        std::size_t axis = 0;

        while (efirst != last)
        {
            if (*efirst >= value_type(shape[axis]) && shape[axis] != 1)
            {
                XTENSOR_THROW(
                    std::out_of_range,
                    "index " + std::to_string(*efirst) + " is out of bounds for axis " + std::to_string(axis)
                        + " with size " + std::to_string(shape[axis])
                );
            }
            ++efirst, ++axis;
        }
    }

    /*******************
     * check_dimension *
     *******************/

    template <class S, class... Args>
    inline void check_dimension(const S& shape, Args...)
    {
        if (sizeof...(Args) > shape.size())
        {
            XTENSOR_THROW(
                std::out_of_range,
                "Number of arguments (" + std::to_string(sizeof...(Args))
                    + ") is greater than the number of dimensions (" + std::to_string(shape.size()) + ")"
            );
        }
    }

    /*******************************
     *  check_axis implementation  *
     *******************************/

    template <class A, class D>
    inline void check_axis_in_dim(A axis, D dim, const char* subject = "Axis")
    {
        const auto sdim = static_cast<std::make_signed_t<D>>(dim);
        if (xtl::cmp_greater_equal(axis, dim) || xtl::cmp_less(axis, -sdim))
        {
            XTENSOR_THROW(
                std::out_of_range,
                std::string(subject) + " (" + std::to_string(axis)
                    + ") is not within the number of dimensions (" + std::to_string(dim) + ')'
            );
        }
    }

    /****************
     * check_access *
     ****************/

    template <class S, class... Args>
    inline void check_access(const S& shape, Args... args)
    {
        check_dimension(shape, args...);
        check_index(shape, args...);
    }

#if (defined(XTENSOR_ENABLE_ASSERT) && !defined(XTENSOR_DISABLE_EXCEPTIONS))
#define XTENSOR_TRY(expr) XTENSOR_TRY_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_TRY_IMPL(expr, file, line)                                                                \
    try                                                                                                   \
    {                                                                                                     \
        expr;                                                                                             \
    }                                                                                                     \
    catch (std::exception & e)                                                                            \
    {                                                                                                     \
        XTENSOR_THROW(                                                                                    \
            std::runtime_error,                                                                           \
            std::string(file) + ':' + std::to_string(line) + ": check failed\n\t" + std::string(e.what()) \
        );                                                                                                \
    }
#else
#define XTENSOR_TRY(expr)
#endif

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ASSERT(expr) XTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_ASSERT_IMPL(expr, file, line)                                                      \
    if (!(expr))                                                                                   \
    {                                                                                              \
        XTENSOR_THROW(                                                                             \
            std::runtime_error,                                                                    \
            std::string(file) + ':' + std::to_string(line) + ": assertion failed (" #expr ") \n\t" \
        );                                                                                         \
    }
#else
#define XTENSOR_ASSERT(expr)
#endif

#ifdef XTENSOR_ENABLE_CHECK_DIMENSION
#define XTENSOR_CHECK_DIMENSION(S, ARGS) XTENSOR_TRY(check_dimension(S, ARGS))
#else
#define XTENSOR_CHECK_DIMENSION(S, ARGS)
#endif

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ASSERT_MSG(expr, msg)                                                                            \
    if (!(expr))                                                                                                 \
    {                                                                                                            \
        XTENSOR_THROW(                                                                                           \
            std::runtime_error,                                                                                  \
            std::string("Assertion error!\n") + msg + "\n  " + __FILE__ + '(' + std::to_string(__LINE__) + ")\n" \
        );                                                                                                       \
    }
#else
#define XTENSOR_ASSERT_MSG(expr, msg)
#endif

#define XTENSOR_PRECONDITION(expr, msg)                                              \
    if (!(expr))                                                                     \
    {                                                                                \
        XTENSOR_THROW(                                                               \
            std::runtime_error,                                                      \
            std::string("Precondition violation!\n") + msg + "\n  " + __FILE__ + '(' \
                + std::to_string(__LINE__) + ")\n"                                   \
        );                                                                           \
    }
}
#endif  // XEXCEPTION_HPP
