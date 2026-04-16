/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_IO_HPP
#define XTENSOR_IO_HPP

#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include "../core/xexpression.hpp"
#include "../core/xmath.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e);

    /*****************
     * print options *
     *****************/

    namespace print_options
    {
        struct print_options_impl
        {
            int edge_items = 3;
            int line_width = 75;
            int threshold = 1000;
            int precision = -1;  // default precision
        };

        inline print_options_impl& print_options()
        {
            static print_options_impl po;
            return po;
        }

        /**
         * @brief Sets the line width. After \a line_width chars,
         *        a new line is added.
         *
         * @param line_width The line width
         */
        inline void set_line_width(int line_width)
        {
            print_options().line_width = line_width;
        }

        /**
         * @brief Sets the threshold after which summarization is triggered (default: 1000).
         *
         * @param threshold The number of elements in the xexpression that triggers
         *                  summarization in the output
         */
        inline void set_threshold(int threshold)
        {
            print_options().threshold = threshold;
        }

        /**
         * @brief Sets the number of edge items. If the summarization is
         *        triggered, this value defines how many items of each dimension
         *        are printed.
         *
         * @param edge_items The number of edge items
         */
        inline void set_edge_items(int edge_items)
        {
            print_options().edge_items = edge_items;
        }

        /**
         * @brief Sets the precision for printing floating point values.
         *
         * @param precision The number of digits for floating point output
         */
        inline void set_precision(int precision)
        {
            print_options().precision = precision;
        }

#define DEFINE_LOCAL_PRINT_OPTION(NAME)                               \
    class NAME                                                        \
    {                                                                 \
    public:                                                           \
                                                                      \
        NAME(int value)                                               \
            : m_value(value)                                          \
        {                                                             \
            id();                                                     \
        }                                                             \
        static int id()                                               \
        {                                                             \
            static int id = std::ios_base::xalloc();                  \
            return id;                                                \
        }                                                             \
        int value() const                                             \
        {                                                             \
            return m_value;                                           \
        }                                                             \
                                                                      \
    private:                                                          \
                                                                      \
        int m_value;                                                  \
    };                                                                \
                                                                      \
    inline std::ostream& operator<<(std::ostream& out, const NAME& n) \
    {                                                                 \
        out.iword(NAME::id()) = n.value();                            \
        return out;                                                   \
    }

        /**
         * @class line_width
         *
         * io manipulator used to set the width of the lines when printing
         * an expression.
         *
         * @code{.cpp}
         * using po = xt::print_options;
         * xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};
         * std::cout << po::line_width(100) << a << std::endl;
         * @endcode
         */
        DEFINE_LOCAL_PRINT_OPTION(line_width)

        /**
         * @class threshold
         *
         * io manipulator used to set the threshold after which summarization is
         * triggered.
         *
         * @code{.cpp}
         * using po = xt::print_options;
         * xt::xarray<double> a = xt::rand::randn<double>({2000, 500});
         * std::cout << po::threshold(50) << a << std::endl;
         * @endcode
         */
        DEFINE_LOCAL_PRINT_OPTION(threshold)

        /**
         * @class edge_items
         *
         * io manipulator used to set the number of egde items if
         * the summarization is triggered.
         *
         * @code{.cpp}
         * using po = xt::print_options;
         * xt::xarray<double> a = xt::rand::randn<double>({2000, 500});
         * std::cout << po::edge_items(5) << a << std::endl;
         * @endcode
         */
        DEFINE_LOCAL_PRINT_OPTION(edge_items)

        /**
         * @class precision
         *
         * io manipulator used to set the precision of the floating point values
         * when printing an expression.
         *
         * @code{.cpp}
         * using po = xt::print_options;
         * xt::xarray<double> a = xt::rand::randn<double>({2000, 500});
         * std::cout << po::precision(5) << a << std::endl;
         * @endcode
         */
        DEFINE_LOCAL_PRINT_OPTION(precision)
    }

    /**************************************
     * xexpression ostream implementation *
     **************************************/

    namespace detail
    {
        template <class E, class F>
        std::ostream& xoutput(
            std::ostream& out,
            const E& e,
            xstrided_slice_vector& slices,
            F& printer,
            std::size_t blanks,
            std::streamsize element_width,
            std::size_t edgeitems,
            std::size_t line_width
        )
        {
            using size_type = typename E::size_type;

            const auto view = xt::strided_view(e, slices);
            if (view.dimension() == 0)
            {
                printer.print_next(out);
            }
            else
            {
                std::string indents(blanks, ' ');

                size_type i = 0;
                size_type elems_on_line = 0;
                const size_type ewp2 = static_cast<size_type>(element_width) + size_type(2);
                const size_type line_lim = static_cast<size_type>(std::floor(line_width / ewp2));

                out << '{';
                for (; i != size_type(view.shape()[0] - 1); ++i)
                {
                    if (edgeitems && size_type(view.shape()[0]) > (edgeitems * 2) && i == edgeitems)
                    {
                        if (view.dimension() == 1 && line_lim != 0 && elems_on_line >= line_lim)
                        {
                            out << " ...,";
                        }
                        else if (view.dimension() > 1)
                        {
                            elems_on_line = 0;
                            out << "...," << std::endl << indents;
                        }
                        else
                        {
                            out << "..., ";
                        }
                        i = size_type(view.shape()[0]) - edgeitems;
                    }
                    if (view.dimension() == 1 && line_lim != 0 && elems_on_line >= line_lim)
                    {
                        out << std::endl << indents;
                        elems_on_line = 0;
                    }
                    slices.push_back(static_cast<int>(i));
                    xoutput(out, e, slices, printer, blanks + 1, element_width, edgeitems, line_width) << ',';
                    slices.pop_back();
                    elems_on_line++;

                    if ((view.dimension() == 1) && !(line_lim != 0 && elems_on_line >= line_lim))
                    {
                        out << ' ';
                    }
                    else if (view.dimension() > 1)
                    {
                        out << std::endl << indents;
                    }
                }
                if (view.dimension() == 1 && line_lim != 0 && elems_on_line >= line_lim)
                {
                    out << std::endl << indents;
                }
                slices.push_back(static_cast<int>(i));
                xoutput(out, e, slices, printer, blanks + 1, element_width, edgeitems, line_width) << '}';
                slices.pop_back();
            }
            return out;
        }

        template <class F, class E>
        void recurser_run(F& fn, const E& e, xstrided_slice_vector& slices, std::size_t lim = 0)
        {
            using size_type = typename E::size_type;
            const auto view = strided_view(e, slices);
            if (view.dimension() == 0)
            {
                fn.update(view());
            }
            else
            {
                size_type i = 0;
                for (; i != static_cast<size_type>(view.shape()[0] - 1); ++i)
                {
                    if (lim && size_type(view.shape()[0]) > (lim * 2) && i == lim)
                    {
                        i = static_cast<size_type>(view.shape()[0]) - lim;
                    }
                    slices.push_back(static_cast<int>(i));
                    recurser_run(fn, e, slices, lim);
                    slices.pop_back();
                }
                slices.push_back(static_cast<int>(i));
                recurser_run(fn, e, slices, lim);
                slices.pop_back();
            }
        }

        template <class T, class E = void>
        struct printer;

        template <class T>
        struct printer<T, std::enable_if_t<std::is_floating_point<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<value_type>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize precision)
                : m_precision(precision)
            {
            }

            void init()
            {
                m_precision = m_required_precision < m_precision ? m_required_precision : m_precision;
                m_it = m_cache.cbegin();
                if (m_scientific)
                {
                    // 3 = sign, number and dot and 4 = "e+00"
                    m_width = m_precision + 7;
                    if (m_large_exponent)
                    {
                        // = e+000 (additional number)
                        m_width += 1;
                    }
                }
                else
                {
                    std::streamsize decimals = 1;  // print a leading 0
                    if (std::floor(m_max) != 0)
                    {
                        decimals += std::streamsize(std::log10(std::floor(m_max)));
                    }
                    // 2 => sign and dot
                    m_width = 2 + decimals + m_precision;
                }
                if (!m_required_precision)
                {
                    --m_width;
                }
            }

            std::ostream& print_next(std::ostream& out)
            {
                if (!m_scientific)
                {
                    std::stringstream buf;
                    buf.width(m_width);
                    buf << std::fixed;
                    buf.precision(m_precision);
                    buf << (*m_it);
                    if (!m_required_precision && !std::isinf(*m_it) && !std::isnan(*m_it))
                    {
                        buf << '.';
                    }
                    std::string res = buf.str();
                    auto sit = res.rbegin();
                    while (*sit == '0')
                    {
                        *sit = ' ';
                        ++sit;
                    }
                    out << res;
                }
                else
                {
                    if (!m_large_exponent)
                    {
                        out << std::scientific;
                        out.width(m_width);
                        out << (*m_it);
                    }
                    else
                    {
                        std::stringstream buf;
                        buf.width(m_width);
                        buf << std::scientific;
                        buf.precision(m_precision);
                        buf << (*m_it);
                        std::string res = buf.str();

                        if (res[res.size() - 4] == 'e')
                        {
                            res.erase(0, 1);
                            res.insert(res.size() - 2, "0");
                        }
                        out << res;
                    }
                }
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                if (val != 0 && !std::isinf(val) && !std::isnan(val))
                {
                    if (!m_scientific || !m_large_exponent)
                    {
                        int exponent = 1 + int(std::log10(math::abs(val)));
                        if (exponent <= -5 || exponent > 7)
                        {
                            m_scientific = true;
                            m_required_precision = m_precision;
                            if (exponent <= -100 || exponent >= 100)
                            {
                                m_large_exponent = true;
                            }
                        }
                    }
                    if (math::abs(val) > m_max)
                    {
                        m_max = math::abs(val);
                    }
                    if (m_required_precision < m_precision)
                    {
                        while (std::floor(val * std::pow(10, m_required_precision))
                               != val * std::pow(10, m_required_precision))
                        {
                            m_required_precision++;
                        }
                    }
                }
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            bool m_large_exponent = false;
            bool m_scientific = false;
            std::streamsize m_width = 9;
            std::streamsize m_precision;
            std::streamsize m_required_precision = 0;
            value_type m_max = 0;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<
            T,
            std::enable_if_t<
                xtl::is_integral<typename T::value_type>::value && !std::is_same<typename T::value_type, bool>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<value_type>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
                m_width = 1 + std::streamsize((m_max > 0) ? std::log10(m_max) : 0) + m_sign;
            }

            std::ostream& print_next(std::ostream& out)
            {
                // + enables printing of chars etc. as numbers
                // TODO should chars be printed as numbers?
                out.width(m_width);
                out << +(*m_it);
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                if (math::abs(val) > m_max)
                {
                    m_max = math::abs(val);
                }
                if (xtl::is_signed<value_type>::value && val < 0)
                {
                    m_sign = true;
                }
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width;
            bool m_sign = false;
            value_type m_max = 0;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<std::is_same<typename T::value_type, bool>::value>>
        {
            using value_type = bool;
            using cache_type = std::vector<bool>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
            }

            std::ostream& print_next(std::ostream& out)
            {
                if (*m_it)
                {
                    out << " true";
                }
                else
                {
                    out << "false";
                }
                // TODO: the following std::setw(5) isn't working correctly on OSX.
                // out << std::boolalpha << std::setw(m_width) << (*m_it);
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width = 5;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<xtl::is_complex<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<bool>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize precision)
                : real_printer(precision)
                , imag_printer(precision)
            {
            }

            void init()
            {
                real_printer.init();
                imag_printer.init();
                m_it = m_signs.cbegin();
            }

            std::ostream& print_next(std::ostream& out)
            {
                real_printer.print_next(out);
                if (*m_it)
                {
                    out << "-";
                }
                else
                {
                    out << "+";
                }
                std::stringstream buf;
                imag_printer.print_next(buf);
                std::string s = buf.str();
                if (s[0] == ' ')
                {
                    s.erase(0, 1);  // erase space for +/-
                }
                // insert j at end of number
                std::size_t idx = s.find_last_not_of(" ");
                s.insert(idx + 1, "i");
                out << s;
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                real_printer.update(val.real());
                imag_printer.update(std::abs(val.imag()));
                m_signs.push_back(std::signbit(val.imag()));
            }

            std::streamsize width()
            {
                return real_printer.width() + imag_printer.width() + 2;
            }

        private:

            printer<value_type> real_printer, imag_printer;
            cache_type m_signs;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<
            T,
            std::enable_if_t<
                !xtl::is_fundamental<typename T::value_type>::value && !xtl::is_complex<typename T::value_type>::value>>
        {
            using const_reference = typename T::const_reference;
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<std::string>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
                if (m_width > 20)
                {
                    m_width = 0;
                }
            }

            std::ostream& print_next(std::ostream& out)
            {
                out.width(m_width);
                out << *m_it;
                ++m_it;
                return out;
            }

            void update(const_reference val)
            {
                std::stringstream buf;
                buf << val;
                std::string s = buf.str();
                if (int(s.size()) > m_width)
                {
                    m_width = std::streamsize(s.size());
                }
                m_cache.push_back(s);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width = 0;
            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class E>
        struct custom_formatter
        {
            using value_type = std::decay_t<typename E::value_type>;

            template <class F>
            custom_formatter(F&& func)
                : m_func(func)
            {
            }

            std::string operator()(const value_type& val) const
            {
                return m_func(val);
            }

        private:

            std::function<std::string(const value_type&)> m_func;
        };
    }

    inline print_options::print_options_impl get_print_options(std::ostream& out)
    {
        print_options::print_options_impl res;
        using print_options::edge_items;
        using print_options::line_width;
        using print_options::precision;
        using print_options::threshold;

        res.edge_items = static_cast<int>(out.iword(edge_items::id()));
        res.line_width = static_cast<int>(out.iword(line_width::id()));
        res.threshold = static_cast<int>(out.iword(threshold::id()));
        res.precision = static_cast<int>(out.iword(precision::id()));

        if (!res.edge_items)
        {
            res.edge_items = print_options::print_options().edge_items;
        }
        else
        {
            out.iword(edge_items::id()) = long(0);
        }
        if (!res.line_width)
        {
            res.line_width = print_options::print_options().line_width;
        }
        else
        {
            out.iword(line_width::id()) = long(0);
        }
        if (!res.threshold)
        {
            res.threshold = print_options::print_options().threshold;
        }
        else
        {
            out.iword(threshold::id()) = long(0);
        }
        if (!res.precision)
        {
            res.precision = print_options::print_options().precision;
        }
        else
        {
            out.iword(precision::id()) = long(0);
        }

        return res;
    }

    template <class E, class F>
    std::ostream& pretty_print(const xexpression<E>& e, F&& func, std::ostream& out = std::cout)
    {
        xfunction<detail::custom_formatter<E>, const_xclosure_t<E>> print_fun(
            detail::custom_formatter<E>(std::forward<F>(func)),
            e
        );
        return pretty_print(print_fun, out);
    }

    namespace detail
    {
        template <class S>
        class fmtflags_guard
        {
        public:

            explicit fmtflags_guard(S& stream)
                : m_stream(stream)
                , m_flags(stream.flags())
            {
            }

            ~fmtflags_guard()
            {
                m_stream.flags(m_flags);
            }

        private:

            S& m_stream;
            std::ios_base::fmtflags m_flags;
        };
    }

    template <class E>
    std::ostream& pretty_print(const xexpression<E>& e, std::ostream& out = std::cout)
    {
        detail::fmtflags_guard<std::ostream> guard(out);

        const E& d = e.derived_cast();

        std::size_t lim = 0;
        std::size_t sz = compute_size(d.shape());

        auto po = get_print_options(out);

        if (sz > static_cast<std::size_t>(po.threshold))
        {
            lim = static_cast<std::size_t>(po.edge_items);
        }
        if (sz == 0)
        {
            out << "{}";
            return out;
        }

        auto temp_precision = out.precision();
        auto precision = temp_precision;
        if (po.precision != -1)
        {
            out.precision(static_cast<std::streamsize>(po.precision));
            precision = static_cast<std::streamsize>(po.precision);
        }

        detail::printer<E> p(precision);

        xstrided_slice_vector sv;
        detail::recurser_run(p, d, sv, lim);
        p.init();
        sv.clear();
        xoutput(out, d, sv, p, 1, p.width(), lim, static_cast<std::size_t>(po.line_width));

        out.precision(temp_precision);  // restore precision

        return out;
    }

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e)
    {
        return pretty_print(e, out);
    }
}
#endif

// Backward compatibility: include xmime.hpp in xio.hpp by default.

#ifdef __CLING__
#include "xmime.hpp"
#endif
