/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SLICE_HPP
#define XTENSOR_SLICE_HPP

#include <cstddef>
#include <map>
#include <type_traits>
#include <utility>

#include <xtl/xtype_traits.hpp>

#include "../containers/xstorage.hpp"
#include "../core/xtensor_config.hpp"
#include "../utils/xutils.hpp"

#ifndef XTENSOR_CONSTEXPR
#if (defined(_MSC_VER) || __GNUC__ < 8)
#define XTENSOR_CONSTEXPR inline
#define XTENSOR_GLOBAL_CONSTEXPR static const
#else
#define XTENSOR_CONSTEXPR constexpr
#define XTENSOR_GLOBAL_CONSTEXPR constexpr
#endif
#endif

namespace xt
{

    /**********************
     * xslice declaration *
     **********************/

    template <class D>
    class xslice
    {
    public:

        using derived_type = D;

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

    protected:

        xslice() = default;
        ~xslice() = default;

        xslice(const xslice&) = default;
        xslice& operator=(const xslice&) = default;

        xslice(xslice&&) = default;
        xslice& operator=(xslice&&) = default;
    };

    template <class S>
    using is_xslice = std::is_base_of<xslice<S>, S>;

    template <class E, class R = void>
    using disable_xslice = typename std::enable_if<!is_xslice<E>::value, R>::type;

    template <class... E>
    using has_xslice = std::disjunction<is_xslice<E>...>;

    /**************
     * slice tags *
     **************/

#define DEFINE_TAG_CONVERSION(NAME)                 \
    template <class T>                              \
    XTENSOR_CONSTEXPR NAME convert() const noexcept \
    {                                               \
        return NAME();                              \
    }

    struct xall_tag
    {
        DEFINE_TAG_CONVERSION(xall_tag)
    };

    struct xnewaxis_tag
    {
        DEFINE_TAG_CONVERSION(xnewaxis_tag)
    };

    struct xellipsis_tag
    {
        DEFINE_TAG_CONVERSION(xellipsis_tag)
    };

#undef DEFINE_TAG_CONVERSION

    /**********************
     * xrange declaration *
     **********************/

    template <class T>
    class xrange : public xslice<xrange<T>>
    {
    public:

        using size_type = T;
        using self_type = xrange<T>;

        xrange() = default;
        xrange(size_type start_val, size_type stop_val) noexcept;

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xrange<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xrange<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_start;
        size_type m_size;

        template <class S>
        friend class xrange;
    };

    /******************************
     * xstepped_range declaration *
     ******************************/

    template <class T>
    class xstepped_range : public xslice<xstepped_range<T>>
    {
    public:

        using size_type = T;
        using self_type = xstepped_range<T>;

        xstepped_range() = default;
        xstepped_range(size_type start_val, size_type stop_val, size_type step) noexcept;

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xstepped_range<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xstepped_range<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_start;
        size_type m_size;
        size_type m_step;

        template <class S>
        friend class xstepped_range;
    };

    /********************
     * xall declaration *
     ********************/

    template <class T>
    class xall : public xslice<xall<T>>
    {
    public:

        using size_type = T;
        using self_type = xall<T>;

        xall() = default;
        explicit xall(size_type size) noexcept;

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xall<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xall<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_size;
    };

    /**
     * Returns a slice representing a full dimension,
     * to be used as an argument of view function.
     * @sa view, strided_view
     */
    inline auto all() noexcept
    {
        return xall_tag();
    }

    /**
     * Returns a slice representing all remaining dimensions,
     * and selecting all in these dimensions. Ellipsis will expand
     * to a series of `all()` slices, until the number of slices is
     * equal to the number of dimensions of the source array.
     *
     * Note: ellipsis can only be used in strided_view!
     *
     * @code{.cpp}
     * xarray<double> a = xarray<double>::from_shape({5, 5, 1, 1, 5});
     * auto v = xt::strided_view(a, {2, xt::ellipsis(), 2});
     * // equivalent to using {2, xt::all(), xt::all(), xt::all(), 2};
     * @endcode
     *
     * @sa strided_view
     */
    inline auto ellipsis() noexcept
    {
        return xellipsis_tag();
    }

    /************************
     * xnewaxis declaration *
     ************************/

    template <class T>
    class xnewaxis : public xslice<xnewaxis<T>>
    {
    public:

        using size_type = T;
        using self_type = xnewaxis<T>;

        xnewaxis() = default;

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xnewaxis<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xnewaxis<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;
    };

    /**
     * Returns a slice representing a new axis of length one,
     * to be used as an argument of view function.
     * @sa view, strided_view
     */
    inline auto newaxis() noexcept
    {
        return xnewaxis_tag();
    }

    /***************************
     * xkeep_slice declaration *
     ***************************/

    template <class T>
    class xkeep_slice;

    namespace detail
    {
        template <class T>
        struct is_xkeep_slice : std::false_type
        {
        };

        template <class T>
        struct is_xkeep_slice<xkeep_slice<T>> : std::true_type
        {
        };

        template <class T>
        using disable_xkeep_slice_t = std::enable_if_t<!is_xkeep_slice<std::decay_t<T>>::value, void>;

        template <class T>
        using enable_xkeep_slice_t = std::enable_if_t<is_xkeep_slice<std::decay_t<T>>::value, void>;
    }

    template <class T>
    class xkeep_slice : public xslice<xkeep_slice<T>>
    {
    public:

        using container_type = svector<T>;
        using size_type = typename container_type::value_type;
        using self_type = xkeep_slice<T>;

        template <class C, typename = detail::disable_xkeep_slice_t<C>>
        explicit xkeep_slice(C& cont);
        explicit xkeep_slice(container_type&& cont);

        template <class S>
        xkeep_slice(std::initializer_list<S> t);

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xkeep_slice<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xkeep_slice<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;
        size_type size() const noexcept;

        void normalize(std::size_t s);

        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        xkeep_slice() = default;

        container_type m_indices;
        container_type m_raw_indices;

        template <class S>
        friend class xkeep_slice;
    };

    namespace detail
    {
        template <class T>
        using disable_integral_keep = std::enable_if_t<
            !xtl::is_integral<std::decay_t<T>>::value,
            xkeep_slice<typename std::decay_t<T>::value_type>>;

        template <class T, class R>
        using enable_integral_keep = std::enable_if_t<xtl::is_integral<T>::value, xkeep_slice<R>>;
    }

    /**
     * Create a non-contigous slice from a container of indices to keep.
     * Note: this slice cannot be used in the xstrided_view!
     *
     * @code{.cpp}
     * xt::xarray<double> a = xt::arange(9);
     * a.reshape({3, 3});
     * xt::view(a, xt::keep(0, 2); // => {{0, 1, 2}, {6, 7, 8}}
     * xt::view(a, xt::keep(1, 1, 1); // => {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}}
     * @endcode
     *
     * @param indices The indices container
     * @return instance of xkeep_slice
     */
    template <class T>
    inline detail::disable_integral_keep<T> keep(T&& indices)
    {
        return xkeep_slice<typename std::decay_t<T>::value_type>(std::forward<T>(indices));
    }

    template <class R = std::ptrdiff_t, class T>
    inline detail::enable_integral_keep<T, R> keep(T i)
    {
        using slice_type = xkeep_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = {static_cast<R>(i)};
        return slice_type(std::move(tmp));
    }

    template <class R = std::ptrdiff_t, class Arg0, class Arg1, class... Args>
    inline xkeep_slice<R> keep(Arg0 i0, Arg1 i1, Args... args)
    {
        using slice_type = xkeep_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = {static_cast<R>(i0), static_cast<R>(i1), static_cast<R>(args)...};
        return slice_type(std::move(tmp));
    }

    /***************************
     * xdrop_slice declaration *
     ***************************/

    template <class T>
    class xdrop_slice;

    namespace detail
    {
        template <class T>
        struct is_xdrop_slice : std::false_type
        {
        };

        template <class T>
        struct is_xdrop_slice<xdrop_slice<T>> : std::true_type
        {
        };

        template <class T>
        using disable_xdrop_slice_t = std::enable_if_t<!is_xdrop_slice<std::decay_t<T>>::value, void>;

        template <class T>
        using enable_xdrop_slice_t = std::enable_if_t<is_xdrop_slice<std::decay_t<T>>::value, void>;
    }

    template <class T>
    class xdrop_slice : public xslice<xdrop_slice<T>>
    {
    public:

        using container_type = svector<T>;
        using size_type = typename container_type::value_type;
        using self_type = xdrop_slice<T>;

        template <class C, typename = detail::disable_xdrop_slice_t<C>>
        explicit xdrop_slice(C& cont);
        explicit xdrop_slice(container_type&& cont);

        template <class S>
        xdrop_slice(std::initializer_list<S> t);

        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        operator xdrop_slice<S>() const noexcept;

        // Same as implicit conversion operator but more convenient to call
        // from a variant visitor
        template <class S, typename = std::enable_if_t<std::is_convertible<S, T>::value, void>>
        xdrop_slice<S> convert() const noexcept;

        size_type operator()(size_type i) const noexcept;
        size_type size() const noexcept;

        void normalize(std::size_t s);

        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        xdrop_slice() = default;

        container_type m_indices;
        container_type m_raw_indices;
        std::map<size_type, size_type> m_inc;
        size_type m_size;

        template <class S>
        friend class xdrop_slice;
    };

    namespace detail
    {
        template <class T>
        using disable_integral_drop = std::enable_if_t<
            !xtl::is_integral<std::decay_t<T>>::value,
            xdrop_slice<typename std::decay_t<T>::value_type>>;

        template <class T, class R>
        using enable_integral_drop = std::enable_if_t<xtl::is_integral<T>::value, xdrop_slice<R>>;
    }

    /**
     * Create a non-contigous slice from a container of indices to drop.
     * Note: this slice cannot be used in the xstrided_view!
     *
     * @code{.cpp}
     * xt::xarray<double> a = xt::arange(9);
     * a.reshape({3, 3});
     * xt::view(a, xt::drop(0, 2); // => {{3, 4, 5}}
     * @endcode
     *
     * @param indices The container of indices to drop
     * @return instance of xdrop_slice
     */
    template <class T>
    inline detail::disable_integral_drop<T> drop(T&& indices)
    {
        return xdrop_slice<typename std::decay_t<T>::value_type>(std::forward<T>(indices));
    }

    template <class R = std::ptrdiff_t, class T>
    inline detail::enable_integral_drop<T, R> drop(T i)
    {
        using slice_type = xdrop_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = {static_cast<R>(i)};
        return slice_type(std::move(tmp));
    }

    template <class R = std::ptrdiff_t, class Arg0, class Arg1, class... Args>
    inline xdrop_slice<R> drop(Arg0 i0, Arg1 i1, Args... args)
    {
        using slice_type = xdrop_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = {static_cast<R>(i0), static_cast<R>(i1), static_cast<R>(args)...};
        return slice_type(std::move(tmp));
    }

    /******************************
     * xrange_adaptor declaration *
     ******************************/

    template <class A, class B = A, class C = A>
    struct xrange_adaptor
    {
        xrange_adaptor(A start_val, B stop_val, C step)
            : m_start(start_val)
            , m_stop(stop_val)
            , m_step(step)
        {
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            xtl::is_integral<MI>::value && xtl::is_integral<MA>::value && xtl::is_integral<STEP>::value,
            xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return get_stepped_range(m_start, m_stop, m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            !xtl::is_integral<MI>::value && xtl::is_integral<MA>::value && xtl::is_integral<STEP>::value,
            xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return get_stepped_range(m_step > 0 ? 0 : static_cast<std::ptrdiff_t>(size) - 1, m_stop, m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            xtl::is_integral<MI>::value && !xtl::is_integral<MA>::value && xtl::is_integral<STEP>::value,
            xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            auto sz = static_cast<std::ptrdiff_t>(size);
            return get_stepped_range(m_start, m_step > 0 ? sz : -(sz + 1), m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            xtl::is_integral<MI>::value && xtl::is_integral<MA>::value && !xtl::is_integral<STEP>::value,
            xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(normalize(m_start, size), normalize(m_stop, size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            !xtl::is_integral<MI>::value && !xtl::is_integral<MA>::value && xtl::is_integral<STEP>::value,
            xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            std::ptrdiff_t start = m_step >= 0 ? 0 : static_cast<std::ptrdiff_t>(size) - 1;
            std::ptrdiff_t stop = m_step >= 0 ? static_cast<std::ptrdiff_t>(size) : -1;
            return xstepped_range<std::ptrdiff_t>(start, stop, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            xtl::is_integral<MI>::value && !xtl::is_integral<MA>::value && !xtl::is_integral<STEP>::value,
            xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(normalize(m_start, size), static_cast<std::ptrdiff_t>(size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            !xtl::is_integral<MI>::value && xtl::is_integral<MA>::value && !xtl::is_integral<STEP>::value,
            xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(0, normalize(m_stop, size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<
            !xtl::is_integral<MI>::value && !xtl::is_integral<MA>::value && !xtl::is_integral<STEP>::value,
            xall<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xall<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(size));
        }

        A start() const
        {
            return m_start;
        }

        B stop() const
        {
            return m_stop;
        }

        C step() const
        {
            return m_step;
        }

    private:

        static auto normalize(std::ptrdiff_t val, std::size_t ssize)
        {
            std::ptrdiff_t size = static_cast<std::ptrdiff_t>(ssize);
            val = (val >= 0) ? val : val + size;
            return (std::max)(std::ptrdiff_t(0), (std::min)(size, val));
        }

        static auto
        get_stepped_range(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step, std::size_t ssize)
        {
            std::ptrdiff_t size = static_cast<std::ptrdiff_t>(ssize);
            start = (start >= 0) ? start : start + size;
            stop = (stop >= 0) ? stop : stop + size;

            if (step > 0)
            {
                start = (std::max)(std::ptrdiff_t(0), (std::min)(size, start));
                stop = (std::max)(std::ptrdiff_t(0), (std::min)(size, stop));
            }
            else
            {
                start = (std::max)(std::ptrdiff_t(-1), (std::min)(size - 1, start));
                stop = (std::max)(std::ptrdiff_t(-1), (std::min)(size - 1, stop));
            }

            return xstepped_range<std::ptrdiff_t>(start, stop, step);
        }

        A m_start;
        B m_stop;
        C m_step;
    };

    /*******************************
     * Placeholders and rangemaker *
     *******************************/

    namespace placeholders
    {
        // xtensor universal placeholder
        struct xtuph
        {
        };

        template <class... Args>
        struct rangemaker
        {
            std::ptrdiff_t rng[3];  // = { 0, 0, 0 };
        };

        XTENSOR_CONSTEXPR xtuph get_tuph_or_val(std::ptrdiff_t /*val*/, std::true_type)
        {
            return xtuph();
        }

        XTENSOR_CONSTEXPR std::ptrdiff_t get_tuph_or_val(std::ptrdiff_t val, std::false_type)
        {
            return val;
        }

        template <class A, class B, class C>
        struct rangemaker<A, B, C>
        {
            XTENSOR_CONSTEXPR operator xrange_adaptor<A, B, C>()
            {
                return xrange_adaptor<A, B, C>(
                    {get_tuph_or_val(rng[0], std::is_same<A, xtuph>()),
                     get_tuph_or_val(rng[1], std::is_same<B, xtuph>()),
                     get_tuph_or_val(rng[2], std::is_same<C, xtuph>())}
                );
            }

            std::ptrdiff_t rng[3];  // = { 0, 0, 0 };
        };

        template <class A, class B>
        struct rangemaker<A, B>
        {
            XTENSOR_CONSTEXPR operator xrange_adaptor<A, B, xt::placeholders::xtuph>()
            {
                return xrange_adaptor<A, B, xt::placeholders::xtuph>(
                    {get_tuph_or_val(rng[0], std::is_same<A, xtuph>()),
                     get_tuph_or_val(rng[1], std::is_same<B, xtuph>()),
                     xtuph()}
                );
            }

            std::ptrdiff_t rng[3];  // = { 0, 0, 0 };
        };

        template <class... OA>
        XTENSOR_CONSTEXPR auto operator|(const rangemaker<OA...>& rng, const std::ptrdiff_t& t)
        {
            auto nrng = rangemaker<OA..., std::ptrdiff_t>({rng.rng[0], rng.rng[1], rng.rng[2]});
            nrng.rng[sizeof...(OA)] = t;
            return nrng;
        }

        template <class... OA>
        XTENSOR_CONSTEXPR auto operator|(const rangemaker<OA...>& rng, const xt::placeholders::xtuph& /*t*/)
        {
            auto nrng = rangemaker<OA..., xt::placeholders::xtuph>({rng.rng[0], rng.rng[1], rng.rng[2]});
            return nrng;
        }

        XTENSOR_GLOBAL_CONSTEXPR xtuph _{};
        XTENSOR_GLOBAL_CONSTEXPR rangemaker<> _r = rangemaker<>({0, 0, 0});
        XTENSOR_GLOBAL_CONSTEXPR xall_tag _a{};
        XTENSOR_GLOBAL_CONSTEXPR xnewaxis_tag _n{};
        XTENSOR_GLOBAL_CONSTEXPR xellipsis_tag _e{};
    }

    inline auto xnone()
    {
        return placeholders::xtuph();
    }

    namespace detail
    {
        template <class T, class E = void>
        struct cast_if_integer
        {
            using type = T;

            type operator()(T t)
            {
                return t;
            }
        };

        template <class T>
        struct cast_if_integer<T, std::enable_if_t<xtl::is_integral<T>::value>>
        {
            using type = std::ptrdiff_t;

            type operator()(T t)
            {
                return static_cast<type>(t);
            }
        };

        template <class T>
        using cast_if_integer_t = typename cast_if_integer<T>::type;
    }

    /**
     * Select a range from start_val to stop_val (excluded).
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * @code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     *
     * range(3, _)  // select from index 3 to the end
     * range(_, 5)  // select from index 0 to 5 (excluded)
     * range(_, _)  // equivalent to `all()`
     * @endcode
     *
     * @sa view, strided_view
     */
    template <class A, class B>
    inline auto range(A start_val, B stop_val)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, placeholders::xtuph>(
            detail::cast_if_integer<A>{}(start_val),
            detail::cast_if_integer<B>{}(stop_val),
            placeholders::xtuph()
        );
    }

    /**
     * Select a range from start_val to stop_val (excluded) with step
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * @code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     * range(3, _, 5)  // select from index 3 to the end with stepsize 5
     * @endcode
     *
     * @sa view, strided_view
     */
    template <class A, class B, class C>
    inline auto range(A start_val, B stop_val, C step)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, detail::cast_if_integer_t<C>>(
            detail::cast_if_integer<A>{}(start_val),
            detail::cast_if_integer<B>{}(stop_val),
            detail::cast_if_integer<C>{}(step)
        );
    }

    /******************************************************
     * homogeneous get_size for integral types and slices *
     ******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> get_size(const S&) noexcept
    {
        return 1;
    }

    template <class S>
    inline auto get_size(const xslice<S>& slice) noexcept
    {
        return slice.derived_cast().size();
    }

    /*******************************************************
     * homogeneous step_size for integral types and slices *
     *******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> step_size(const S&, std::size_t) noexcept
    {
        return 0;
    }

    template <class S>
    inline disable_xslice<S, std::size_t> step_size(const S&, std::size_t, std::size_t) noexcept
    {
        return 0;
    }

    template <class S>
    inline auto step_size(const xslice<S>& slice, std::size_t idx) noexcept
    {
        return slice.derived_cast().step_size(idx);
    }

    template <class S>
    inline auto step_size(const xslice<S>& slice, std::size_t idx, std::size_t n) noexcept
    {
        return slice.derived_cast().step_size(idx, n);
    }

    /*********************************************
     * homogeneous value for integral and slices *
     *********************************************/

    template <class S, class I>
    inline disable_xslice<S, std::size_t> value(const S& s, I) noexcept
    {
        return static_cast<std::size_t>(s);
    }

    template <class S, class I>
    inline auto value(const xslice<S>& slice, I i) noexcept
    {
        using ST = typename S::size_type;
        return slice.derived_cast()(static_cast<ST>(i));
    }

    /****************************************
     * homogeneous get_slice_implementation *
     ****************************************/

    namespace detail
    {
        template <class T>
        struct slice_implementation_getter
        {
            template <class E, class SL>
            inline decltype(auto) operator()(E& e, SL&& slice, std::size_t index) const
            {
                return get_slice(e, std::forward<SL>(slice), index, xtl::is_signed<std::decay_t<SL>>());
            }

        private:

            template <class E, class SL>
            inline decltype(auto) get_slice(E&, SL&& slice, std::size_t, std::false_type) const
            {
                return std::forward<SL>(slice);
            }

            template <class E, class SL>
            inline decltype(auto) get_slice(E& e, SL&& slice, std::size_t index, std::true_type) const
            {
                using int_type = std::decay_t<SL>;
                return slice < int_type(0) ? slice + static_cast<std::ptrdiff_t>(e.shape(index))
                                           : std::ptrdiff_t(slice);
            }
        };

        struct keep_drop_getter
        {
            template <class E, class SL>
            inline decltype(auto) operator()(E& e, SL&& slice, std::size_t index) const
            {
                slice.normalize(e.shape()[index]);
                return std::forward<SL>(slice);
            }

            template <class E, class SL>
            inline auto operator()(E& e, const SL& slice, std::size_t index) const
            {
                return this->operator()(e, SL(slice), index);
            }
        };

        template <class T>
        struct slice_implementation_getter<xkeep_slice<T>> : keep_drop_getter
        {
        };

        template <class T>
        struct slice_implementation_getter<xdrop_slice<T>> : keep_drop_getter
        {
        };

        template <>
        struct slice_implementation_getter<xall_tag>
        {
            template <class E, class SL>
            inline auto operator()(E& e, SL&&, std::size_t index) const
            {
                return xall<typename E::size_type>(e.shape()[index]);
            }
        };

        template <>
        struct slice_implementation_getter<xnewaxis_tag>
        {
            template <class E, class SL>
            inline auto operator()(E&, SL&&, std::size_t) const
            {
                return xnewaxis<typename E::size_type>();
            }
        };

        template <class A, class B, class C>
        struct slice_implementation_getter<xrange_adaptor<A, B, C>>
        {
            template <class E, class SL>
            inline auto operator()(E& e, SL&& adaptor, std::size_t index) const
            {
                return adaptor.get(e.shape()[index]);
            }
        };
    }

    template <class E, class SL>
    inline auto get_slice_implementation(E& e, SL&& slice, std::size_t index)
    {
        detail::slice_implementation_getter<std::decay_t<SL>> getter;
        return getter(e, std::forward<SL>(slice), index);
    }

    /******************************
     * homogeneous get_slice_type *
     ******************************/

    namespace detail
    {
        template <class E, class SL>
        struct get_slice_type_impl
        {
            using type = SL;
        };

        template <class E>
        struct get_slice_type_impl<E, xall_tag>
        {
            using type = xall<typename E::size_type>;
        };

        template <class E>
        struct get_slice_type_impl<E, xnewaxis_tag>
        {
            using type = xnewaxis<typename E::size_type>;
        };

        template <class E, class A, class B, class C>
        struct get_slice_type_impl<E, xrange_adaptor<A, B, C>>
        {
            using type = decltype(xrange_adaptor<A, B, C>(A(), B(), C()).get(0));
        };
    }

    template <class E, class SL>
    using get_slice_type = typename detail::get_slice_type_impl<E, std::remove_reference_t<SL>>::type;

    /*************************
     * xslice implementation *
     *************************/

    template <class D>
    inline auto xslice<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xslice<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /*************************
     * xrange implementation *
     *************************/

    template <class T>
    inline xrange<T>::xrange(size_type start_val, size_type stop_val) noexcept
        : m_start(start_val)
        , m_size(stop_val > start_val ? stop_val - start_val : 0)
    {
    }

    template <class T>
    template <class S, typename>
    inline xrange<T>::operator xrange<S>() const noexcept
    {
        xrange<S> ret;
        ret.m_start = static_cast<S>(m_start);
        ret.m_size = static_cast<S>(m_size);
        return ret;
    }

    template <class T>
    template <class S, typename>
    inline xrange<S> xrange<T>::convert() const noexcept
    {
        return xrange<S>(*this);
    }

    template <class T>
    inline auto xrange<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_start + i;
    }

    template <class T>
    inline auto xrange<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xrange<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xrange<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return static_cast<size_type>(n);
    }

    template <class T>
    inline auto xrange<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i - m_start;
    }

    template <class T>
    inline bool xrange<T>::contains(size_type i) const noexcept
    {
        return i >= m_start && i < m_start + m_size;
    }

    template <class T>
    inline bool xrange<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_start == rhs.m_start) && (m_size == rhs.m_size);
    }

    template <class T>
    inline bool xrange<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /********************************
     * xtepped_range implementation *
     ********************************/

    template <class T>
    inline xstepped_range<T>::xstepped_range(size_type start_val, size_type stop_val, size_type step) noexcept
        : m_start(start_val)
        , m_size(size_type(0))
        , m_step(step)
    {
        size_type n = stop_val - start_val;
        m_size = n / step + (((n < 0) ^ (step > 0)) && (n % step));
    }

    template <class T>
    template <class S, typename>
    inline xstepped_range<T>::operator xstepped_range<S>() const noexcept
    {
        xstepped_range<S> ret;
        ret.m_start = static_cast<S>(m_start);
        ret.m_size = static_cast<S>(m_size);
        ret.m_step = static_cast<S>(m_step);
        return ret;
    }

    template <class T>
    template <class S, typename>
    inline xstepped_range<S> xstepped_range<T>::convert() const noexcept
    {
        return xstepped_range<S>(*this);
    }

    template <class T>
    inline auto xstepped_range<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_start + i * m_step;
    }

    template <class T>
    inline auto xstepped_range<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xstepped_range<T>::step_size() const noexcept -> size_type
    {
        return m_step;
    }

    template <class T>
    inline auto xstepped_range<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return m_step * static_cast<size_type>(n);
    }

    template <class T>
    inline auto xstepped_range<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return (i - m_start) / m_step;
    }

    template <class T>
    inline bool xstepped_range<T>::contains(size_type i) const noexcept
    {
        return i >= m_start && i < m_start + m_size * m_step && ((i - m_start) % m_step == 0);
    }

    template <class T>
    inline bool xstepped_range<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_start == rhs.m_start) && (m_size == rhs.m_size) && (m_step == rhs.m_step);
    }

    template <class T>
    inline bool xstepped_range<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /***********************
     * xall implementation *
     ***********************/

    template <class T>
    inline xall<T>::xall(size_type size) noexcept
        : m_size(size)
    {
    }

    template <class T>
    template <class S, typename>
    inline xall<T>::operator xall<S>() const noexcept
    {
        return xall<S>(static_cast<S>(m_size));
    }

    template <class T>
    template <class S, typename>
    inline xall<S> xall<T>::convert() const noexcept
    {
        return xall<S>(*this);
    }

    template <class T>
    inline auto xall<T>::operator()(size_type i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline auto xall<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xall<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xall<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return static_cast<size_type>(n);
    }

    template <class T>
    inline auto xall<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xall<T>::contains(size_type i) const noexcept
    {
        return i < m_size;
    }

    template <class T>
    inline bool xall<T>::operator==(const self_type& rhs) const noexcept
    {
        return m_size == rhs.m_size;
    }

    template <class T>
    inline bool xall<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /***************************
     * xnewaxis implementation *
     ***************************/

    template <class T>
    template <class S, typename>
    inline xnewaxis<T>::operator xnewaxis<S>() const noexcept
    {
        return xnewaxis<S>();
    }

    template <class T>
    template <class S, typename>
    inline xnewaxis<S> xnewaxis<T>::convert() const noexcept
    {
        return xnewaxis<S>(*this);
    }

    template <class T>
    inline auto xnewaxis<T>::operator()(size_type) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xnewaxis<T>::step_size() const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::step_size(std::size_t /*i*/, std::size_t /*n*/) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xnewaxis<T>::contains(size_type i) const noexcept
    {
        return i == 0;
    }

    template <class T>
    inline bool xnewaxis<T>::operator==(const self_type& /*rhs*/) const noexcept
    {
        return true;
    }

    template <class T>
    inline bool xnewaxis<T>::operator!=(const self_type& /*rhs*/) const noexcept
    {
        return true;
    }

    /******************************
     * xkeep_slice implementation *
     ******************************/

    template <class T>
    template <class C, typename>
    inline xkeep_slice<T>::xkeep_slice(C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    inline xkeep_slice<T>::xkeep_slice(container_type&& cont)
        : m_raw_indices(std::move(cont))
    {
    }

    template <class T>
    template <class S>
    inline xkeep_slice<T>::xkeep_slice(std::initializer_list<S> t)
        : m_raw_indices(t.size())
    {
        std::transform(
            t.begin(),
            t.end(),
            m_raw_indices.begin(),
            [](auto t)
            {
                return static_cast<size_type>(t);
            }
        );
    }

    template <class T>
    template <class S, typename>
    inline xkeep_slice<T>::operator xkeep_slice<S>() const noexcept
    {
        xkeep_slice<S> ret;
        using us_type = typename container_type::size_type;
        us_type sz = static_cast<us_type>(size());
        ret.m_raw_indices.resize(sz);
        ret.m_indices.resize(sz);
        std::transform(
            m_raw_indices.cbegin(),
            m_raw_indices.cend(),
            ret.m_raw_indices.begin(),
            [](const T& val)
            {
                return static_cast<S>(val);
            }
        );
        std::transform(
            m_indices.cbegin(),
            m_indices.cend(),
            ret.m_indices.begin(),
            [](const T& val)
            {
                return static_cast<S>(val);
            }
        );
        return ret;
    }

    template <class T>
    template <class S, typename>
    inline xkeep_slice<S> xkeep_slice<T>::convert() const noexcept
    {
        return xkeep_slice<S>(*this);
    }

    template <class T>
    inline void xkeep_slice<T>::normalize(std::size_t shape)
    {
        m_indices.resize(m_raw_indices.size());
        std::size_t sz = m_indices.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            m_indices[i] = m_raw_indices[i] < 0 ? static_cast<size_type>(shape) + m_raw_indices[i]
                                                : m_raw_indices[i];
        }
    }

    template <class T>
    inline auto xkeep_slice<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_indices.size() == size_type(1) ? m_indices.front() : m_indices[static_cast<std::size_t>(i)];
    }

    template <class T>
    inline auto xkeep_slice<T>::size() const noexcept -> size_type
    {
        return static_cast<size_type>(m_raw_indices.size());
    }

    template <class T>
    inline auto xkeep_slice<T>::step_size(std::size_t i, std::size_t n) const noexcept -> size_type
    {
        if (m_indices.size() == 1)
        {
            return 0;
        }
        if (i + n >= m_indices.size())
        {
            return m_indices.back() - m_indices[i] + 1;
        }
        else
        {
            return m_indices[i + n] - m_indices[i];
        }
    }

    template <class T>
    inline auto xkeep_slice<T>::revert_index(std::size_t i) const -> size_type
    {
        auto it = std::find(m_indices.begin(), m_indices.end(), i);
        if (it != m_indices.end())
        {
            return std::distance(m_indices.begin(), it);
        }
        else
        {
            XTENSOR_THROW(std::runtime_error, "Index i (" + std::to_string(i) + ") not in indices of islice.");
        }
    }

    template <class T>
    inline bool xkeep_slice<T>::contains(size_type i) const noexcept
    {
        return (std::find(m_indices.begin(), m_indices.end(), i) == m_indices.end()) ? false : true;
    }

    template <class T>
    inline bool xkeep_slice<T>::operator==(const self_type& rhs) const noexcept
    {
        return m_indices == rhs.m_indices;
    }

    template <class T>
    inline bool xkeep_slice<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /******************************
     * xdrop_slice implementation *
     ******************************/

    template <class T>
    template <class C, typename>
    inline xdrop_slice<T>::xdrop_slice(C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    inline xdrop_slice<T>::xdrop_slice(container_type&& cont)
        : m_raw_indices(std::move(cont))
    {
    }

    template <class T>
    template <class S>
    inline xdrop_slice<T>::xdrop_slice(std::initializer_list<S> t)
        : m_raw_indices(t.size())
    {
        std::transform(
            t.begin(),
            t.end(),
            m_raw_indices.begin(),
            [](auto t)
            {
                return static_cast<size_type>(t);
            }
        );
    }

    template <class T>
    template <class S, typename>
    inline xdrop_slice<T>::operator xdrop_slice<S>() const noexcept
    {
        xdrop_slice<S> ret;
        ret.m_raw_indices.resize(m_raw_indices.size());
        ret.m_indices.resize(m_indices.size());
        std::transform(
            m_raw_indices.cbegin(),
            m_raw_indices.cend(),
            ret.m_raw_indices.begin(),
            [](const T& val)
            {
                return static_cast<S>(val);
            }
        );
        std::transform(
            m_indices.cbegin(),
            m_indices.cend(),
            ret.m_indices.begin(),
            [](const T& val)
            {
                return static_cast<S>(val);
            }
        );
        std::transform(
            m_inc.cbegin(),
            m_inc.cend(),
            std::inserter(ret.m_inc, ret.m_inc.begin()),
            [](const auto& val)
            {
                return std::make_pair(static_cast<S>(val.first), static_cast<S>(val.second));
            }
        );
        ret.m_size = static_cast<S>(m_size);
        return ret;
    }

    template <class T>
    template <class S, typename>
    inline xdrop_slice<S> xdrop_slice<T>::convert() const noexcept
    {
        return xdrop_slice<S>(*this);
    }

    template <class T>
    inline void xdrop_slice<T>::normalize(std::size_t shape)
    {
        m_size = static_cast<size_type>(shape - m_raw_indices.size());

        m_indices.resize(m_raw_indices.size());
        std::size_t sz = m_indices.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            m_indices[i] = m_raw_indices[i] < 0 ? static_cast<size_type>(shape) + m_raw_indices[i]
                                                : m_raw_indices[i];
        }
        size_type cum = size_type(0);
        size_type prev_cum = cum;
        for (std::size_t i = 0; i < sz; ++i)
        {
            std::size_t ind = i;
            size_type d = m_indices[i];
            while (i + 1 < sz && m_indices[i + 1] == m_indices[i] + 1)
            {
                ++i;
            }
            cum += (static_cast<size_type>(i) - static_cast<size_type>(ind)) + 1;
            m_inc[d - prev_cum] = cum;
            prev_cum = cum;
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::operator()(size_type i) const noexcept -> size_type
    {
        if (m_inc.empty() || i < m_inc.begin()->first)
        {
            return i;
        }
        else
        {
            auto iter = --m_inc.upper_bound(i);
            return i + iter->second;
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xdrop_slice<T>::step_size(std::size_t i, std::size_t n) const noexcept -> size_type
    {
        if (i + n >= static_cast<std::size_t>(m_size))
        {
            return (*this)(static_cast<size_type>(m_size - 1)) - (*this)(static_cast<size_type>(i)) + 1;
        }
        else
        {
            return (*this)(static_cast<size_type>(i + n)) - (*this)(static_cast<size_type>(i));
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::revert_index(std::size_t i) const -> size_type
    {
        if (i < m_inc.begin()->first)
        {
            return i;
        }
        else
        {
            auto iter = --m_inc.lower_bound(i);
            auto check = iter->first + iter->second;
            if (check > i)
            {
                --iter;
            }
            return i - iter->second;
        }
    }

    template <class T>
    inline bool xdrop_slice<T>::contains(size_type i) const noexcept
    {
        return (std::find(m_indices.begin(), m_indices.end(), i) == m_indices.end()) ? true : false;
    }

    template <class T>
    inline bool xdrop_slice<T>::operator==(const self_type& rhs) const noexcept
    {
        return m_indices == rhs.m_indices;
    }

    template <class T>
    inline bool xdrop_slice<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }
}

#undef XTENSOR_CONSTEXPR

#endif
