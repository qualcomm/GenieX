/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_FIXED_HPP
#define XTENSOR_FIXED_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <xtl/xsequence.hpp>

#include "../containers/xcontainer.hpp"
#include "../containers/xstorage.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"

namespace xtl
{
    namespace detail
    {
        template <class T, std::size_t N>
        struct sequence_builder<xt::const_array<T, N>>
        {
            using sequence_type = xt::const_array<T, N>;
            using value_type = typename sequence_type::value_type;
            using size_type = typename sequence_type::size_type;

            inline static sequence_type make(size_type /*size*/, value_type /*v*/)
            {
                return sequence_type();
            }
        };
    }
}

namespace xt
{

    /**********************
     * xfixed declaration *
     **********************/

    template <class ET, class S, layout_type L, bool SH, class Tag>
    class xfixed_container;

    namespace detail
    {
        /**************************************************************************************
           The following is something we can currently only dream about -- for when we drop
           support for a lot of the old compilers (e.g. GCC 4.9, MSVC 2017 ;)

        template <class T>
        constexpr std::size_t calculate_stride(T& shape, std::size_t idx, layout_type L)
        {
            if (shape[idx] == 1)
            {
                return std::size_t(0);
            }

            std::size_t data_size = 1;
            std::size_t stride = 1;
            if (L == layout_type::row_major)
            {
                // because we have a integer sequence that counts
                // from 0 to sz - 1, we need to "invert" idx here
                idx = shape.size() - idx;
                for (std::size_t i = idx; i != 0; --i)
                {
                    stride = data_size;
                    data_size = stride * shape[i - 1];
                }
            }
            else
            {
                for (std::size_t i = 0; i < idx + 1; ++i)
                {
                    stride = data_size;
                    data_size = stride * shape[i];
                }
            }
            return stride;
        }

        *****************************************************************************************/

        template <layout_type L, std::size_t I, std::size_t... X>
        struct calculate_stride;

        template <std::size_t I, std::size_t Y, std::size_t... X>
        struct calculate_stride<layout_type::column_major, I, Y, X...>
        {
            static constexpr std::ptrdiff_t value = Y
                                                    * calculate_stride<layout_type::column_major, I - 1, X...>::value;
        };

        template <std::size_t Y, std::size_t... X>
        struct calculate_stride<layout_type::column_major, 0, Y, X...>
        {
            static constexpr std::ptrdiff_t value = 1;
        };

        template <std::size_t I, std::size_t... X>
        struct calculate_stride_row_major
        {
            static constexpr std::ptrdiff_t value = at<sizeof...(X) - I, X...>::value
                                                    * calculate_stride_row_major<I - 1, X...>::value;
        };

        template <std::size_t... X>
        struct calculate_stride_row_major<0, X...>
        {
            static constexpr std::ptrdiff_t value = 1;
        };

        template <std::size_t I, std::size_t... X>
        struct calculate_stride<layout_type::row_major, I, X...>
        {
            static constexpr std::ptrdiff_t value = calculate_stride_row_major<sizeof...(X) - I - 1, X...>::value;
        };

        namespace workaround
        {
            template <layout_type L, size_t I, class SEQ>
            struct computed_strides;

            template <layout_type L, size_t I, size_t... X>
            struct computed_strides<L, I, std::index_sequence<X...>>
            {
                static constexpr std::ptrdiff_t value = calculate_stride<L, I, X...>::value;
            };

            template <layout_type L, size_t I, class SEQ>
            constexpr std::ptrdiff_t get_computed_strides(bool cond)
            {
                return cond ? 0 : computed_strides<L, I, SEQ>::value;
            }
        }

        template <layout_type L, class R, std::size_t... X, std::size_t... I>
        constexpr R get_strides_impl(const xt::fixed_shape<X...>& shape, std::index_sequence<I...>)
        {
            static_assert(
                (L == layout_type::row_major) || (L == layout_type::column_major),
                "Layout not supported for fixed array"
            );
#if (_MSC_VER >= 1910)
            using temp_type = std::index_sequence<X...>;
            return R({workaround::get_computed_strides<L, I, temp_type>(shape[I] == 1)...});
#else
            return R({shape[I] == 1 ? 0 : calculate_stride<L, I, X...>::value...});
#endif
        }

        template <class S, class T, std::size_t... I>
        constexpr T get_backstrides_impl(const S& shape, const T& strides, std::index_sequence<I...>)
        {
            return T({(strides[I] * std::ptrdiff_t(shape[I] - 1))...});
        }

        template <std::size_t... X>
        struct fixed_compute_size_impl;

        template <std::size_t Y, std::size_t... X>
        struct fixed_compute_size_impl<Y, X...>
        {
            static constexpr std::size_t value = Y * fixed_compute_size_impl<X...>::value;
        };

        template <std::size_t X>
        struct fixed_compute_size_impl<X>
        {
            static constexpr std::size_t value = X;
        };

        template <>
        struct fixed_compute_size_impl<>
        {
            // support for 0D xtensor fixed (empty shape = xshape<>)
            static constexpr std::size_t value = 1;
        };

        // TODO unify with constexpr compute_size when dropping MSVC 2015
        template <class T>
        struct fixed_compute_size;

        template <std::size_t... X>
        struct fixed_compute_size<xt::fixed_shape<X...>>
        {
            static constexpr std::size_t value = fixed_compute_size_impl<X...>::value;
        };

        template <class V, std::size_t... X>
        struct get_init_type_impl;

        template <class V, std::size_t Y>
        struct get_init_type_impl<V, Y>
        {
            using type = V[Y];
        };

        template <class V>
        struct get_init_type_impl<V>
        {
            using type = V[1];
        };

        template <class V, std::size_t Y, std::size_t... X>
        struct get_init_type_impl<V, Y, X...>
        {
            using tmp_type = typename get_init_type_impl<V, X...>::type;
            using type = tmp_type[Y];
        };
    }

    template <layout_type L, class R, std::size_t... X>
    constexpr R get_strides(const fixed_shape<X...>& shape) noexcept
    {
        return detail::get_strides_impl<L, R>(shape, std::make_index_sequence<sizeof...(X)>{});
    }

    template <class S, class T>
    constexpr T get_backstrides(const S& shape, const T& strides) noexcept
    {
        return detail::get_backstrides_impl(shape, strides, std::make_index_sequence<std::tuple_size<T>::value>{});
    }

    template <class V, class S>
    struct get_init_type;

    template <class V, std::size_t... X>
    struct get_init_type<V, fixed_shape<X...>>
    {
        using type = typename detail::get_init_type_impl<V, X...>::type;
    };

    template <class V, class S>
    using get_init_type_t = typename get_init_type<V, S>::type;

    template <class ET, class S, layout_type L, bool SH, class Tag>
    struct xcontainer_inner_types<xfixed_container<ET, S, L, SH, Tag>>
    {
        using shape_type = S;
        using inner_shape_type = typename S::cast_type;
        using strides_type = get_strides_t<inner_shape_type>;
        using inner_strides_type = strides_type;
        using backstrides_type = inner_strides_type;
        using inner_backstrides_type = backstrides_type;

        // NOTE: 0D (S::size() == 0) results in storage for 1 element (scalar)
#if defined(_MSC_VER) && _MSC_VER < 1910 && !defined(_WIN64)
        // WORKAROUND FOR MSVC 2015 32 bit, fallback to unaligned container for 0D scalar case
        using storage_type = std::array<ET, detail::fixed_compute_size<S>::value>;
#else
        using storage_type = aligned_array<ET, detail::fixed_compute_size<S>::value>;
#endif

        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using temporary_type = xfixed_container<ET, S, L, SH, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class ET, class S, layout_type L, bool SH, class Tag>
    struct xiterable_inner_types<xfixed_container<ET, S, L, SH, Tag>>
        : xcontainer_iterable_types<xfixed_container<ET, S, L, SH, Tag>>
    {
    };

    /**
     * @class xfixed_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xfixed_container class implements a dense multidimensional container
     * with tensor semantic and fixed dimension
     *
     * @tparam ET The type of the elements.
     * @tparam S The xshape template paramter of the container.
     * @tparam L The layout_type of the tensor.
     * @tparam SH Wether the tensor can be used as a shared expression.
     * @tparam Tag The expression tag.
     * @sa xtensor_fixed
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    class xfixed_container : public xcontainer<xfixed_container<ET, S, L, SH, Tag>>,
                             public xcontainer_semantic<xfixed_container<ET, S, L, SH, Tag>>
    {
    public:

        using self_type = xfixed_container<ET, S, L, SH, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;

        using storage_type = typename base_type::storage_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        static constexpr std::size_t N = std::tuple_size<shape_type>::value;
        static constexpr std::size_t rank = N;

        xfixed_container() = default;
        xfixed_container(const value_type& v);
        explicit xfixed_container(const inner_shape_type& shape, layout_type l = L);
        explicit xfixed_container(const inner_shape_type& shape, value_type v, layout_type l = L);

        template <class IX = std::integral_constant<std::size_t, N>>
        xfixed_container(nested_initializer_list_t<value_type, N> t)
            requires(IX::value != 0);

        ~xfixed_container() = default;

        xfixed_container(const xfixed_container&) = default;
        xfixed_container& operator=(const xfixed_container&) = default;

        xfixed_container(xfixed_container&&) = default;
        xfixed_container& operator=(xfixed_container&&) = default;

        template <class E>
        xfixed_container(const xexpression<E>& e);

        template <class E>
        xfixed_container& operator=(const xexpression<E>& e);

        template <class ST = std::array<std::size_t, N>>
        static xfixed_container from_shape(ST&& /*s*/);

        template <class ST = std::array<std::size_t, N>>
        void resize(ST&& shape, bool force = false) const;
        template <class ST = shape_type>
        void resize(ST&& shape, layout_type l) const;
        template <class ST = shape_type>
        void resize(ST&& shape, const strides_type& strides) const;

        template <class ST = std::array<std::size_t, N>>
        const auto& reshape(ST&& shape, layout_type layout = L) const;

        template <class ST>
        bool broadcast_shape(ST& s, bool reuse_cache = false) const;

        constexpr layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

    private:

        storage_type m_storage;

        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_shape_type m_shape = S();
        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_strides_type m_strides = get_strides<L, inner_strides_type>(S());
        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_backstrides_type
            m_backstrides = get_backstrides(m_shape, m_strides);

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        XTENSOR_CONSTEXPR_RETURN const inner_shape_type& shape_impl() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_strides_type& strides_impl() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_backstrides_type& backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_container<ET, S, L, SH, Tag>>;
    };

#ifdef XTENSOR_HAS_CONSTEXPR_ENHANCED
    // Out of line definitions to prevent linker errors prior to C++17
    template <class ET, class S, layout_type L, bool SH, class Tag>
    constexpr
        typename xfixed_container<ET, S, L, SH, Tag>::inner_shape_type xfixed_container<ET, S, L, SH, Tag>::m_shape;

    template <class ET, class S, layout_type L, bool SH, class Tag>
    constexpr
        typename xfixed_container<ET, S, L, SH, Tag>::inner_strides_type xfixed_container<ET, S, L, SH, Tag>::m_strides;

    template <class ET, class S, layout_type L, bool SH, class Tag>
    constexpr typename xfixed_container<ET, S, L, SH, Tag>::inner_backstrides_type
        xfixed_container<ET, S, L, SH, Tag>::m_backstrides;
#endif

    /****************************************
     * xfixed_container_adaptor declaration *
     ****************************************/

    template <class EC, class S, layout_type L, bool SH, class Tag>
    class xfixed_adaptor;

    template <class EC, class S, layout_type L, bool SH, class Tag>
    struct xcontainer_inner_types<xfixed_adaptor<EC, S, L, SH, Tag>>
    {
        using storage_type = std::remove_reference_t<EC>;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = S;
        using inner_shape_type = typename S::cast_type;
        using strides_type = get_strides_t<inner_shape_type>;
        using backstrides_type = strides_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xfixed_container<typename storage_type::value_type, S, L, SH, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, class S, layout_type L, bool SH, class Tag>
    struct xiterable_inner_types<xfixed_adaptor<EC, S, L, SH, Tag>>
        : xcontainer_iterable_types<xfixed_adaptor<EC, S, L, SH, Tag>>
    {
    };

    /**
     * @class xfixed_adaptor
     * @brief Dense multidimensional container adaptor with tensor semantic
     * and fixed dimension.
     *
     * The xfixed_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam S The xshape template parameter for the fixed shape of the adaptor
     * @tparam L The layout_type of the adaptor.
     * @tparam SH Wether the adaptor can be used as a shared expression.
     * @tparam Tag The expression tag.
     */
    template <class EC, class S, layout_type L, bool SH, class Tag>
    class xfixed_adaptor : public xcontainer<xfixed_adaptor<EC, S, L, SH, Tag>>,
                           public xcontainer_semantic<xfixed_adaptor<EC, S, L, SH, Tag>>
    {
    public:

        using container_closure_type = EC;

        using self_type = xfixed_adaptor<EC, S, L, SH, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using storage_type = typename base_type::storage_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        static constexpr std::size_t N = S::size();

        xfixed_adaptor(storage_type&& data);
        xfixed_adaptor(const storage_type& data);

        template <class D>
        xfixed_adaptor(D&& data);

        ~xfixed_adaptor() = default;

        xfixed_adaptor(const xfixed_adaptor&) = default;
        xfixed_adaptor& operator=(const xfixed_adaptor&);

        xfixed_adaptor(xfixed_adaptor&&) = default;
        xfixed_adaptor& operator=(xfixed_adaptor&&);
        xfixed_adaptor& operator=(temporary_type&&);

        template <class E>
        xfixed_adaptor& operator=(const xexpression<E>& e);

        template <class ST = std::array<std::size_t, N>>
        void resize(ST&& shape, bool force = false) const;
        template <class ST = shape_type>
        void resize(ST&& shape, layout_type l) const;
        template <class ST = shape_type>
        void resize(ST&& shape, const strides_type& strides) const;

        template <class ST = std::array<std::size_t, N>>
        const auto& reshape(ST&& shape, layout_type layout = L) const;

        template <class ST>
        bool broadcast_shape(ST& s, bool reuse_cache = false) const;

        constexpr layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

    private:

        container_closure_type m_storage;

        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_shape_type m_shape = S();
        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_strides_type m_strides = get_strides<L, inner_strides_type>(S());
        XTENSOR_CONSTEXPR_ENHANCED_STATIC inner_backstrides_type
            m_backstrides = get_backstrides(m_shape, m_strides);

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        XTENSOR_CONSTEXPR_RETURN const inner_shape_type& shape_impl() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_strides_type& strides_impl() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_backstrides_type& backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_adaptor<EC, S, L, SH, Tag>>;
    };

#ifdef XTENSOR_HAS_CONSTEXPR_ENHANCED
    // Out of line definitions to prevent linker errors prior to C++17
    template <class EC, class S, layout_type L, bool SH, class Tag>
    constexpr
        typename xfixed_adaptor<EC, S, L, SH, Tag>::inner_shape_type xfixed_adaptor<EC, S, L, SH, Tag>::m_shape;

    template <class EC, class S, layout_type L, bool SH, class Tag>
    constexpr
        typename xfixed_adaptor<EC, S, L, SH, Tag>::inner_strides_type xfixed_adaptor<EC, S, L, SH, Tag>::m_strides;

    template <class EC, class S, layout_type L, bool SH, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, SH, Tag>::inner_backstrides_type
        xfixed_adaptor<EC, S, L, SH, Tag>::m_backstrides;
#endif

    /************************************
     * xfixed_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{

    /**
     * Create an uninitialized xfixed_container.
     * Note this function is only provided for homogeneity, and the shape & layout argument is
     * disregarded (the template shape is always used).
     *
     * @param shape the shape of the xfixed_container (unused!)
     * @param l the layout_type of the xfixed_container (unused!)
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline xfixed_container<ET, S, L, SH, Tag>::xfixed_container(const inner_shape_type& shape, layout_type l)
    {
        (void) (shape);
        (void) (l);
        XTENSOR_ASSERT(shape.size() == N && std::equal(shape.begin(), shape.end(), m_shape.begin()));
        XTENSOR_ASSERT(L == l);
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline xfixed_container<ET, S, L, SH, Tag>::xfixed_container(const value_type& v)
    {
        if (this->size() != 1)
        {
            XTENSOR_THROW(std::runtime_error, "wrong shape for scalar assignment (has to be xshape<>).");
        }
        m_storage[0] = v;
    }

    /**
     * Create an xfixed_container, and initialize with the value of v.
     * Note, the shape argument to this function is only provided for homogeneity,
     * and the shape argument is disregarded (the template shape is always used).
     *
     * @param shape the shape of the xfixed_container (unused!)
     * @param v the fill value
     * @param l the layout_type of the xfixed_container (unused!)
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline xfixed_container<ET, S, L, SH, Tag>::xfixed_container(
        const inner_shape_type& shape,
        value_type v,
        layout_type l
    )
    {
        (void) (shape);
        (void) (l);
        XTENSOR_ASSERT(shape.size() == N && std::equal(shape.begin(), shape.end(), m_shape.begin()));
        XTENSOR_ASSERT(L == l);
        std::fill(m_storage.begin(), m_storage.end(), v);
    }

    namespace detail
    {
        template <std::size_t X>
        struct check_initializer_list_shape
        {
            template <class T, class S>
            static bool run(const T& t, const S& shape)
            {
                std::size_t IX = shape.size() - X;
                bool result = (shape[IX] == t.size());
                for (std::size_t i = 0; i < shape[IX]; ++i)
                {
                    result = result && check_initializer_list_shape<X - 1>::run(t.begin()[i], shape);
                }
                return result;
            }
        };

        template <>
        struct check_initializer_list_shape<0>
        {
            template <class T, class S>
            static bool run(const T& /*t*/, const S& /*shape*/)
            {
                return true;
            }
        };
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline xfixed_container<ET, S, L, SH, Tag> xfixed_container<ET, S, L, SH, Tag>::from_shape(ST&& shape)
    {
        (void) shape;
        self_type tmp;
        XTENSOR_ASSERT(shape.size() == N && std::equal(shape.begin(), shape.end(), tmp.shape().begin()));
        return tmp;
    }

    /**
     * Allocates an xfixed_container with shape S with values from a C array.
     * The type returned by get_init_type_t is raw C array ``value_type[X][Y][Z]`` for
     * ``xt::xshape<X, Y, Z>``. C arrays can be initialized with the initializer list syntax,
     * but the size is checked at compile time to prevent errors.
     * Note: for clang < 3.8 this is an initializer_list and the size is not checked at compile-or runtime.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class IX>
    inline xfixed_container<ET, S, L, SH, Tag>::xfixed_container(nested_initializer_list_t<value_type, N> t)
        requires(IX::value != 0)
    {
        XTENSOR_ASSERT_MSG(
            detail::check_initializer_list_shape<N>::run(t, this->shape()) == true,
            "initializer list shape does not match fixed shape"
        );
        constexpr auto tmp = layout_type::row_major;
        L == tmp ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<tmp>(), t);
    }

    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class E>
    inline xfixed_container<ET, S, L, SH, Tag>::xfixed_container(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class E>
    inline auto xfixed_container<ET, S, L, SH, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    /**
     * Note that the xfixed_container **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_container<ET, S, L, SH, Tag>::resize(ST&& shape, bool) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        XTENSOR_ASSERT(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size());
    }

    /**
     * Note that the xfixed_container **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_container<ET, S, L, SH, Tag>::resize(ST&& shape, layout_type l) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        (void) (l);
        XTENSOR_ASSERT(
            std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size() && L == l
        );
    }

    /**
     * Note that the xfixed_container **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_container<ET, S, L, SH, Tag>::resize(ST&& shape, const strides_type& strides) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        (void) (strides);
        XTENSOR_ASSERT(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size());
        XTENSOR_ASSERT(
            std::equal(strides.begin(), strides.end(), m_strides.begin()) && strides.size() == m_strides.size()
        );
    }

    /**
     * Note that the xfixed_container **cannot** be reshaped to a shape different from ``S``.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline const auto& xfixed_container<ET, S, L, SH, Tag>::reshape(ST&& shape, layout_type layout) const
    {
        if (!(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size()
              && layout == L))
        {
            XTENSOR_THROW(std::runtime_error, "Trying to reshape xtensor_fixed with different shape or layout.");
        }
        return *this;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline bool xfixed_container<ET, S, L, SH, Tag>::broadcast_shape(ST& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    constexpr layout_type xfixed_container<ET, S, L, SH, Tag>::layout() const noexcept
    {
        return base_type::static_layout;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline bool xfixed_container<ET, S, L, SH, Tag>::is_contiguous() const noexcept
    {
        using str_type = typename inner_strides_type::value_type;
        return m_strides.empty() || (layout() == layout_type::row_major && m_strides.back() == str_type(1))
               || (layout() == layout_type::column_major && m_strides.front() == str_type(1));
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_container<ET, S, L, SH, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_container<ET, S, L, SH, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_container<ET, S, L, SH, Tag>::shape_impl() const noexcept
        -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_container<ET, S, L, SH, Tag>::strides_impl() const noexcept
        -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_container<ET, S, L, SH, Tag>::backstrides_impl() const noexcept
        -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /*******************
     * xfixed_adaptor *
     *******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xfixed_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline xfixed_adaptor<EC, S, L, SH, Tag>::xfixed_adaptor(storage_type&& data)
        : base_type()
        , m_storage(std::move(data))
    {
    }

    /**
     * Constructs an xfixed_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline xfixed_adaptor<EC, S, L, SH, Tag>::xfixed_adaptor(const storage_type& data)
        : base_type()
        , m_storage(data)
    {
    }

    /**
     * Constructs an xfixed_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param data the container to adapt
     */
    template <class EC, class S, layout_type L, bool SH, class Tag>
    template <class D>
    inline xfixed_adaptor<EC, S, L, SH, Tag>::xfixed_adaptor(D&& data)
        : base_type()
        , m_storage(std::forward<D>(data))
    {
    }

    //@}

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::operator=(const xfixed_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::operator=(xfixed_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::operator=(temporary_type&& rhs) -> self_type&
    {
        m_storage.resize(rhs.storage().size());
        std::copy(rhs.storage().cbegin(), rhs.storage().cend(), m_storage.begin());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, class S, layout_type L, bool SH, class Tag>
    template <class E>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    /**
     * Note that the xfixed_adaptor **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_adaptor<ET, S, L, SH, Tag>::resize(ST&& shape, bool) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        XTENSOR_ASSERT(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size());
    }

    /**
     * Note that the xfixed_adaptor **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_adaptor<ET, S, L, SH, Tag>::resize(ST&& shape, layout_type l) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        (void) (l);
        XTENSOR_ASSERT(
            std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size() && L == l
        );
    }

    /**
     * Note that the xfixed_adaptor **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline void xfixed_adaptor<ET, S, L, SH, Tag>::resize(ST&& shape, const strides_type& strides) const
    {
        (void) (shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        (void) (strides);
        XTENSOR_ASSERT(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size());
        XTENSOR_ASSERT(
            std::equal(strides.begin(), strides.end(), m_strides.begin()) && strides.size() == m_strides.size()
        );
    }

    /**
     * Note that the xfixed_container **cannot** be reshaped to a shape different from ``S``.
     */
    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline const auto& xfixed_adaptor<ET, S, L, SH, Tag>::reshape(ST&& shape, layout_type layout) const
    {
        if (!(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size()
              && layout == L))
        {
            XTENSOR_THROW(std::runtime_error, "Trying to reshape xtensor_fixed with different shape or layout.");
        }
        return *this;
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    template <class ST>
    inline bool xfixed_adaptor<ET, S, L, SH, Tag>::broadcast_shape(ST& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline auto xfixed_adaptor<EC, S, L, SH, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    constexpr layout_type xfixed_adaptor<EC, S, L, SH, Tag>::layout() const noexcept
    {
        return base_type::static_layout;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    inline bool xfixed_adaptor<EC, S, L, SH, Tag>::is_contiguous() const noexcept
    {
        using str_type = typename inner_strides_type::value_type;
        return m_strides.empty() || (layout() == layout_type::row_major && m_strides.back() == str_type(1))
               || (layout() == layout_type::column_major && m_strides.front() == str_type(1));
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, SH, Tag>::shape_impl() const noexcept
        -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, SH, Tag>::strides_impl() const noexcept
        -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class EC, class S, layout_type L, bool SH, class Tag>
    XTENSOR_CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, SH, Tag>::backstrides_impl() const noexcept
        -> const inner_backstrides_type&
    {
        return m_backstrides;
    }
}

#endif
