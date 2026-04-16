/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ITERATOR_HPP
#define XTENSOR_ITERATOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

#include <xtl/xcompare.hpp>
#include <xtl/xiterator_base.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xsequence.hpp>

#include "../core/xlayout.hpp"
#include "../core/xshape.hpp"
#include "../utils/xexception.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /***********************
     * iterator meta utils *
     ***********************/

    template <class CT>
    class xscalar;

    template <bool is_const, class CT>
    class xscalar_stepper;

    namespace detail
    {
        template <class C>
        struct get_stepper_iterator_impl
        {
            using type = typename C::container_iterator;
        };

        template <class C>
        struct get_stepper_iterator_impl<const C>
        {
            using type = typename C::const_container_iterator;
        };

        template <class CT>
        struct get_stepper_iterator_impl<xscalar<CT>>
        {
            using type = typename xscalar<CT>::dummy_iterator;
        };

        template <class CT>
        struct get_stepper_iterator_impl<const xscalar<CT>>
        {
            using type = typename xscalar<CT>::const_dummy_iterator;
        };
    }

    template <class C>
    using get_stepper_iterator = typename detail::get_stepper_iterator_impl<C>::type;

    /********************************
     * xindex_type_t implementation *
     ********************************/

    namespace detail
    {
        template <class ST>
        struct index_type_impl
        {
            using type = dynamic_shape<typename ST::value_type>;
        };

        template <class V, std::size_t L>
        struct index_type_impl<std::array<V, L>>
        {
            using type = std::array<V, L>;
        };

        template <std::size_t... I>
        struct index_type_impl<fixed_shape<I...>>
        {
            using type = std::array<std::size_t, sizeof...(I)>;
        };
    }

    template <class C>
    using xindex_type_t = typename detail::index_type_impl<C>::type;

    /************
     * xstepper *
     ************/

    template <class C>
    class xstepper
    {
    public:

        using storage_type = C;
        using subiterator_type = get_stepper_iterator<C>;
        using subiterator_traits = std::iterator_traits<subiterator_type>;
        using value_type = typename subiterator_traits::value_type;
        using reference = typename subiterator_traits::reference;
        using pointer = typename subiterator_traits::pointer;
        using difference_type = typename subiterator_traits::difference_type;
        using size_type = typename storage_type::size_type;
        using shape_type = typename storage_type::shape_type;
        using simd_value_type = xt_simd::simd_type<value_type>;

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        xstepper() = default;
        xstepper(storage_type* c, subiterator_type it, size_type offset) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        template <class T>
        simd_return_type<T> step_simd();

        void step_leading();

        template <class R>
        void store_simd(const R& vec);

    private:

        storage_type* p_c;
        subiterator_type m_it;
        size_type m_offset;
    };

    template <layout_type L>
    struct stepper_tools
    {
        // For performance reasons, increment_stepper and decrement_stepper are
        // specialized for the case where n=1, which underlies operator++ and
        // operator-- on xiterators.

        template <class S, class IT, class ST>
        static void increment_stepper(S& stepper, IT& index, const ST& shape);

        template <class S, class IT, class ST>
        static void decrement_stepper(S& stepper, IT& index, const ST& shape);

        template <class S, class IT, class ST>
        static void increment_stepper(S& stepper, IT& index, const ST& shape, typename S::size_type n);

        template <class S, class IT, class ST>
        static void decrement_stepper(S& stepper, IT& index, const ST& shape, typename S::size_type n);
    };

    /********************
     * xindexed_stepper *
     ********************/

    template <class E, bool is_const>
    class xindexed_stepper
    {
    public:

        using self_type = xindexed_stepper<E, is_const>;
        using xexpression_type = std::conditional_t<is_const, const E, E>;

        using value_type = typename xexpression_type::value_type;
        using reference = std::
            conditional_t<is_const, typename xexpression_type::const_reference, typename xexpression_type::reference>;
        using pointer = std::
            conditional_t<is_const, typename xexpression_type::const_pointer, typename xexpression_type::pointer>;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xexpression_type::shape_type;
        using index_type = xindex_type_t<shape_type>;

        xindexed_stepper() = default;
        xindexed_stepper(xexpression_type* e, size_type offset, bool end = false) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

    private:

        xexpression_type* p_e;
        index_type m_index;
        size_type m_offset;
    };

    template <class T>
    struct is_indexed_stepper
    {
        static const bool value = false;
    };

    template <class T, bool B>
    struct is_indexed_stepper<xindexed_stepper<T, B>>
    {
        static const bool value = true;
    };

    template <class T, class R = T>
    struct enable_indexed_stepper : std::enable_if<is_indexed_stepper<T>::value, R>
    {
    };

    template <class T, class R = T>
    using enable_indexed_stepper_t = typename enable_indexed_stepper<T, R>::type;

    template <class T, class R = T>
    struct disable_indexed_stepper : std::enable_if<!is_indexed_stepper<T>::value, R>
    {
    };

    template <class T, class R = T>
    using disable_indexed_stepper_t = typename disable_indexed_stepper<T, R>::type;

    /*************
     * xiterator *
     *************/

    namespace detail
    {
        template <class S>
        class shape_storage
        {
        public:

            using shape_type = S;
            using param_type = const S&;

            shape_storage() = default;
            shape_storage(param_type shape);
            const S& shape() const;

        private:

            S m_shape;
        };

        template <class S>
        class shape_storage<S*>
        {
        public:

            using shape_type = S;
            using param_type = const S*;

            shape_storage(param_type shape = 0);
            const S& shape() const;

        private:

            const S* p_shape;
        };

        template <layout_type L>
        struct LAYOUT_FORBIDEN_FOR_XITERATOR;
    }

    template <class St, class S, layout_type L>
    class xiterator : public xtl::xrandom_access_iterator_base<
                          xiterator<St, S, L>,
                          typename St::value_type,
                          typename St::difference_type,
                          typename St::pointer,
                          typename St::reference>,
                      private detail::shape_storage<S>
    {
    public:

        using self_type = xiterator<St, S, L>;

        using stepper_type = St;
        using value_type = typename stepper_type::value_type;
        using reference = typename stepper_type::reference;
        using pointer = typename stepper_type::pointer;
        using difference_type = typename stepper_type::difference_type;
        using size_type = typename stepper_type::size_type;
        using iterator_category = std::random_access_iterator_tag;

        using private_base = detail::shape_storage<S>;
        using shape_type = typename private_base::shape_type;
        using shape_param_type = typename private_base::param_type;
        using index_type = xindex_type_t<shape_type>;

        xiterator() = default;

        // end_index means either reverse_iterator && !end or !reverse_iterator && end
        xiterator(St st, shape_param_type shape, bool end_index);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool equal(const xiterator& rhs) const;
        bool less_than(const xiterator& rhs) const;

    private:

        stepper_type m_st;
        index_type m_index;
        difference_type m_linear_index;

        using checking_type = typename detail::LAYOUT_FORBIDEN_FOR_XITERATOR<L>::type;
    };

    template <class St, class S, layout_type L>
    bool operator==(const xiterator<St, S, L>& lhs, const xiterator<St, S, L>& rhs);

    template <class St, class S, layout_type L>
    bool operator<(const xiterator<St, S, L>& lhs, const xiterator<St, S, L>& rhs);

    template <class St, class S, layout_type L>
    struct is_contiguous_container<xiterator<St, S, L>> : std::false_type
    {
    };

    /*********************
     * xbounded_iterator *
     *********************/

    template <class It, class BIt>
    class xbounded_iterator : public xtl::xrandom_access_iterator_base<
                                  xbounded_iterator<It, BIt>,
                                  typename std::iterator_traits<It>::value_type,
                                  typename std::iterator_traits<It>::difference_type,
                                  typename std::iterator_traits<It>::pointer,
                                  typename std::iterator_traits<It>::reference>
    {
    public:

        using self_type = xbounded_iterator<It, BIt>;

        using subiterator_type = It;
        using bound_iterator_type = BIt;
        using value_type = typename std::iterator_traits<It>::value_type;
        using reference = typename std::iterator_traits<It>::reference;
        using pointer = typename std::iterator_traits<It>::pointer;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        xbounded_iterator() = default;
        xbounded_iterator(It it, BIt bound_it);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        value_type operator*() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        subiterator_type m_it;
        bound_iterator_type m_bound_it;
    };

    template <class It, class BIt>
    bool operator==(const xbounded_iterator<It, BIt>& lhs, const xbounded_iterator<It, BIt>& rhs);

    template <class It, class BIt>
    bool operator<(const xbounded_iterator<It, BIt>& lhs, const xbounded_iterator<It, BIt>& rhs);

    /*****************************
     * linear_begin / linear_end *
     *****************************/

    namespace detail
    {
        template <class C, class = void_t<>>
        struct has_linear_iterator : std::false_type
        {
        };

        template <class C>
        struct has_linear_iterator<C, void_t<decltype(std::declval<C>().linear_cbegin())>> : std::true_type
        {
        };
    }

    template <class C>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(C& c) noexcept
    {
        if constexpr (detail::has_linear_iterator<C>::value)
        {
            return c.linear_begin();
        }
        else
        {
            return c.begin();
        }
    }

    template <class C>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(C& c) noexcept
    {
        if constexpr (detail::has_linear_iterator<C>::value)
        {
            return c.linear_end();
        }
        else
        {
            return c.end();
        }
    }

    template <class C>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(const C& c) noexcept
    {
        if constexpr (detail::has_linear_iterator<C>::value)
        {
            return c.linear_cbegin();
        }
        else
        {
            return c.cbegin();
        }
    }

    template <class C>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(const C& c) noexcept
    {
        if constexpr (detail::has_linear_iterator<C>::value)
        {
            return c.linear_cend();
        }
        else
        {
            return c.cend();
        }
    }

    /***************************
     * xstepper implementation *
     ***************************/

    template <class C>
    inline xstepper<C>::xstepper(storage_type* c, subiterator_type it, size_type offset) noexcept
        : p_c(c)
        , m_it(it)
        , m_offset(offset)
    {
    }

    template <class C>
    inline auto xstepper<C>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class C>
    inline void xstepper<C>::step(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            using strides_value_type = typename std::decay_t<decltype(p_c->strides())>::value_type;
            m_it += difference_type(static_cast<strides_value_type>(n) * p_c->strides()[dim - m_offset]);
        }
    }

    template <class C>
    inline void xstepper<C>::step_back(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            using strides_value_type = typename std::decay_t<decltype(p_c->strides())>::value_type;
            m_it -= difference_type(static_cast<strides_value_type>(n) * p_c->strides()[dim - m_offset]);
        }
    }

    template <class C>
    inline void xstepper<C>::reset(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_it -= difference_type(p_c->backstrides()[dim - m_offset]);
        }
    }

    template <class C>
    inline void xstepper<C>::reset_back(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_it += difference_type(p_c->backstrides()[dim - m_offset]);
        }
    }

    template <class C>
    inline void xstepper<C>::to_begin()
    {
        m_it = p_c->data_xbegin();
    }

    template <class C>
    inline void xstepper<C>::to_end(layout_type l)
    {
        m_it = p_c->data_xend(l, m_offset);
    }

    namespace detail
    {
        template <class It>
        struct step_simd_invoker
        {
            template <class R>
            static R apply(const It& it)
            {
                R reg;
                return reg.load_unaligned(&(*it));
                // return reg;
            }
        };

        template <bool is_const, class T, class S, layout_type L>
        struct step_simd_invoker<xiterator<xscalar_stepper<is_const, T>, S, L>>
        {
            template <class R>
            static R apply(const xiterator<xscalar_stepper<is_const, T>, S, L>& it)
            {
                return R(*it);
            }
        };
    }

    template <class C>
    template <class T>
    inline auto xstepper<C>::step_simd() -> simd_return_type<T>
    {
        using simd_type = simd_return_type<T>;
        simd_type reg = detail::step_simd_invoker<subiterator_type>::template apply<simd_type>(m_it);
        m_it += xt_simd::revert_simd_traits<simd_type>::size;
        return reg;
    }

    template <class C>
    template <class R>
    inline void xstepper<C>::store_simd(const R& vec)
    {
        vec.store_unaligned(&(*m_it));
        m_it += xt_simd::revert_simd_traits<R>::size;
        ;
    }

    template <class C>
    void xstepper<C>::step_leading()
    {
        ++m_it;
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::row_major>::increment_stepper(S& stepper, IT& index, const ST& shape)
    {
        using size_type = typename S::size_type;
        const size_type size = index.size();
        size_type i = size;
        while (i != 0)
        {
            --i;
            if (index[i] != shape[i] - 1)
            {
                ++index[i];
                stepper.step(i);
                return;
            }
            else
            {
                index[i] = 0;
                if (i != 0)
                {
                    stepper.reset(i);
                }
            }
        }
        if (i == 0)
        {
            if (size != size_type(0))
            {
                std::transform(
                    shape.cbegin(),
                    shape.cend() - 1,
                    index.begin(),
                    [](const auto& v)
                    {
                        return v - 1;
                    }
                );
                index[size - 1] = shape[size - 1];
            }
            stepper.to_end(layout_type::row_major);
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::row_major>::increment_stepper(
        S& stepper,
        IT& index,
        const ST& shape,
        typename S::size_type n
    )
    {
        using size_type = typename S::size_type;
        const size_type size = index.size();
        const size_type leading_i = size - 1;
        size_type i = size;
        while (i != 0 && n != 0)
        {
            --i;
            size_type inc = (i == leading_i) ? n : 1;
            if (xtl::cmp_less(index[i] + inc, shape[i]))
            {
                index[i] += inc;
                stepper.step(i, inc);
                n -= inc;
                if (i != leading_i || index.size() == 1)
                {
                    i = index.size();
                }
            }
            else
            {
                if (i == leading_i)
                {
                    size_type off = shape[i] - index[i] - 1;
                    stepper.step(i, off);
                    n -= off;
                }
                index[i] = 0;
                if (i != 0)
                {
                    stepper.reset(i);
                }
            }
        }
        if (i == 0 && n != 0)
        {
            if (size != size_type(0))
            {
                std::transform(
                    shape.cbegin(),
                    shape.cend() - 1,
                    index.begin(),
                    [](const auto& v)
                    {
                        return v - 1;
                    }
                );
                index[leading_i] = shape[leading_i];
            }
            stepper.to_end(layout_type::row_major);
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::row_major>::decrement_stepper(S& stepper, IT& index, const ST& shape)
    {
        using size_type = typename S::size_type;
        size_type i = index.size();
        while (i != 0)
        {
            --i;
            if (index[i] != 0)
            {
                --index[i];
                stepper.step_back(i);
                return;
            }
            else
            {
                index[i] = shape[i] - 1;
                if (i != 0)
                {
                    stepper.reset_back(i);
                }
            }
        }
        if (i == 0)
        {
            stepper.to_begin();
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::row_major>::decrement_stepper(
        S& stepper,
        IT& index,
        const ST& shape,
        typename S::size_type n
    )
    {
        using size_type = typename S::size_type;
        size_type i = index.size();
        size_type leading_i = index.size() - 1;
        while (i != 0 && n != 0)
        {
            --i;
            size_type inc = (i == leading_i) ? n : 1;
            if (xtl::cmp_greater_equal(index[i], inc))
            {
                index[i] -= inc;
                stepper.step_back(i, inc);
                n -= inc;
                if (i != leading_i || index.size() == 1)
                {
                    i = index.size();
                }
            }
            else
            {
                if (i == leading_i)
                {
                    size_type off = index[i];
                    stepper.step_back(i, off);
                    n -= off;
                }
                index[i] = shape[i] - 1;
                if (i != 0)
                {
                    stepper.reset_back(i);
                }
            }
        }
        if (i == 0 && n != 0)
        {
            stepper.to_begin();
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::column_major>::increment_stepper(S& stepper, IT& index, const ST& shape)
    {
        using size_type = typename S::size_type;
        const size_type size = index.size();
        size_type i = 0;
        while (i != size)
        {
            if (index[i] != shape[i] - 1)
            {
                ++index[i];
                stepper.step(i);
                return;
            }
            else
            {
                index[i] = 0;
                if (i != size - 1)
                {
                    stepper.reset(i);
                }
            }
            ++i;
        }
        if (i == size)
        {
            if (size != size_type(0))
            {
                std::transform(
                    shape.cbegin() + 1,
                    shape.cend(),
                    index.begin() + 1,
                    [](const auto& v)
                    {
                        return v - 1;
                    }
                );
                index[0] = shape[0];
            }
            stepper.to_end(layout_type::column_major);
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::column_major>::increment_stepper(
        S& stepper,
        IT& index,
        const ST& shape,
        typename S::size_type n
    )
    {
        using size_type = typename S::size_type;
        const size_type size = index.size();
        const size_type leading_i = 0;
        size_type i = 0;
        while (i != size && n != 0)
        {
            size_type inc = (i == leading_i) ? n : 1;
            if (index[i] + inc < shape[i])
            {
                index[i] += inc;
                stepper.step(i, inc);
                n -= inc;
                if (i != leading_i || size == 1)
                {
                    i = 0;
                    continue;
                }
            }
            else
            {
                if (i == leading_i)
                {
                    size_type off = shape[i] - index[i] - 1;
                    stepper.step(i, off);
                    n -= off;
                }
                index[i] = 0;
                if (i != size - 1)
                {
                    stepper.reset(i);
                }
            }
            ++i;
        }
        if (i == size && n != 0)
        {
            if (size != size_type(0))
            {
                std::transform(
                    shape.cbegin() + 1,
                    shape.cend(),
                    index.begin() + 1,
                    [](const auto& v)
                    {
                        return v - 1;
                    }
                );
                index[leading_i] = shape[leading_i];
            }
            stepper.to_end(layout_type::column_major);
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::column_major>::decrement_stepper(S& stepper, IT& index, const ST& shape)
    {
        using size_type = typename S::size_type;
        size_type size = index.size();
        size_type i = 0;
        while (i != size)
        {
            if (index[i] != 0)
            {
                --index[i];
                stepper.step_back(i);
                return;
            }
            else
            {
                index[i] = shape[i] - 1;
                if (i != size - 1)
                {
                    stepper.reset_back(i);
                }
            }
            ++i;
        }
        if (i == size)
        {
            stepper.to_begin();
        }
    }

    template <>
    template <class S, class IT, class ST>
    void stepper_tools<layout_type::column_major>::decrement_stepper(
        S& stepper,
        IT& index,
        const ST& shape,
        typename S::size_type n
    )
    {
        using size_type = typename S::size_type;
        size_type size = index.size();
        size_type i = 0;
        size_type leading_i = 0;
        while (i != size && n != 0)
        {
            size_type inc = (i == leading_i) ? n : 1;
            if (index[i] >= inc)
            {
                index[i] -= inc;
                stepper.step_back(i, inc);
                n -= inc;
                if (i != leading_i || index.size() == 1)
                {
                    i = 0;
                    continue;
                }
            }
            else
            {
                if (i == leading_i)
                {
                    size_type off = index[i];
                    stepper.step_back(i, off);
                    n -= off;
                }
                index[i] = shape[i] - 1;
                if (i != size - 1)
                {
                    stepper.reset_back(i);
                }
            }
            ++i;
        }
        if (i == size && n != 0)
        {
            stepper.to_begin();
        }
    }

    /***********************************
     * xindexed_stepper implementation *
     ***********************************/

    template <class C, bool is_const>
    inline xindexed_stepper<C, is_const>::xindexed_stepper(xexpression_type* e, size_type offset, bool end) noexcept
        : p_e(e)
        , m_index(xtl::make_sequence<index_type>(e->shape().size(), size_type(0)))
        , m_offset(offset)
    {
        if (end)
        {
            // Note: the layout here doesn't matter (unused) but using default traversal looks more "correct".
            to_end(XTENSOR_DEFAULT_TRAVERSAL);
        }
    }

    template <class C, bool is_const>
    inline auto xindexed_stepper<C, is_const>::operator*() const -> reference
    {
        return p_e->element(m_index.cbegin(), m_index.cend());
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::step(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_index[dim - m_offset] += static_cast<typename index_type::value_type>(n);
        }
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::step_back(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_index[dim - m_offset] -= static_cast<typename index_type::value_type>(n);
        }
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::reset(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_index[dim - m_offset] = 0;
        }
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::reset_back(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_index[dim - m_offset] = p_e->shape()[dim - m_offset] - 1;
        }
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::to_begin()
    {
        std::fill(m_index.begin(), m_index.end(), size_type(0));
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::to_end(layout_type l)
    {
        const auto& shape = p_e->shape();
        std::transform(
            shape.cbegin(),
            shape.cend(),
            m_index.begin(),
            [](const auto& v)
            {
                return v - 1;
            }
        );

        size_type l_dim = (l == layout_type::row_major) ? shape.size() - 1 : 0;
        m_index[l_dim] = shape[l_dim];
    }

    /****************************
     * xiterator implementation *
     ****************************/

    namespace detail
    {
        template <class S>
        inline shape_storage<S>::shape_storage(param_type shape)
            : m_shape(shape)
        {
        }

        template <class S>
        inline const S& shape_storage<S>::shape() const
        {
            return m_shape;
        }

        template <class S>
        inline shape_storage<S*>::shape_storage(param_type shape)
            : p_shape(shape)
        {
        }

        template <class S>
        inline const S& shape_storage<S*>::shape() const
        {
            return *p_shape;
        }

        template <>
        struct LAYOUT_FORBIDEN_FOR_XITERATOR<layout_type::row_major>
        {
            using type = int;
        };

        template <>
        struct LAYOUT_FORBIDEN_FOR_XITERATOR<layout_type::column_major>
        {
            using type = int;
        };
    }

    template <class St, class S, layout_type L>
    inline xiterator<St, S, L>::xiterator(St st, shape_param_type shape, bool end_index)
        : private_base(shape)
        , m_st(st)
        , m_index(
              end_index ? xtl::forward_sequence<index_type, const shape_type&>(this->shape())
                        : xtl::make_sequence<index_type>(this->shape().size(), size_type(0))
          )
        , m_linear_index(0)
    {
        // end_index means either reverse_iterator && !end or !reverse_iterator && end
        if (end_index)
        {
            if (m_index.size() != size_type(0))
            {
                auto iter_begin = (L == layout_type::row_major) ? m_index.begin() : m_index.begin() + 1;
                auto iter_end = (L == layout_type::row_major) ? m_index.end() - 1 : m_index.end();
                std::transform(
                    iter_begin,
                    iter_end,
                    iter_begin,
                    [](const auto& v)
                    {
                        return v - 1;
                    }
                );
            }
            m_linear_index = difference_type(std::accumulate(
                this->shape().cbegin(),
                this->shape().cend(),
                size_type(1),
                std::multiplies<size_type>()
            ));
        }
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator++() -> self_type&
    {
        stepper_tools<L>::increment_stepper(m_st, m_index, this->shape());
        ++m_linear_index;
        return *this;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator--() -> self_type&
    {
        stepper_tools<L>::decrement_stepper(m_st, m_index, this->shape());
        --m_linear_index;
        return *this;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator+=(difference_type n) -> self_type&
    {
        if (n >= 0)
        {
            stepper_tools<L>::increment_stepper(m_st, m_index, this->shape(), static_cast<size_type>(n));
        }
        else
        {
            stepper_tools<L>::decrement_stepper(m_st, m_index, this->shape(), static_cast<size_type>(-n));
        }
        m_linear_index += n;
        return *this;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator-=(difference_type n) -> self_type&
    {
        if (n >= 0)
        {
            stepper_tools<L>::decrement_stepper(m_st, m_index, this->shape(), static_cast<size_type>(n));
        }
        else
        {
            stepper_tools<L>::increment_stepper(m_st, m_index, this->shape(), static_cast<size_type>(-n));
        }
        m_linear_index -= n;
        return *this;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_linear_index - rhs.m_linear_index;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator*() const -> reference
    {
        return *m_st;
    }

    template <class St, class S, layout_type L>
    inline auto xiterator<St, S, L>::operator->() const -> pointer
    {
        return &(*m_st);
    }

    template <class St, class S, layout_type L>
    inline bool xiterator<St, S, L>::equal(const xiterator& rhs) const
    {
        XTENSOR_ASSERT(this->shape() == rhs.shape());
        return m_linear_index == rhs.m_linear_index;
    }

    template <class St, class S, layout_type L>
    inline bool xiterator<St, S, L>::less_than(const xiterator& rhs) const
    {
        XTENSOR_ASSERT(this->shape() == rhs.shape());
        return m_linear_index < rhs.m_linear_index;
    }

    template <class St, class S, layout_type L>
    inline bool operator==(const xiterator<St, S, L>& lhs, const xiterator<St, S, L>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class St, class S, layout_type L>
    bool operator<(const xiterator<St, S, L>& lhs, const xiterator<St, S, L>& rhs)
    {
        return lhs.less_than(rhs);
    }

    /************************************
     * xbounded_iterator implementation *
     ************************************/

    template <class It, class BIt>
    xbounded_iterator<It, BIt>::xbounded_iterator(It it, BIt bound_it)
        : m_it(it)
        , m_bound_it(bound_it)
    {
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator++() -> self_type&
    {
        ++m_it;
        ++m_bound_it;
        return *this;
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator--() -> self_type&
    {
        --m_it;
        --m_bound_it;
        return *this;
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator+=(difference_type n) -> self_type&
    {
        m_it += n;
        m_bound_it += n;
        return *this;
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator-=(difference_type n) -> self_type&
    {
        m_it -= n;
        m_bound_it -= n;
        return *this;
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_it - rhs.m_it;
    }

    template <class It, class BIt>
    inline auto xbounded_iterator<It, BIt>::operator*() const -> value_type
    {
        using type = decltype(*m_bound_it);
        return (static_cast<type>(*m_it) < *m_bound_it) ? *m_it : static_cast<value_type>((*m_bound_it) - 1);
    }

    template <class It, class BIt>
    inline bool xbounded_iterator<It, BIt>::equal(const self_type& rhs) const
    {
        return m_it == rhs.m_it && m_bound_it == rhs.m_bound_it;
    }

    template <class It, class BIt>
    inline bool xbounded_iterator<It, BIt>::less_than(const self_type& rhs) const
    {
        return m_it < rhs.m_it;
    }

    template <class It, class BIt>
    inline bool operator==(const xbounded_iterator<It, BIt>& lhs, const xbounded_iterator<It, BIt>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class It, class BIt>
    inline bool operator<(const xbounded_iterator<It, BIt>& lhs, const xbounded_iterator<It, BIt>& rhs)
    {
        return lhs.less_than(rhs);
    }
}

#endif
