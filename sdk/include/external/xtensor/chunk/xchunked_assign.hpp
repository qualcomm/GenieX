/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_CHUNKED_ASSIGN_HPP
#define XTENSOR_CHUNKED_ASSIGN_HPP

#include "../core/xnoalias.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{

    /*******************
     * xchunk_assigner *
     *******************/

    template <class T, class chunk_storage>
    class xchunked_assigner
    {
    public:

        using temporary_type = T;

        template <class E, class DST>
        void build_and_assign_temporary(const xexpression<E>& e, DST& dst);
    };

    /*********************************
     * xchunked_semantic declaration *
     *********************************/

    template <class D>
    class xchunked_semantic : public xsemantic_base<D>
    {
    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        template <class E>
        derived_type& assign_xexpression(const xexpression<E>& e);

        template <class E>
        derived_type& computed_assign(const xexpression<E>& e);

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);

    protected:

        xchunked_semantic() = default;
        ~xchunked_semantic() = default;

        xchunked_semantic(const xchunked_semantic&) = default;
        xchunked_semantic& operator=(const xchunked_semantic&) = default;

        xchunked_semantic(xchunked_semantic&&) = default;
        xchunked_semantic& operator=(xchunked_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>& e);

    private:

        template <class CS>
        xchunked_assigner<temporary_type, CS> get_assigner(const CS&) const;
    };

    /*******************
     * xchunk_iterator *
     *******************/

    template <class CS>
    class xchunked_array;

    template <class E>
    class xchunked_view;

    namespace detail
    {
        template <class T>
        struct is_xchunked_array : std::false_type
        {
        };

        template <class CS>
        struct is_xchunked_array<xchunked_array<CS>> : std::true_type
        {
        };

        template <class T>
        struct is_xchunked_view : std::false_type
        {
        };

        template <class E>
        struct is_xchunked_view<xchunked_view<E>> : std::true_type
        {
        };

        struct invalid_chunk_iterator
        {
        };

        template <class A>
        struct xchunk_iterator_array
        {
            using reference = decltype(*(std::declval<A>().chunks().begin()));

            inline decltype(auto) get_chunk(A& arr, typename A::size_type i, const xstrided_slice_vector&) const
            {
                using difference_type = typename A::difference_type;
                return *(arr.chunks().begin() + static_cast<difference_type>(i));
            }
        };

        template <class V>
        struct xchunk_iterator_view
        {
            using reference = decltype(xt::strided_view(
                std::declval<V>().expression(),
                std::declval<xstrided_slice_vector>()
            ));

            inline auto get_chunk(V& view, typename V::size_type, const xstrided_slice_vector& sv) const
            {
                return xt::strided_view(view.expression(), sv);
            }
        };

        template <class T>
        struct xchunk_iterator_base
            : std::conditional_t<
                  is_xchunked_array<std::decay_t<T>>::value,
                  xchunk_iterator_array<T>,
                  std::conditional_t<is_xchunked_view<std::decay_t<T>>::value, xchunk_iterator_view<T>, invalid_chunk_iterator>>
        {
        };
    }

    template <class E>
    class xchunk_iterator : private detail::xchunk_iterator_base<E>
    {
    public:

        using base_type = detail::xchunk_iterator_base<E>;
        using self_type = xchunk_iterator<E>;
        using size_type = typename E::size_type;
        using shape_type = typename E::shape_type;
        using slice_vector = xstrided_slice_vector;

        using reference = typename base_type::reference;
        using value_type = std::remove_reference_t<reference>;
        using pointer = value_type*;
        using difference_type = typename E::difference_type;
        using iterator_category = std::forward_iterator_tag;


        xchunk_iterator() = default;
        xchunk_iterator(E& chunked_expression, shape_type&& chunk_index, size_type chunk_linear_index);

        self_type& operator++();
        self_type operator++(int);
        decltype(auto) operator*() const;

        bool operator==(const self_type& rhs) const;
        bool operator!=(const self_type& rhs) const;

        const shape_type& chunk_index() const;

        const slice_vector& get_slice_vector() const;
        slice_vector get_chunk_slice_vector() const;

    private:

        void fill_slice_vector(size_type index);

        E* p_chunked_expression;
        shape_type m_chunk_index;
        size_type m_chunk_linear_index;
        xstrided_slice_vector m_slice_vector;
    };

    /************************************
     * xchunked_semantic implementation *
     ************************************/

    template <class T, class CS>
    template <class E, class DST>
    inline void xchunked_assigner<T, CS>::build_and_assign_temporary(const xexpression<E>& e, DST& dst)
    {
        temporary_type tmp(e, CS(), dst.chunk_shape());
        dst = std::move(tmp);
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::assign_xexpression(const xexpression<E>& e) -> derived_type&
    {
        auto& d = this->derived_cast();
        const auto& chunk_shape = d.chunk_shape();
        size_t i = 0;
        auto it_end = d.chunk_end();
        for (auto it = d.chunk_begin(); it != it_end; ++it, ++i)
        {
            auto rhs = strided_view(e.derived_cast(), it.get_slice_vector());
            if (rhs.shape() != chunk_shape)
            {
                noalias(strided_view(*it, it.get_chunk_slice_vector())) = rhs;
            }
            else
            {
                noalias(*it) = rhs;
            }
        }

        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::computed_assign(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        if (e.derived_cast().dimension() > d.dimension() || e.derived_cast().shape() > d.shape())
        {
            return operator=(e);
        }
        else
        {
            return assign_xexpression(e);
        }
    }

    template <class D>
    template <class E, class F>
    inline auto xchunked_semantic<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        for (auto& c : this->derived_cast().chunks())
        {
            c.scalar_computed_assign(e, f);
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        get_assigner(d.chunks()).build_and_assign_temporary(e, d);
        return d;
    }

    template <class D>
    template <class CS>
    inline auto xchunked_semantic<D>::get_assigner(const CS&) const -> xchunked_assigner<temporary_type, CS>
    {
        return xchunked_assigner<temporary_type, CS>();
    }

    /**********************************
     * xchunk_iterator implementation *
     **********************************/

    template <class E>
    inline xchunk_iterator<E>::xchunk_iterator(E& expression, shape_type&& chunk_index, size_type chunk_linear_index)
        : p_chunked_expression(&expression)
        , m_chunk_index(std::move(chunk_index))
        , m_chunk_linear_index(chunk_linear_index)
        , m_slice_vector(m_chunk_index.size())
    {
        for (size_type i = 0; i < m_chunk_index.size(); ++i)
        {
            fill_slice_vector(i);
        }
    }

    template <class E>
    inline xchunk_iterator<E>& xchunk_iterator<E>::operator++()
    {
        if (m_chunk_linear_index + 1u != p_chunked_expression->grid_size())
        {
            size_type i = p_chunked_expression->dimension();
            while (i != 0)
            {
                --i;
                if (m_chunk_index[i] + 1u == p_chunked_expression->grid_shape()[i])
                {
                    m_chunk_index[i] = 0;
                    fill_slice_vector(i);
                }
                else
                {
                    m_chunk_index[i] += 1;
                    fill_slice_vector(i);
                    break;
                }
            }
        }
        m_chunk_linear_index++;
        return *this;
    }

    template <class E>
    inline xchunk_iterator<E> xchunk_iterator<E>::operator++(int)
    {
        xchunk_iterator<E> it = *this;
        ++(*this);
        return it;
    }

    template <class E>
    inline decltype(auto) xchunk_iterator<E>::operator*() const
    {
        return base_type::get_chunk(*p_chunked_expression, m_chunk_linear_index, m_slice_vector);
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator==(const xchunk_iterator& other) const
    {
        return m_chunk_linear_index == other.m_chunk_linear_index;
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator!=(const xchunk_iterator& other) const
    {
        return !(*this == other);
    }

    template <class E>
    inline auto xchunk_iterator<E>::get_slice_vector() const -> const slice_vector&
    {
        return m_slice_vector;
    }

    template <class E>
    auto xchunk_iterator<E>::chunk_index() const -> const shape_type&
    {
        return m_chunk_index;
    }

    template <class E>
    inline auto xchunk_iterator<E>::get_chunk_slice_vector() const -> slice_vector
    {
        slice_vector slices(m_chunk_index.size());
        for (size_type i = 0; i < m_chunk_index.size(); ++i)
        {
            size_type chunk_shape = p_chunked_expression->chunk_shape()[i];
            size_type end = std::min(
                chunk_shape,
                p_chunked_expression->shape()[i] - m_chunk_index[i] * chunk_shape
            );
            slices[i] = range(0u, end);
        }
        return slices;
    }

    template <class E>
    inline void xchunk_iterator<E>::fill_slice_vector(size_type i)
    {
        size_type range_start = m_chunk_index[i] * p_chunked_expression->chunk_shape()[i];
        size_type range_end = std::min(
            (m_chunk_index[i] + 1) * p_chunked_expression->chunk_shape()[i],
            p_chunked_expression->shape()[i]
        );
        m_slice_vector[i] = range(range_start, range_end);
    }
}

#endif
