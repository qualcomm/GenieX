/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XMULTIINDEX_ITERATOR
#define XTENSOR_XMULTIINDEX_ITERATOR

#include "../views/xstrided_view.hpp"
#include "xtl/xsequence.hpp"

namespace xt
{

    template <class S>
    class xmultiindex_iterator
    {
    public:

        using self_type = xmultiindex_iterator<S>;
        using shape_type = S;

        using value_type = shape_type;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::size_t;
        using iterator_category = std::forward_iterator_tag;

        xmultiindex_iterator() = default;

        template <class B, class E, class C>
        xmultiindex_iterator(B&& begin, E&& end, C&& current, const std::size_t linear_index)
            : m_begin(std::forward<B>(begin))
            , m_end(std::forward<E>(end))
            , m_current(std::forward<C>(current))
            , m_linear_index(linear_index)
        {
        }

        self_type& operator++()
        {
            std::size_t i = m_begin.size();
            while (i != 0)
            {
                --i;
                if (m_current[i] + 1u == m_end[i])
                {
                    m_current[i] = m_begin[i];
                }
                else
                {
                    m_current[i] += 1;
                    break;
                }
            }
            m_linear_index++;
            return *this;
        }

        self_type operator++(int)
        {
            self_type it = *this;
            ++(*this);
            return it;
        }

        shape_type& operator*()
        {
            return m_current;
        }

        const shape_type& operator*() const
        {
            return m_current;
        }

        bool operator==(const self_type& rhs) const
        {
            return m_linear_index == rhs.m_linear_index;
        }

        bool operator!=(const self_type& rhs) const
        {
            return !this->operator==(rhs);
        }

    private:

        shape_type m_begin;
        shape_type m_end;
        shape_type m_current;
        std::size_t m_linear_index{0};
    };

    template <class S, class B, class E>
    auto multiindex_iterator_begin(B&& roi_begin, E&& roi_end)
    {
        S current;
        resize_container(current, roi_begin.size());
        std::copy(roi_begin.begin(), roi_begin.end(), current.begin());
        return xmultiindex_iterator<S>(std::forward<B>(roi_begin), std::forward<E>(roi_end), std::move(current), 0);
    }

    template <class S, class B, class E>
    auto multiindex_iterator_end(B&& roi_begin, E&& roi_end)
    {
        S current;
        resize_container(current, roi_begin.size());
        std::copy(roi_end.begin(), roi_end.end(), current.begin());

        std::size_t linear_index = 1;
        for (std::size_t i = 0; i < roi_begin.size(); ++i)
        {
            linear_index *= roi_end[i] - roi_begin[i];
        }

        return xmultiindex_iterator<S>(
            std::forward<B>(roi_begin),
            std::forward<E>(roi_end),
            std::move(current),
            linear_index
        );
    }

}

#endif
