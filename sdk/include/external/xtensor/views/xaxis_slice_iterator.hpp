/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_AXIS_SLICE_ITERATOR_HPP
#define XTENSOR_AXIS_SLICE_ITERATOR_HPP

#include "../views/xstrided_view.hpp"

namespace xt
{

    /**
     * @class xaxis_slice_iterator
     * @brief Class for iteration over one-dimensional slices
     *
     * The xaxis_slice_iterator iterates over one-dimensional slices
     * oriented along the specified axis
     *
     * @tparam CT the closure type of the \ref xexpression
     */
    template <class CT>
    class xaxis_slice_iterator
    {
    public:

        using self_type = xaxis_slice_iterator<CT>;

        using xexpression_type = std::decay_t<CT>;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;
        using shape_type = typename xexpression_type::shape_type;
        using strides_type = typename xexpression_type::strides_type;
        using value_type = xstrided_view<CT, shape_type>;
        using reference = std::remove_reference_t<xtl::apply_cv_t<CT, value_type>>;
        using pointer = xtl::xclosure_pointer<std::remove_reference_t<xtl::apply_cv_t<CT, value_type>>>;

        using iterator_category = std::forward_iterator_tag;

        template <class CTA>
        xaxis_slice_iterator(CTA&& e, size_type axis);
        template <class CTA>
        xaxis_slice_iterator(CTA&& e, size_type axis, size_type index, size_type offset);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;

    private:

        using storing_type = xtl::ptr_closure_type_t<CT>;
        mutable storing_type p_expression;
        size_type m_index;
        size_type m_offset;
        size_type m_axis_stride;
        size_type m_lower_shape;
        size_type m_upper_shape;
        size_type m_iter_size;
        bool m_is_target_axis;
        value_type m_sv;

        template <class T, class CTA>
        std::enable_if_t<std::is_pointer<T>::value, T> get_storage_init(CTA&& e) const;

        template <class T, class CTA>
        std::enable_if_t<!std::is_pointer<T>::value, T> get_storage_init(CTA&& e) const;
    };

    template <class CT>
    bool operator==(const xaxis_slice_iterator<CT>& lhs, const xaxis_slice_iterator<CT>& rhs);

    template <class CT>
    bool operator!=(const xaxis_slice_iterator<CT>& lhs, const xaxis_slice_iterator<CT>& rhs);

    template <class E>
    auto xaxis_slice_begin(E&& e);

    template <class E>
    auto xaxis_slice_begin(E&& e, typename std::decay_t<E>::size_type axis);

    template <class E>
    auto xaxis_slice_end(E&& e);

    template <class E>
    auto xaxis_slice_end(E&& e, typename std::decay_t<E>::size_type axis);

    /***************************************
     * xaxis_slice_iterator implementation *
     ***************************************/

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<std::is_pointer<T>::value, T>
    xaxis_slice_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return &e;
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<!std::is_pointer<T>::value, T>
    xaxis_slice_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return e;
    }

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xaxis_slice_iterator
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over taking one dimensional slices
     */
    template <class CT>
    template <class CTA>
    inline xaxis_slice_iterator<CT>::xaxis_slice_iterator(CTA&& e, size_type axis)
        : xaxis_slice_iterator(std::forward<CTA>(e), axis, 0, e.data_offset())
    {
    }

    /**
     * Constructs an xaxis_slice_iterator starting at specified index and offset
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over taking one dimensional slices
     * @param index the starting index for the iterator
     * @param offset the starting offset for the iterator
     */
    template <class CT>
    template <class CTA>
    inline xaxis_slice_iterator<CT>::xaxis_slice_iterator(CTA&& e, size_type axis, size_type index, size_type offset)
        : p_expression(get_storage_init<storing_type>(std::forward<CTA>(e)))
        , m_index(index)
        , m_offset(offset)
        , m_axis_stride(static_cast<size_type>(e.strides()[axis]) * (e.shape()[axis] - 1u))
        , m_lower_shape(0)
        , m_upper_shape(0)
        , m_iter_size(0)
        , m_is_target_axis(false)
        , m_sv(strided_view(
              std::forward<CT>(e),
              std::forward<shape_type>({e.shape()[axis]}),
              std::forward<strides_type>({e.strides()[axis]}),
              offset,
              e.layout()
          ))
    {
        if (e.layout() == layout_type::row_major)
        {
            m_is_target_axis = axis == e.dimension() - 1;
            m_lower_shape = std::accumulate(
                e.shape().begin() + axis + 1,
                e.shape().end(),
                size_t(1),
                std::multiplies<>()
            );
            m_iter_size = std::accumulate(e.shape().begin() + 1, e.shape().end(), size_t(1), std::multiplies<>());
        }
        else
        {
            m_is_target_axis = axis == 0;
            m_lower_shape = std::accumulate(
                e.shape().begin(),
                e.shape().begin() + axis,
                size_t(1),
                std::multiplies<>()
            );
            m_iter_size = std::accumulate(e.shape().begin(), e.shape().end() - 1, size_t(1), std::multiplies<>());
        }
        m_upper_shape = m_lower_shape + m_axis_stride;
    }

    //@}

    /**
     * @name Increment
     */
    //@{
    /**
     * Increments the iterator to the next position and returns it.
     */
    template <class CT>
    inline auto xaxis_slice_iterator<CT>::operator++() -> self_type&
    {
        ++m_index;
        ++m_offset;
        auto index_compare = (m_offset % m_iter_size);
        if (m_is_target_axis || (m_upper_shape >= index_compare && index_compare >= m_lower_shape))
        {
            m_offset += m_axis_stride;
        }
        m_sv.set_offset(m_offset);
        return *this;
    }

    /**
     * Makes a copy of the iterator, increments it to the next
     * position, and returns the copy.
     */
    template <class CT>
    inline auto xaxis_slice_iterator<CT>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    //@}

    /**
     * @name Reference
     */
    //@{
    /**
     * Returns the strided view at the current iteration position
     *
     * @return a strided_view
     */
    template <class CT>
    inline auto xaxis_slice_iterator<CT>::operator*() const -> reference
    {
        return m_sv;
    }

    /**
     * Returns a pointer to the strided view at the current iteration position
     *
     * @return a pointer to a strided_view
     */
    template <class CT>
    inline auto xaxis_slice_iterator<CT>::operator->() const -> pointer
    {
        return xtl::closure_pointer(operator*());
    }

    //@}

    /*
     * @name Comparisons
     */
    //@{
    /**
     * Checks equality of the xaxis_slice_iterator and \c rhs.
     *
     * @return true if the iterators are equivalent, false otherwise
     */
    template <class CT>
    inline bool xaxis_slice_iterator<CT>::equal(const self_type& rhs) const
    {
        return p_expression == rhs.p_expression && m_index == rhs.m_index;
    }

    /**
     * Checks equality of the iterators.
     *
     * @return true if the iterators are equivalent, false otherwise
     */
    template <class CT>
    inline bool operator==(const xaxis_slice_iterator<CT>& lhs, const xaxis_slice_iterator<CT>& rhs)
    {
        return lhs.equal(rhs);
    }

    /**
     * Checks inequality of the iterators
     * @return true if the iterators are different, true otherwise
     */
    template <class CT>
    inline bool operator!=(const xaxis_slice_iterator<CT>& lhs, const xaxis_slice_iterator<CT>& rhs)
    {
        return !(lhs == rhs);
    }

    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression for axis 0
     *
     * @param e the expession to iterate over
     * @return an instance of xaxis_slice_iterator
     */
    template <class E>
    inline auto axis_slice_begin(E&& e)
    {
        using return_type = xaxis_slice_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), 0);
    }

    /**
     * Returns an iterator to the first element of the expression for the specified axis
     *
     * @param e the expession to iterate over
     * @param axis the axis to iterate over
     * @return an instance of xaxis_slice_iterator
     */
    template <class E>
    inline auto axis_slice_begin(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using return_type = xaxis_slice_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), axis, 0, e.data_offset());
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression for axis 0
     *
     * @param e the expession to iterate over
     * @return an instance of xaxis_slice_iterator
     */
    template <class E>
    inline auto axis_slice_end(E&& e)
    {
        using return_type = xaxis_slice_iterator<xtl::closure_type_t<E>>;
        return return_type(
            std::forward<E>(e),
            0,
            std::accumulate(e.shape().begin() + 1, e.shape().end(), size_t(1), std::multiplies<>()),
            e.size()
        );
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression for the specified axis
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over
     * @return an instance of xaxis_slice_iterator
     */
    template <class E>
    inline auto axis_slice_end(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using return_type = xaxis_slice_iterator<xtl::closure_type_t<E>>;
        auto index_sum = std::accumulate(
            e.shape().begin(),
            e.shape().begin() + axis,
            size_t(1),
            std::multiplies<>()
        );
        return return_type(
            std::forward<E>(e),
            axis,
            std::accumulate(e.shape().begin() + axis + 1, e.shape().end(), index_sum, std::multiplies<>()),
            e.size() + axis
        );
    }

    //@}
}

#endif
