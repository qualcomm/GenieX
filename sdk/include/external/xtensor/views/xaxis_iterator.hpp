/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_AXIS_ITERATOR_HPP
#define XTENSOR_AXIS_ITERATOR_HPP

#include "../views/xstrided_view.hpp"

namespace xt
{

    /******************
     * xaxis_iterator *
     ******************/

    /**
     * @class xaxis_iterator
     * @brief Class for iteration over (N-1)-dimensional slices, where
     * N is the dimension of the underlying expression
     *
     * If N is the number of dimensions of an expression, the xaxis_iterator
     * iterates over (N-1)-dimensional slices oriented along the specified axis.
     *
     * @tparam CT the closure type of the \ref xexpression
     */
    template <class CT>
    class xaxis_iterator
    {
    public:

        using self_type = xaxis_iterator<CT>;

        using xexpression_type = std::decay_t<CT>;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;
        using shape_type = typename xexpression_type::shape_type;
        using value_type = xstrided_view<CT, shape_type>;
        using reference = std::remove_reference_t<xtl::apply_cv_t<CT, value_type>>;
        using pointer = xtl::xclosure_pointer<std::remove_reference_t<xtl::apply_cv_t<CT, value_type>>>;

        using iterator_category = std::forward_iterator_tag;

        template <class CTA>
        xaxis_iterator(CTA&& e, size_type axis);
        template <class CTA>
        xaxis_iterator(CTA&& e, size_type axis, size_type index, size_type offset);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;

    private:

        using storing_type = xtl::ptr_closure_type_t<CT>;
        mutable storing_type p_expression;
        size_type m_index;
        size_type m_add_offset;
        value_type m_sv;

        template <class T, class CTA>
        std::enable_if_t<std::is_pointer<T>::value, T> get_storage_init(CTA&& e) const;

        template <class T, class CTA>
        std::enable_if_t<!std::is_pointer<T>::value, T> get_storage_init(CTA&& e) const;
    };

    template <class CT>
    bool operator==(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs);

    template <class CT>
    bool operator!=(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs);

    template <class E>
    auto axis_begin(E&& e);

    template <class E>
    auto axis_begin(E&& e, typename std::decay_t<E>::size_type axis);

    template <class E>
    auto axis_end(E&& e);

    template <class E>
    auto axis_end(E&& e, typename std::decay_t<E>::size_type axis);

    /*********************************
     * xaxis_iterator implementation *
     *********************************/

    namespace detail
    {
        template <class CT>
        auto derive_xstrided_view(
            CT&& e,
            typename std::decay_t<CT>::size_type axis,
            typename std::decay_t<CT>::size_type offset
        )
        {
            using xexpression_type = std::decay_t<CT>;
            using shape_type = typename xexpression_type::shape_type;
            using strides_type = typename xexpression_type::strides_type;

            const auto& e_shape = e.shape();
            shape_type shape(e_shape.size() - 1);
            auto nxt = std::copy(e_shape.cbegin(), e_shape.cbegin() + axis, shape.begin());
            std::copy(e_shape.cbegin() + axis + 1, e_shape.end(), nxt);

            const auto& e_strides = e.strides();
            strides_type strides(e_strides.size() - 1);
            auto nxt_strides = std::copy(e_strides.cbegin(), e_strides.cbegin() + axis, strides.begin());
            std::copy(e_strides.cbegin() + axis + 1, e_strides.end(), nxt_strides);

            return strided_view(std::forward<CT>(e), std::move(shape), std::move(strides), offset, e.layout());
        }
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<std::is_pointer<T>::value, T> xaxis_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return &e;
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<!std::is_pointer<T>::value, T> xaxis_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return e;
    }

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xaxis_iterator
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over taking N-1 dimensional slices
     */
    template <class CT>
    template <class CTA>
    inline xaxis_iterator<CT>::xaxis_iterator(CTA&& e, size_type axis)
        : xaxis_iterator(std::forward<CTA>(e), axis, 0, e.data_offset())
    {
    }

    /**
     * Constructs an xaxis_iterator starting at specified index and offset
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over taking N-1 dimensional slices
     * @param index the starting index for the iterator
     * @param offset the starting offset for the iterator
     */
    template <class CT>
    template <class CTA>
    inline xaxis_iterator<CT>::xaxis_iterator(CTA&& e, size_type axis, size_type index, size_type offset)
        : p_expression(get_storage_init<storing_type>(std::forward<CTA>(e)))
        , m_index(index)
        , m_add_offset(static_cast<size_type>(e.strides()[axis]))
        , m_sv(detail::derive_xstrided_view<CTA>(std::forward<CTA>(e), axis, offset))
    {
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
    inline auto xaxis_iterator<CT>::operator++() -> self_type&
    {
        m_sv.set_offset(m_sv.data_offset() + m_add_offset);
        ++m_index;
        return *this;
    }

    /**
     * Makes a copy of the iterator, increments it to the next
     * position, and returns the copy.
     */
    template <class CT>
    inline auto xaxis_iterator<CT>::operator++(int) -> self_type
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
    inline auto xaxis_iterator<CT>::operator*() const -> reference
    {
        return m_sv;
    }

    /**
     * Returns a pointer to the strided view at the current iteration position
     *
     * @return a pointer to a strided_view
     */
    template <class CT>
    inline auto xaxis_iterator<CT>::operator->() const -> pointer
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
     * @param
     * @return true if the iterators are equivalent, false otherwise
     */
    template <class CT>
    inline bool xaxis_iterator<CT>::equal(const self_type& rhs) const
    {
        return p_expression == rhs.p_expression && m_index == rhs.m_index
               && m_sv.data_offset() == rhs.m_sv.data_offset();
    }

    /**
     * Checks equality of the iterators.
     *
     * @return true if the iterators are equivalent, false otherwise
     */
    template <class CT>
    inline bool operator==(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs)
    {
        return lhs.equal(rhs);
    }

    /**
     * Checks inequality of the iterators
     * @return true if the iterators are different, true otherwise
     */
    template <class CT>
    inline bool operator!=(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs)
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
     * @return an instance of xaxis_iterator
     */
    template <class E>
    inline auto axis_begin(E&& e)
    {
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), 0);
    }

    /**
     * Returns an iterator to the first element of the expression for the specified axis
     *
     * @param e the expession to iterate over
     * @param axis the axis to iterate over
     * @return an instance of xaxis_iterator
     */
    template <class E>
    inline auto axis_begin(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), axis);
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression for axis 0
     *
     * @param e the expession to iterate over
     * @return an instance of xaxis_iterator
     */
    template <class E>
    inline auto axis_end(E&& e)
    {
        using size_type = typename std::decay_t<E>::size_type;
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        return return_type(
            std::forward<E>(e),
            0,
            e.shape()[0],
            static_cast<size_type>(e.strides()[0]) * e.shape()[0]
        );
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression for the specified axis
     *
     * @param e the expression to iterate over
     * @param axis the axis to iterate over
     * @return an instance of xaxis_iterator
     */
    template <class E>
    inline auto axis_end(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using size_type = typename std::decay_t<E>::size_type;
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        return return_type(
            std::forward<E>(e),
            axis,
            e.shape()[axis],
            static_cast<size_type>(e.strides()[axis]) * e.shape()[axis]
        );
    }

    //@}
}

#endif
