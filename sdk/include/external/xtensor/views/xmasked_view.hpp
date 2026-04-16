/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XMASKED_VIEW_HPP
#define XTENSOR_XMASKED_VIEW_HPP

#include "../core/xaccessible.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xshape.hpp"
#include "../core/xtensor_forward.hpp"
#include "../utils/xutils.hpp"
#include "xtl/xmasked_value.hpp"

namespace xt
{
    /****************************
     * xmasked_view declaration  *
     *****************************/

    template <class CTD, class CTM>
    class xmasked_view;

    template <class D, bool is_const>
    class xmasked_view_stepper;

    template <class T>
    struct xcontainer_inner_types;

    template <class CTD, class CTM>
    struct xcontainer_inner_types<xmasked_view<CTD, CTM>>
    {
        using data_type = std::decay_t<CTD>;
        using mask_type = std::decay_t<CTM>;
        using base_value_type = typename data_type::value_type;
        using flag_type = typename mask_type::value_type;
        using val_reference = inner_reference_t<CTD>;
        using mask_reference = inner_reference_t<CTM>;
        using value_type = xtl::xmasked_value<base_value_type, flag_type>;
        using reference = xtl::xmasked_value<val_reference, mask_reference>;
        using const_reference = xtl::xmasked_value<typename data_type::const_reference, typename mask_type::const_reference>;
        using size_type = typename data_type::size_type;
        using temporary_type = xarray<xtl::xmasked_value<base_value_type, flag_type>>;
    };

    template <class CTD, class CTM>
    struct xiterable_inner_types<xmasked_view<CTD, CTM>>
    {
        using masked_view_type = xmasked_view<CTD, CTM>;
        using inner_shape_type = typename std::decay_t<CTD>::inner_shape_type;
        using stepper = xmasked_view_stepper<masked_view_type, false>;
        using const_stepper = xmasked_view_stepper<masked_view_type, true>;
    };

    /**
     * @class xmasked_view
     * @brief View on an xoptional_assembly or xoptional_assembly_adaptor
     * hiding values depending on a given mask.
     *
     * The xmasked_view class implements a view on an xoptional_assembly or
     * xoptional_assembly_adaptor, it takes this xoptional_assembly and a
     * mask as input. The mask is an xexpression containing boolean values,
     * whenever the value of the mask is false, the optional value of
     * xmasked_view is considered missing, otherwise it depends on the
     * underlying xoptional_assembly.
     *
     * @tparam CTD The type of expression holding the values.
     * @tparam CTM The type of expression holding the mask.
     */
    template <class CTD, class CTM>
    class xmasked_view : public xview_semantic<xmasked_view<CTD, CTM>>,
                         private xaccessible<xmasked_view<CTD, CTM>>,
                         private xiterable<xmasked_view<CTD, CTM>>
    {
    public:

        using self_type = xmasked_view<CTD, CTM>;
        using semantic_base = xview_semantic<xmasked_view<CTD, CTM>>;
        using accessible_base = xaccessible<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
        using temporary_type = typename inner_types::temporary_type;

        using data_type = typename inner_types::data_type;
        using mask_type = typename inner_types::mask_type;
        using value_expression = CTD;
        using mask_expression = CTM;

        static constexpr bool is_data_const = std::is_const<std::remove_reference_t<value_expression>>::value;

        using base_value_type = typename inner_types::base_value_type;
        using base_reference = typename data_type::reference;
        using base_const_reference = typename data_type::const_reference;

        using flag_type = typename inner_types::flag_type;
        using flag_reference = typename mask_type::reference;
        using flag_const_reference = typename mask_type::const_reference;

        using val_reference = typename inner_types::val_reference;
        using mask_reference = typename inner_types::mask_reference;

        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;

        using pointer = xtl::xclosure_pointer<reference>;
        using const_pointer = xtl::xclosure_pointer<const_reference>;

        using size_type = typename inner_types::size_type;
        using difference_type = typename data_type::difference_type;

        using bool_load_type = xtl::xmasked_value<typename data_type::bool_load_type, mask_type>;

        using shape_type = typename data_type::shape_type;
        using strides_type = typename data_type::strides_type;

        static constexpr layout_type static_layout = data_type::static_layout;
        static constexpr bool contiguous_layout = false;

        using inner_shape_type = typename data_type::inner_shape_type;
        using inner_strides_type = typename data_type::inner_strides_type;
        using inner_backstrides_type = typename data_type::inner_backstrides_type;

        using expression_tag = xtensor_expression_tag;

        using iterable_base = xiterable<xmasked_view<CTD, CTM>>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        template <class D, class M>
        xmasked_view(D&& data, M&& mask);

        xmasked_view(const xmasked_view&) = default;

        size_type size() const noexcept;
        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;
        using accessible_base::dimension;
        using accessible_base::shape;

        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class T>
        void fill(const T& value);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        using accessible_base::at;
        using accessible_base::operator[];
        using accessible_base::back;
        using accessible_base::front;
        using accessible_base::in_bounds;
        using accessible_base::periodic;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        data_type& value() noexcept;
        const data_type& value() const noexcept;

        mask_type& visible() noexcept;
        const mask_type& visible() const noexcept;

        using iterable_base::begin;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::crbegin;
        using iterable_base::crend;
        using iterable_base::end;
        using iterable_base::rbegin;
        using iterable_base::rend;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        self_type& operator=(const self_type& rhs);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

    private:

        CTD m_data;
        CTM m_mask;

        void assign_temporary_impl(temporary_type&& tmp);

        friend class xiterable<self_type>;
        friend class xconst_iterable<self_type>;
        friend class xview_semantic<self_type>;
        friend class xaccessible<self_type>;
        friend class xconst_accessible<self_type>;
    };

    template <class D, bool is_const>
    class xmasked_view_stepper
    {
    public:

        using self_type = xmasked_view_stepper<D, is_const>;
        using masked_view_type = std::decay_t<D>;
        using value_type = typename masked_view_type::value_type;
        using reference = std::
            conditional_t<is_const, typename masked_view_type::const_reference, typename masked_view_type::reference>;
        using pointer = std::
            conditional_t<is_const, typename masked_view_type::const_pointer, typename masked_view_type::pointer>;
        using size_type = typename masked_view_type::size_type;
        using difference_type = typename masked_view_type::difference_type;
        using data_type = typename masked_view_type::data_type;
        using mask_type = typename masked_view_type::mask_type;
        using value_stepper = std::conditional_t<is_const, typename data_type::const_stepper, typename data_type::stepper>;
        using mask_stepper = std::conditional_t<is_const, typename mask_type::const_stepper, typename mask_type::stepper>;

        xmasked_view_stepper(value_stepper vs, mask_stepper fs) noexcept;


        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        reference operator*() const;

    private:

        value_stepper m_vs;
        mask_stepper m_ms;
    };

    /*******************************
     * xmasked_view implementation *
     *******************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Creates an xmasked_view, given the xoptional_assembly or
     * xoptional_assembly_adaptor and the mask
     *
     * @param data the underlying xoptional_assembly or xoptional_assembly_adaptor
     * @param mask the mask.
     */
    template <class CTD, class CTM>
    template <class D, class M>
    inline xmasked_view<CTD, CTM>::xmasked_view(D&& data, M&& mask)
        : m_data(std::forward<D>(data))
        , m_mask(std::forward<M>(mask))
    {
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of elements in the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::size() const noexcept -> size_type
    {
        return m_data.size();
    }

    /**
     * Returns the shape of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::shape() const noexcept -> const inner_shape_type&
    {
        return m_data.shape();
    }

    /**
     * Returns the strides of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::strides() const noexcept -> const inner_strides_type&
    {
        return m_data.strides();
    }

    /**
     * Returns the backstrides of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return m_data.backstrides();
    }

    //@}

    /**
     * Return the layout_type of the xmasked_view
     * @return layout_type of the xmasked_view
     */
    template <class CTD, class CTM>
    inline layout_type xmasked_view<CTD, CTM>::layout() const noexcept
    {
        return m_data.layout();
    }

    template <class CTD, class CTM>
    inline bool xmasked_view<CTD, CTM>::is_contiguous() const noexcept
    {
        return false;
    }

    /**
     * Fills the data with the given value.
     * @param value the value to fill the data with.
     */
    template <class CTD, class CTM>
    template <class T>
    inline void xmasked_view<CTD, CTM>::fill(const T& value)
    {
        std::fill(this->begin(), this->end(), value);
    }

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::operator()(Args... args) -> reference
    {
        return reference(m_data(args...), m_mask(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::operator()(Args... args) const -> const_reference
    {
        return const_reference(m_data(args...), m_mask(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the  xmasked_view.
     * @param args a list of indices specifying the position in the  xmasked_view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  xmasked_view, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::unchecked(Args... args) -> reference
    {
        return reference(m_data.unchecked(args...), m_mask.unchecked(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the  xmasked_view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  xmasked_view, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::unchecked(Args... args) const -> const_reference
    {
        return const_reference(m_data.unchecked(args...), m_mask.unchecked(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the xmasked_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class It>
    inline auto xmasked_view<CTD, CTM>::element(It first, It last) -> reference
    {
        return reference(m_data.element(first, last), m_mask.element(first, last));
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class It>
    inline auto xmasked_view<CTD, CTM>::element(It first, It last) const -> const_reference
    {
        return const_reference(m_data.element(first, last), m_mask.element(first, last));
    }

    //@}

    template <class CTD, class CTM>
    template <class S>
    inline bool xmasked_view<CTD, CTM>::has_linear_assign(const S& strides) const noexcept
    {
        return m_data.has_linear_assign(strides) && m_mask.has_linear_assign(strides);
    }

    /**
     * Return an expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() noexcept -> data_type&
    {
        return m_data;
    }

    /**
     * Return a constant expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() const noexcept -> const data_type&
    {
        return m_data;
    }

    /**
     * Return an expression for the mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::visible() noexcept -> mask_type&
    {
        return m_mask;
    }

    /**
     * Return a constant expression for the mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::visible() const noexcept -> const mask_type&
    {
        return m_mask;
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(value().stepper_begin(shape), visible().stepper_begin(shape));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(value().stepper_end(shape, l), visible().stepper_end(shape, l));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_begin(shape), visible().stepper_begin(shape));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        return const_stepper(value().stepper_end(shape, l), visible().stepper_end(shape, l));
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::operator=(const self_type& rhs) -> self_type&
    {
        temporary_type tmp(rhs);
        return this->assign_temporary(std::move(tmp));
    }

    template <class CTD, class CTM>
    template <class E>
    inline auto xmasked_view<CTD, CTM>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class CTD, class CTM>
    template <class E>
    inline auto xmasked_view<CTD, CTM>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CTD, class CTM>
    inline void xmasked_view<CTD, CTM>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    template <class CTD, class CTM>
    inline xmasked_view<CTD, CTM> masked_view(CTD&& data, CTM&& mask)
    {
        return xmasked_view<CTD, CTM>(std::forward<CTD>(data), std::forward<CTM>(mask));
    }

    /***************************************
     * xmasked_view_stepper implementation *
     ***************************************/

    template <class D, bool C>
    inline xmasked_view_stepper<D, C>::xmasked_view_stepper(value_stepper vs, mask_stepper ms) noexcept
        : m_vs(vs)
        , m_ms(ms)
    {
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::step(size_type dim)
    {
        m_vs.step(dim);
        m_ms.step(dim);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::step_back(size_type dim)
    {
        m_vs.step_back(dim);
        m_ms.step_back(dim);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::step(size_type dim, size_type n)
    {
        m_vs.step(dim, n);
        m_ms.step(dim, n);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::step_back(size_type dim, size_type n)
    {
        m_vs.step_back(dim, n);
        m_ms.step_back(dim, n);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::reset(size_type dim)
    {
        m_vs.reset(dim);
        m_ms.reset(dim);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::reset_back(size_type dim)
    {
        m_vs.reset_back(dim);
        m_ms.reset_back(dim);
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::to_begin()
    {
        m_vs.to_begin();
        m_ms.to_begin();
    }

    template <class D, bool C>
    inline void xmasked_view_stepper<D, C>::to_end(layout_type l)
    {
        m_vs.to_end(l);
        m_ms.to_end(l);
    }

    template <class D, bool C>
    inline auto xmasked_view_stepper<D, C>::operator*() const -> reference
    {
        return reference(*m_vs, *m_ms);
    }
}

#endif
