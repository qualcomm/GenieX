/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_HPP
#define XOPTIONAL_ASSEMBLY_HPP

#include "../core/xsemantic.hpp"
#include "../optional/xoptional.hpp"
#include "../optional/xoptional_assembly_base.hpp"

namespace xt
{

    /**********************************
     * xoptional_assembly declaration *
     **********************************/

    template <class VE, class FE>
    class xoptional_assembly;

    template <class VE, class FE>
    struct xcontainer_inner_types<xoptional_assembly<VE, FE>>
    {
        using raw_value_expression = VE;
        using value_storage_type = typename raw_value_expression::storage_type&;
        using raw_flag_expression = FE;
        using flag_storage_type = typename raw_flag_expression::storage_type&;
        using storage_type = xoptional_assembly_storage<value_storage_type, flag_storage_type>;
        using temporary_type = xoptional_assembly<VE, FE>;
    };

    template <class VE, class FE>
    struct xiterable_inner_types<xoptional_assembly<VE, FE>>
    {
        using assembly_type = xoptional_assembly<VE, FE>;
        using inner_shape_type = typename VE::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    /**
     * @class xoptional_assembly
     * @brief Dense multidimensional container holding optional values,
     * optimized for tensor operations.
     *
     * The xoptional_assembly class implements a dense multidimensional container
     * hoding optional values. This container is optimized of tensor operations:
     * contrary to xarray_optional, xoptional_assembly holds two separated expressions,
     * one for the values, the other for the missing mask.
     *
     * @tparam VE The type of expression holding the values.
     * @tparam FE The type of expression holding the missing mask.
     */
    template <class VE, class FE>
    class xoptional_assembly : public xoptional_assembly_base<xoptional_assembly<VE, FE>>,
                               public xcontainer_semantic<xoptional_assembly<VE, FE>>
    {
    public:

        using self_type = xoptional_assembly<VE, FE>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using raw_value_expression = typename base_type::raw_value_expression;
        using raw_flag_expression = typename base_type::raw_flag_expression;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using const_value_expression = typename base_type::const_value_expression;
        using const_flag_expression = typename base_type::const_flag_expression;
        using storage_type = typename base_type::storage_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        using assembly_type = base_type;

        xoptional_assembly();
        explicit xoptional_assembly(const shape_type& shape, layout_type l = base_type::static_layout);
        explicit xoptional_assembly(
            const shape_type& shape,
            const value_type& value,
            layout_type l = base_type::static_layout
        );
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides);
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides, const value_type& value);

        xoptional_assembly(const VE& ve);
        xoptional_assembly(VE&& ve);

        template <class OVE, class OFE, typename = std::enable_if_t<is_xexpression<OVE>::value && is_xexpression<OFE>::value>>
        xoptional_assembly(OVE&& ove, OFE&& ofe);

        xoptional_assembly(const value_type& value);
        xoptional_assembly(nested_initializer_list_t<value_type, 1> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 2> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 3> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 4> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 5> t);

        template <class S = shape_type>
        static xoptional_assembly from_shape(S&& s);

        ~xoptional_assembly() = default;

        xoptional_assembly(const xoptional_assembly&);
        xoptional_assembly& operator=(const xoptional_assembly&);

        xoptional_assembly(xoptional_assembly&&);
        xoptional_assembly& operator=(xoptional_assembly&&);

        template <class E>
        xoptional_assembly(const xexpression<E>& e);

        template <class E>
        xoptional_assembly& operator=(const xexpression<E>& e);

    private:

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        value_expression value_impl() noexcept;
        const_value_expression value_impl() const noexcept;

        flag_expression has_value_impl() noexcept;
        const_flag_expression has_value_impl() const noexcept;

        raw_value_expression m_value;
        raw_flag_expression m_has_value;
        storage_type m_storage;

        friend class xoptional_assembly_base<xoptional_assembly<VE, FE>>;
    };

    /******************************************
     * xoptional_assembly_adaptor declaration *
     ******************************************/

    template <class VEC, class FEC>
    class xoptional_assembly_adaptor;

    template <class VEC, class FEC>
    struct xcontainer_inner_types<xoptional_assembly_adaptor<VEC, FEC>>
    {
        using raw_value_expression = std::remove_reference_t<VEC>;
        using value_storage_type = std::conditional_t<
            std::is_const<raw_value_expression>::value,
            const typename raw_value_expression::storage_type&,
            typename raw_value_expression::storage_type&>;
        using raw_flag_expression = std::remove_reference_t<FEC>;
        using flag_storage_type = std::conditional_t<
            std::is_const<raw_flag_expression>::value,
            const typename raw_flag_expression::storage_type&,
            typename raw_flag_expression::storage_type&>;
        using storage_type = xoptional_assembly_storage<value_storage_type, flag_storage_type>;
        using temporary_type = xoptional_assembly<raw_value_expression, raw_flag_expression>;
    };

    template <class VEC, class FEC>
    struct xiterable_inner_types<xoptional_assembly_adaptor<VEC, FEC>>
    {
        using assembly_type = xoptional_assembly_adaptor<VEC, FEC>;
        using inner_shape_type = typename std::decay_t<VEC>::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    /**
     * @class xoptional_assembly_adaptor
     * @brief Dense multidimensional adaptor holding optional values,
     * optimized for tensor operations.
     *
     * The xoptional_assembly_adaptor class implements a dense multidimensional adaptor
     * hoding optional values. It is used to provide an optional expression semantic
     * to two tensor expressions, one holding the value, the hother holding the missing mask.
     *
     * @tparam VEC The closure for the type of expression holding the values.
     * @tparam FE The closure for the type of expression holding the missing mask.
     */
    template <class VEC, class FEC>
    class xoptional_assembly_adaptor : public xoptional_assembly_base<xoptional_assembly_adaptor<VEC, FEC>>,
                                       public xcontainer_semantic<xoptional_assembly_adaptor<VEC, FEC>>
    {
    public:

        using self_type = xoptional_assembly_adaptor<VEC, FEC>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using storage_type = typename base_type::storage_type;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using const_value_expression = typename base_type::const_value_expression;
        using const_flag_expression = typename base_type::const_flag_expression;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using temporary_type = typename semantic_base::temporary_type;

        using assembly_type = base_type;

        template <class OVE, class OFE>
        xoptional_assembly_adaptor(OVE&& ve, OFE&& fe);

        ~xoptional_assembly_adaptor() = default;

        xoptional_assembly_adaptor(const xoptional_assembly_adaptor&);
        xoptional_assembly_adaptor& operator=(const xoptional_assembly_adaptor&);

        xoptional_assembly_adaptor(xoptional_assembly_adaptor&&);
        xoptional_assembly_adaptor& operator=(xoptional_assembly_adaptor&&);
        xoptional_assembly_adaptor& operator=(temporary_type&&);

        template <class E>
        xoptional_assembly_adaptor& operator=(const xexpression<E>& e);

    private:

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        value_expression value_impl() noexcept;
        const_value_expression value_impl() const noexcept;

        flag_expression has_value_impl() noexcept;
        const_flag_expression has_value_impl() const noexcept;

        VEC m_value;
        FEC m_has_value;
        storage_type m_storage;

        friend class xoptional_assembly_base<xoptional_assembly_adaptor<VEC, FEC>>;
    };

    /*************************************
     * xoptional_assembly implementation *
     *************************************/

    namespace detail
    {
        template <class T, class S>
        inline void nested_optional_copy(T&& iter, const S& s)
        {
            iter->value() = s.value();
            iter->has_value() = s.has_value();
            ++iter;
        }

        template <class T, class S>
        inline void nested_optional_copy(T&& iter, std::initializer_list<S> s)
        {
            for (auto it = s.begin(); it != s.end(); ++it)
            {
                nested_optional_copy(std::forward<T>(iter), *it);
            }
        }
    }

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xoptional_assembly that holds 0 element.
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly()
        : m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and
     * layout_type.
     * @param shape the shape of the xoptional_assembly
     * @param l the layout_type of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, layout_type l)
        : m_value(shape, l)
        , m_has_value(shape, l)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an xoptional_assembly with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xoptional_assembly
     * @param value the value of the elements
     * @param l the layout_type of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(
        const shape_type& shape,
        const value_type& value,
        layout_type l
    )
        : m_value(shape, value.value(), l)
        , m_has_value(shape, value.has_value(), l)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and strides.
     * @param shape the shape of the xoptional_assembly
     * @param strides the strides of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, const strides_type& strides)
        : m_value(shape, strides)
        , m_has_value(shape, strides)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an uninitialized xoptional_assembly with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xoptional_assembly
     * @param strides the strides of the xoptional_assembly
     * @param value the value of the elements
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(
        const shape_type& shape,
        const strides_type& strides,
        const value_type& value
    )
        : m_value(shape, strides, value.value())
        , m_has_value(shape, strides, value.has_value())
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an xoptional_assembly that holds a single element initialized to the
     * specified value.
     * @param value the value of the element
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const value_type& value)
        : m_value(value.value())
        , m_has_value(value.has_value())
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression. The flag
     * expression is initialized as if no value is missing.
     * @param ve the expression holding the values
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const VE& ve)
        : m_value(ve)
        , m_has_value(ve.shape(), true, ve.layout())
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression. The flag
     * expression is initialized as if no value is missing. The value expression is
     * moved inside the xoptional_assembly and is therefore not available after
     * the xoptional_assembly has been constructed.
     * @param ve the expression holding the values
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(VE&& ve)
        : m_value(std::move(ve))
        , m_has_value(ve.shape(), true, ve.layout())
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    /**
     * Allocates an xoptional_assembly from the specified value expression and missing
     * mask expression.
     * @param ove the expression holding the values
     * @param ofe the expression holding the missing mask
     */
    template <class VE, class FE>
    template <class OVE, class OFE, typename>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(OVE&& ove, OFE&& ofe)
        : m_value(std::forward<OVE>(ove))
        , m_has_value(std::forward<OFE>(ofe))
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    //@}

    /**
     * @name Constructors from initializer list
     */
    //@{
    /**
     * Allocates a one-dimensional xoptional_assembly.
     * @param t the elements of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 1> t)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        base_type::resize(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major
                         && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->linear_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a two-dimensional xoptional_assembly.
     * @param t the elements of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 2> t)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        base_type::resize(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major
                         && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->linear_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a three-dimensional xoptional_assembly.
     * @param t the elements of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 3> t)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        base_type::resize(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major
                         && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->linear_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a four-dimensional xoptional_assembly.
     * @param t the elements of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 4> t)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        base_type::resize(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major
                         && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->linear_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a five-dimensional xoptional_assembly.
     * @param t the elements of the xoptional_assembly
     */
    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 5> t)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        base_type::resize(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major
                         && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->linear_begin(), t)
                  : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    //@}

    /**
     * Allocates and returns an xoptional_assembly with the specified shape.
     * @param s the shape of the xoptional_assembly
     */
    template <class VE, class FE>
    template <class S>
    inline xoptional_assembly<VE, FE> xoptional_assembly<VE, FE>::from_shape(S&& s)
    {
        shape_type shape = xtl::forward_sequence<shape_type, S>(s);
        return self_type(shape);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const self_type& rhs)
        : base_type()
        , semantic_base()
        , m_value(rhs.m_value)
        , m_has_value(rhs.m_has_value)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(self_type&& rhs)
        : base_type()
        , semantic_base()
        , m_value(rhs.m_value)
        , m_has_value(rhs.m_has_value)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::operator=(const self_type& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_value = rhs.m_value;
        m_has_value = rhs.m_has_value;
        return *this;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::operator=(self_type&& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_value = std::move(rhs.m_value);
        m_has_value = std::move(rhs.m_has_value);
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class VE, class FE>
    template <class E>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const xexpression<E>& e)
        : base_type()
        , m_value()
        , m_has_value()
        , m_storage(m_value.storage(), m_has_value.storage())
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class VE, class FE>
    template <class E>
    inline auto xoptional_assembly<VE, FE>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::value_impl() noexcept -> value_expression
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::value_impl() const noexcept -> const_value_expression
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::has_value_impl() noexcept -> flag_expression
    {
        return m_has_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::has_value_impl() const noexcept -> const_flag_expression
    {
        return m_has_value;
    }

    /*********************************************
     * xoptional_assembly_adaptor implementation *
     *********************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xoptional_assembly_adaptor of the given value and
     * missing mask expressions.
     * @param ve the expression holding the values
     * @param fe the expression holding the missing mask
     */
    template <class VEC, class FEC>
    template <class OVE, class OFE>
    inline xoptional_assembly_adaptor<VEC, FEC>::xoptional_assembly_adaptor(OVE&& ve, OFE&& fe)
        : m_value(std::forward<OVE>(ve))
        , m_has_value(std::forward<OFE>(fe))
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    //@}

    template <class VEC, class FEC>
    inline xoptional_assembly_adaptor<VEC, FEC>::xoptional_assembly_adaptor(const self_type& rhs)
        : base_type()
        , semantic_base()
        , m_value(rhs.m_value)
        , m_has_value(rhs.m_has_value)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::operator=(const self_type& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_value = rhs.m_value;
        m_has_value = rhs.m_has_value;
        return *this;
    }

    template <class VEC, class FEC>
    inline xoptional_assembly_adaptor<VEC, FEC>::xoptional_assembly_adaptor(self_type&& rhs)
        : base_type()
        , semantic_base()
        , m_value(rhs.m_value)
        , m_has_value(rhs.m_has_value)
        , m_storage(m_value.storage(), m_has_value.storage())
    {
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::operator=(self_type&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_value = rhs.m_value;
        m_has_value = rhs.m_has_value;
        return *this;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::operator=(temporary_type&& tmp) -> self_type&
    {
        m_value = std::move(tmp.value());
        m_has_value = std::move(tmp.has_value());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class VEC, class FEC>
    template <class E>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::value_impl() noexcept -> value_expression
    {
        return m_value;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::value_impl() const noexcept -> const_value_expression
    {
        return m_value;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::has_value_impl() noexcept -> flag_expression
    {
        return m_has_value;
    }

    template <class VEC, class FEC>
    inline auto xoptional_assembly_adaptor<VEC, FEC>::has_value_impl() const noexcept -> const_flag_expression
    {
        return m_has_value;
    }
}

#endif
