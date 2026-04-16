/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_NOALIAS_HPP
#define XTENSOR_NOALIAS_HPP

#include "../core/xsemantic.hpp"

namespace xt
{

    template <class A>
    class noalias_proxy
    {
    public:

        noalias_proxy(A a) noexcept;

        template <class E>
        disable_xexpression<E, A> operator=(const E&);

        template <class E>
        disable_xexpression<E, A> operator+=(const E&);

        template <class E>
        disable_xexpression<E, A> operator-=(const E&);

        template <class E>
        disable_xexpression<E, A> operator*=(const E&);

        template <class E>
        disable_xexpression<E, A> operator/=(const E&);

        template <class E>
        disable_xexpression<E, A> operator%=(const E&);

        template <class E>
        disable_xexpression<E, A> operator&=(const E&);

        template <class E>
        disable_xexpression<E, A> operator|=(const E&);

        template <class E>
        disable_xexpression<E, A> operator^=(const E&);

        template <class E>
        A operator=(const xexpression<E>& e);

        template <class E>
        A operator+=(const xexpression<E>& e);

        template <class E>
        A operator-=(const xexpression<E>& e);

        template <class E>
        A operator*=(const xexpression<E>& e);

        template <class E>
        A operator/=(const xexpression<E>& e);

        template <class E>
        A operator%=(const xexpression<E>& e);

        template <class E>
        A operator&=(const xexpression<E>&);

        template <class E>
        A operator|=(const xexpression<E>&);

        template <class E>
        A operator^=(const xexpression<E>&);

    private:

        A m_array;
    };

    template <class A>
    noalias_proxy<xtl::closure_type_t<A>> noalias(A&& a) noexcept;

    /********************************
     * noalias_proxy implementation *
     ********************************/

    template <class A>
    inline noalias_proxy<A>::noalias_proxy(A a) noexcept
        : m_array(std::forward<A>(a))
    {
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.assign(xscalar<E>(e));
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator+=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::plus<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator-=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::minus<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator*=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::multiplies<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator/=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::divides<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator%=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::modulus<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator&=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_and<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator|=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_or<>());
    }

    template <class A>
    template <class E>
    inline auto noalias_proxy<A>::operator^=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_xor<>());
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator=(const xexpression<E>& e)
    {
        return m_array.assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator+=(const xexpression<E>& e)
    {
        return m_array.plus_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator-=(const xexpression<E>& e)
    {
        return m_array.minus_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator*=(const xexpression<E>& e)
    {
        return m_array.multiplies_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator/=(const xexpression<E>& e)
    {
        return m_array.divides_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator%=(const xexpression<E>& e)
    {
        return m_array.modulus_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator&=(const xexpression<E>& e)
    {
        return m_array.bit_and_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator|=(const xexpression<E>& e)
    {
        return m_array.bit_or_assign(e);
    }

    template <class A>
    template <class E>
    inline A noalias_proxy<A>::operator^=(const xexpression<E>& e)
    {
        return m_array.bit_xor_assign(e);
    }

    template <class A>
    inline noalias_proxy<xtl::closure_type_t<A>> noalias(A&& a) noexcept
    {
        return noalias_proxy<xtl::closure_type_t<A>>(a);
    }
}

#endif
