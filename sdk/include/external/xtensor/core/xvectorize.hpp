/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VECTORIZE_HPP
#define XTENSOR_VECTORIZE_HPP

#include <type_traits>
#include <utility>

#include "../core/xfunction.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /***************
     * xvectorizer *
     ***************/

    template <class F, class R>
    class xvectorizer
    {
    public:

        template <class... E>
        using xfunction_type = xfunction<F, xclosure_t<E>...>;

        template <class Func, class = std::enable_if_t<!std::is_same<std::decay_t<Func>, xvectorizer>::value>>
        xvectorizer(Func&& f);

        template <class... E>
        xfunction_type<E...> operator()(E&&... e) const;

    private:

        typename std::remove_reference<F>::type m_f;
    };

    namespace detail
    {
        template <class F>
        using get_function_type = remove_class_t<decltype(&std::remove_reference_t<F>::operator())>;
    }

    template <class R, class... Args>
    xvectorizer<R (*)(Args...), R> vectorize(R (*f)(Args...));

    template <class F, class R, class... Args>
    xvectorizer<F, R> vectorize(F&& f, R (*)(Args...));

// Workaround for Visual Studio 15.7.1.
// Error C2668 (ambiguous call to overloaded function) mistaking a declarations
// for the definition of another overload.
#ifndef _MSC_VER
    template <class F>
    auto vectorize(F&& f)
        -> decltype(vectorize(std::forward<F>(f), std::declval<detail::get_function_type<F>*>()));
#endif

    /******************************
     * xvectorizer implementation *
     ******************************/

    template <class F, class R>
    template <class Func, class>
    inline xvectorizer<F, R>::xvectorizer(Func&& f)
        : m_f(std::forward<Func>(f))
    {
    }

    template <class F, class R>
    template <class... E>
    inline auto xvectorizer<F, R>::operator()(E&&... e) const -> xfunction_type<E...>
    {
        return xfunction_type<E...>(m_f, std::forward<E>(e)...);
    }

    template <class R, class... Args>
    inline xvectorizer<R (*)(Args...), R> vectorize(R (*f)(Args...))
    {
        return xvectorizer<R (*)(Args...), R>(f);
    }

    template <class F, class R, class... Args>
    inline xvectorizer<F, R> vectorize(F&& f, R (*)(Args...))
    {
        return xvectorizer<F, R>(std::forward<F>(f));
    }

    template <class F>
    inline auto vectorize(F&& f)
        -> decltype(vectorize(std::forward<F>(f), std::declval<detail::get_function_type<F>*>()))
    {
        return vectorize(std::forward<F>(f), static_cast<detail::get_function_type<F>*>(nullptr));
    }
}

#endif
