/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XPROXY_WRAPPER_HPP
#define XTL_XPROXY_WRAPPER_HPP

#include "xclosure.hpp"

namespace xtl
{

    template <class P>
    class xproxy_wrapper_impl : public P
    {
    public:

        using self_type = xproxy_wrapper_impl<P>;
        using lv_pointer = xclosure_pointer<P&>;
        using rv_pointer = xclosure_pointer<P>;

        explicit xproxy_wrapper_impl(P&& rhs)
            : P(std::move(rhs))
        {
        }

        inline lv_pointer operator&() & { return lv_pointer(*this); }
        inline rv_pointer operator&() && { return rv_pointer(std::move(*this)); }
    };

    template <class P>
    using xproxy_wrapper = std::conditional_t<std::is_class<P>::value,
                                              xproxy_wrapper_impl<P>,
                                              xclosure_wrapper<P>>;

    template <class P>
    inline xproxy_wrapper<P> proxy_wrapper(P&& proxy)
    {
        return xproxy_wrapper<P>(std::forward<P>(proxy));
    }
}

#endif

