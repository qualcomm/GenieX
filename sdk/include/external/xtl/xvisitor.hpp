/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_VISITOR_HPP
#define XTL_VISITOR_HPP

#include <stdexcept>
#include "xmeta_utils.hpp"

namespace xtl
{
    // Loki's visitor ported to C++14
    // Original implementation can be found at:
    // https://github.com/snaewe/loki-lib/blob/master/include/loki/Visitor.h

    /****************
     * base_visitor *
     ****************/

    class base_visitor
    {
    public:

        virtual ~base_visitor() = default;
    };

    /***********
     * visitor *
     ***********/

    template <class T, class R = void, bool is_const = true>
    class visitor
    {
    public:

        using return_type = R;
        using param_type = std::conditional_t<is_const, const T, T>;

        virtual ~visitor() = default;

        virtual return_type visit(param_type&) = 0;
    };

    template <class R, bool is_const>
    class visitor<mpl::vector<>, R, is_const>
    {
    };

    template <class R, bool is_const, class T, class... U>
    class visitor<mpl::vector<T, U...>, R, is_const>
        : public visitor<T, R, is_const>
        , public visitor<mpl::vector<U...>, R, is_const>
    {
    };

    /**********************
     * catch_all policies *
     **********************/

    template <class R, class T>
    struct default_catch_all
    {
        static R on_unknown_visitor(T&, base_visitor&)
        {
            return R();
        }
    };

    template <class R, class T>
    struct throwing_catch_all
    {
        static R on_unknown_visitor(T&, base_visitor&)
        {
            XTL_THROW(std::runtime_error, "Unknown visited type");
        }
    };

    /******************
     * base_visitable *
     ******************/

    template
    <
        class R = void,
        bool const_visitable = false,
        template <class, class> class catch_all = default_catch_all
    >
    class base_visitable;

    template <class R, template <class, class> class catch_all>
    class base_visitable<R, false, catch_all>
    {
    public:

        using return_type = R;

        virtual ~base_visitable() = default;
        virtual return_type accept(base_visitor&) = 0;

    protected:

        template <class T>
        static return_type accept_impl(T& visited, base_visitor& vis)
        {
            if (auto* p = dynamic_cast<visitor<T, R, false>*>(&vis))
            {
                return p->visit(visited);
            }
            return catch_all<R, T>::on_unknown_visitor(visited, vis);
        }
    };

    template <class R, template <class, class> class catch_all>
    class base_visitable<R, true, catch_all>
    {
    public:

        using return_type = R;

        virtual ~base_visitable() = default;
        virtual return_type accept(base_visitor&) const = 0;

    protected:

        template <class T>
        static return_type accept_impl(const T& visited, base_visitor& vis)
        {
            if (auto* p = dynamic_cast<visitor<T, R, true>*>(&vis))
            {
                return p->visit(visited);
            }
            return catch_all<R, const T>::on_unknown_visitor(visited, vis);
        }
    };

    /************************
     * XTL_DEFINE_VISITABLE *
     ************************/

#define XTL_DEFINE_VISITABLE() \
    return_type accept(::xtl::base_visitor& vis) override \
    { return accept_impl(*this, vis); }

#define XTL_DEFINE_CONST_VISITABLE() \
    return_type accept(::xtl::base_visitor& vis) const override \
    { return accept_impl(*this, vis); }

    /******************
     * cyclic_visitor *
     ******************/

    template <class T, class R, bool is_const = true>
    class cyclic_visitor;

    template <class R, bool is_const, class... T>
    class cyclic_visitor<mpl::vector<T...>, R, is_const>
        : public visitor<mpl::vector<T...>, R, is_const>
    {
    public:

        using return_type = R;

        template <class V>
        return_type generic_visit(V& visited)
        {
            visitor<std::remove_const_t<V>, return_type, is_const>& sub_obj = *this;
            return sub_obj.visit(visited);
        }
    };

    /*******************************
     * XTL_DEFINE_CYCLIC_VISITABLE *
     *******************************/

#define XTL_DEFINE_CYCLIC_VISITABLE(some_visitor)                     \
    virtual some_visitor::return_type accept(some_visitor& vis)       \
    {                                                                 \
        return vis.generic_visit(*this);                              \
    }

#define XTL_DEFINE_CONST_CYCLIC_VISITABLE(some_visitor)               \
    virtual some_visitor::return_type accept(some_visitor& vis) const \
    {                                                                 \
        return vis.generic_visit(*this);                              \
    }
}

#endif

