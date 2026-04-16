/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_HIERARCHY_GENERATOR_HPP
#define XTL_HIERARCHY_GENERATOR_HPP

#include "xmeta_utils.hpp"

namespace xtl
{

    /*********************************
     * scattered hierarchy generator *
     *********************************/

    template <class TL, template <class> class U>
    class xscatter_hierarchy_generator;

    template <template <class> class U, class T, class... Args>
    class xscatter_hierarchy_generator<mpl::vector<T, Args...>, U>
        : public U<T>, public xscatter_hierarchy_generator<mpl::vector<Args...>, U>
    {
    };

    template <template <class> class U>
    class xscatter_hierarchy_generator<mpl::vector<>, U>
    {
    };

    /******************************
     * linear hierarchy generator *
     ******************************/

    class default_root {};

    template <class TL, template <class, class> class U, class Root = default_root>
    class xlinear_hierarchy_generator;

    template <template <class, class> class U, class Root, class T0, class... Args>
    class xlinear_hierarchy_generator<mpl::vector<T0, Args...>, U, Root>
        : public U<T0, xlinear_hierarchy_generator<mpl::vector<Args...>, U, Root>>
    {
    public:

        using base_type = U<T0, xlinear_hierarchy_generator<mpl::vector<Args...>, U, Root>>;
        template <class... T>
        inline xlinear_hierarchy_generator(T&&... args)
            : base_type(std::forward<T>(args)...)
        {
        }
    };

    template <template <class, class> class U, class Root>
    class xlinear_hierarchy_generator<mpl::vector<>, U, Root>
        : public Root
    {
    public:

        template <class... T>
        inline xlinear_hierarchy_generator(T&&... args)
            : Root(std::forward<T>(args)...)
        {
        }
    };
}

#endif
