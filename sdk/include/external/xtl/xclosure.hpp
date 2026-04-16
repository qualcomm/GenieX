/***************************************************************************
* Copyright (c) Sylvain Corlay and Johan Mabille and Wolf Vollprecht       *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_CLOSURE_HPP
#define XTL_CLOSURE_HPP

#include <memory>
#include <type_traits>
#include <utility>

#include "xtl_config.hpp"

namespace xtl
{

#ifdef __cpp_lib_as_const
    using std::as_const;
#else
    template <class T>
    constexpr std::add_const_t<T>& as_const(T& t) noexcept
    {
        return t;
    }

    template <class T>
    constexpr std::add_const_t<T&&>& as_const(T&& t) noexcept = delete;
#endif

    /****************
     * closure_type *
     ****************/

    template <class S>
    struct closure_type
    {
        using underlying_type = std::conditional_t<std::is_const<std::remove_reference_t<S>>::value,
                                                   const std::decay_t<S>,
                                                   std::decay_t<S>>;
        using type = typename std::conditional<std::is_lvalue_reference<S>::value,
                                               underlying_type&,
                                               underlying_type>::type;
    };

    template <class S>
    using closure_type_t = typename closure_type<S>::type;

    template <class S>
    struct const_closure_type
    {
        using underlying_type = std::decay_t<S>;
        using type = typename std::conditional<std::is_lvalue_reference<S>::value,
                                               std::add_const_t<underlying_type>&,
                                               underlying_type>::type;
    };

    template <class S>
    using const_closure_type_t = typename const_closure_type<S>::type;

    /********************
     * ptr_closure_type *
     ********************/

    template <class S>
    struct ptr_closure_type
    {
        using underlying_type = std::conditional_t<std::is_const<std::remove_reference_t<S>>::value,
                                                   const std::decay_t<S>,
                                                   std::decay_t<S>>;
        using type = std::conditional_t<std::is_lvalue_reference<S>::value,
                                        underlying_type*,
                                        underlying_type>;
    };

    template <class S>
    using ptr_closure_type_t = typename ptr_closure_type<S>::type;

    template <class S>
    struct const_ptr_closure_type
    {
        using underlying_type = const std::decay_t<S>;
        using type = std::conditional_t<std::is_lvalue_reference<S>::value,
                                        underlying_type*,
                                        underlying_type>;
    };

    template <class S>
    using const_ptr_closure_type_t = typename const_ptr_closure_type<S>::type;

    /********************
     * xclosure_wrapper *
     ********************/

    template <class CT>
    class xclosure_wrapper
    {
    public:

        using self_type = xclosure_wrapper<CT>;
        using closure_type = CT;
        using const_closure_type = std::add_const_t<CT>;
        using value_type = std::decay_t<CT>;

        using reference = std::conditional_t<
            std::is_const<std::remove_reference_t<CT>>::value,
            const value_type&, value_type&
        >;

        using pointer = std::conditional_t<
            std::is_const<std::remove_reference_t<CT>>::value,
            const value_type*, value_type*
        >;

        xclosure_wrapper(value_type&& e);
        xclosure_wrapper(reference e);

        xclosure_wrapper(const self_type& rhs) = default;
        xclosure_wrapper(self_type&& rhs) = default;

        self_type& operator=(const self_type& rhs);
        self_type& operator=(self_type&& rhs);
        
        template <class T>
        self_type& operator=(T&&);

        operator closure_type() noexcept;
        operator const_closure_type() const noexcept;

        std::add_lvalue_reference_t<closure_type> get() & noexcept;
        std::add_lvalue_reference_t<std::add_const_t<closure_type>> get() const & noexcept;
        closure_type get() && noexcept;

        pointer operator&() noexcept;

        bool equal(const self_type& rhs) const;
        void swap(self_type& rhs);

    private:

        using storing_type = ptr_closure_type_t<CT>;
        storing_type m_wrappee;

        template <class T>
        std::enable_if_t<std::is_lvalue_reference<CT>::value, std::add_lvalue_reference_t<std::remove_pointer_t<T>>>
        deref(T val) const;

        template <class T>
        std::enable_if_t<!std::is_lvalue_reference<CT>::value, std::add_lvalue_reference_t<T>>
        deref(T& val) const;

        template <class T>
        std::enable_if_t<std::is_lvalue_reference<CT>::value, T>
        get_pointer(T val) const;

        template <class T>
        std::enable_if_t<!std::is_lvalue_reference<CT>::value, std::add_pointer_t<T>>
        get_pointer(T& val) const;

        template <class T, class CTA>
        std::enable_if_t<std::is_lvalue_reference<CT>::value, T>
        get_storage_init(CTA&& e) const;

        template <class T, class CTA>
        std::enable_if_t<!std::is_lvalue_reference<CT>::value, T>
        get_storage_init(CTA&& e) const;
    };

    // TODO: remove this (backward compatibility)
    template <class CT>
    using closure_wrapper = xclosure_wrapper<CT>;

    /********************
     * xclosure_pointer *
     ********************/

    template <class CT>
    class xclosure_pointer
    {
    public:

        using self_type = xclosure_pointer<CT>;
        using closure_type = CT;
        using value_type = std::decay_t<CT>;

        using reference = std::conditional_t<
            std::is_const<std::remove_reference_t<CT>>::value,
            const value_type&, value_type&
        >;

        using const_reference = const value_type&;

        using pointer = std::conditional_t<
            std::is_const<std::remove_reference_t<CT>>::value,
            const value_type*, value_type*
        >;

        xclosure_pointer(value_type&& e);
        xclosure_pointer(reference e);

        reference operator*() noexcept;
        const_reference operator*() const noexcept;
        pointer operator->() const noexcept;

    private:

        using storing_type = closure_type_t<CT>;
        storing_type m_wrappee;
    };

    /***********************************
     * xclosure_wrapper implementation *
     ***********************************/

    template <class CT>
    inline xclosure_wrapper<CT>::xclosure_wrapper(value_type&& e)
        : m_wrappee(get_storage_init<storing_type>(std::move(e)))
    {
    }

    template <class CT>
    inline xclosure_wrapper<CT>::xclosure_wrapper(reference e)
        : m_wrappee(get_storage_init<storing_type>(e))
    {
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::operator=(const self_type& rhs) -> self_type&
    {
        deref(m_wrappee) = deref(rhs.m_wrappee);
        return *this;
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::operator=(self_type&& rhs) -> self_type&
    {
        swap(rhs);
        return *this;
    }

    template <class CT>
    template <class T>
    inline auto xclosure_wrapper<CT>::operator=(T&& t) -> self_type&
    {
        deref(m_wrappee) = std::forward<T>(t);
        return *this;
    }

    template <class CT>
    inline xclosure_wrapper<CT>::operator typename xclosure_wrapper<CT>::closure_type() noexcept
    {
        return deref(m_wrappee);
    }

    template <class CT>
    inline xclosure_wrapper<CT>::operator typename xclosure_wrapper<CT>::const_closure_type() const noexcept
    {
        return deref(m_wrappee);
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::get() & noexcept -> std::add_lvalue_reference_t<closure_type>
    {
        return deref(m_wrappee);
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::get() const & noexcept -> std::add_lvalue_reference_t<std::add_const_t<closure_type>>
    {
        return deref(m_wrappee);
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::get() && noexcept -> closure_type
    {
        return deref(m_wrappee);
    }

    template <class CT>
    inline auto xclosure_wrapper<CT>::operator&() noexcept -> pointer
    {
        return get_pointer(m_wrappee);
    }

    template <class CT>
    template <class T>
    inline std::enable_if_t<std::is_lvalue_reference<CT>::value, std::add_lvalue_reference_t<std::remove_pointer_t<T>>>
    xclosure_wrapper<CT>::deref(T val) const
    {
        return *val;
    }

    template <class CT>
    template <class T>
    inline std::enable_if_t<!std::is_lvalue_reference<CT>::value, std::add_lvalue_reference_t<T>>
    xclosure_wrapper<CT>::deref(T& val) const
    {
        return val;
    }

    template <class CT>
    template <class T>
    inline std::enable_if_t<std::is_lvalue_reference<CT>::value, T>
    xclosure_wrapper<CT>::get_pointer(T val) const
    {
        return val;
    }

    template <class CT>
    template <class T>
    inline std::enable_if_t<!std::is_lvalue_reference<CT>::value, std::add_pointer_t<T>>
    xclosure_wrapper<CT>::get_pointer(T& val) const
    {
        return &val;
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<std::is_lvalue_reference<CT>::value, T>
    xclosure_wrapper<CT>::get_storage_init(CTA&& e) const
    {
        return &e;
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<!std::is_lvalue_reference<CT>::value, T>
    xclosure_wrapper<CT>::get_storage_init(CTA&& e) const
    {
        return e;
    }

    template <class CT>
    inline bool xclosure_wrapper<CT>::equal(const self_type& rhs) const
    {
        return deref(m_wrappee) == rhs.deref(rhs.m_wrappee);
    }

    template <class CT>
    inline void xclosure_wrapper<CT>::swap(self_type& rhs)
    {
        using std::swap;
        swap(deref(m_wrappee), deref(rhs.m_wrappee));
    }

    template <class CT>
    inline bool operator==(const xclosure_wrapper<CT>& lhs, const xclosure_wrapper<CT>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class CT>
    inline bool operator!=(const xclosure_wrapper<CT>& lhs, const xclosure_wrapper<CT>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class CT>
    inline void swap(xclosure_wrapper<CT>& lhs, xclosure_wrapper<CT>& rhs)
    {
        lhs.swap(rhs);
    }

    /***********************************
     * xclosure_pointer implementation *
     ***********************************/

    template <class CT>
    inline xclosure_pointer<CT>::xclosure_pointer(value_type&& e)
        : m_wrappee(std::move(e))
    {
    }

    template <class CT>
    inline xclosure_pointer<CT>::xclosure_pointer(reference e)
        : m_wrappee(e)
    {
    }

    template <class CT>
    inline auto xclosure_pointer<CT>::operator*() noexcept -> reference
    {
        return m_wrappee;
    }

    template <class CT>
    inline auto xclosure_pointer<CT>::operator*() const noexcept -> const_reference
    {
        return m_wrappee;
    }

    template <class CT>
    inline auto xclosure_pointer<CT>::operator->() const noexcept -> pointer
    {
        return const_cast<pointer>(std::addressof(m_wrappee));
    }

    /*****************************
     * closure and const_closure *
     *****************************/

    template <class T>
    inline decltype(auto) closure(T&& t)
    {
        return xclosure_wrapper<closure_type_t<T>>(std::forward<T>(t));
    }

    template <class T>
    inline decltype(auto) const_closure(T&& t)
    {
        return xclosure_wrapper<const_closure_type_t<T>>(std::forward<T>(t));
    }

    /*********************************************
     * closure_pointer and const_closure_pointer *
     *********************************************/

    template <class T>
    inline auto closure_pointer(T&& t)
    {
        return xclosure_pointer<closure_type_t<T>>(std::forward<T>(t));
    }

    template <class T>
    inline auto const_closure_pointer(T&& t)
    {
        return xclosure_pointer<const_closure_type_t<T>>(std::forward<T>(t));
    }
}

#endif
