// https://github.com/tcbrindle/span/blob/master/include/tcb/span.hpp
// TCP SPAN @commit cd0c6d0

/*
This is an implementation of std::span from P0122R7
http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0122r7.pdf
*/

//          Copyright Tristan Brindle 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../../LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#ifndef TCB_SPAN_HPP_INCLUDED
#define TCB_SPAN_HPP_INCLUDED

#include <array>
#include <cstddef>
#include <exception>
#include <type_traits>

#ifndef TCB_SPAN_NO_EXCEPTIONS
// Attempt to discover whether we're being compiled with exception support
#if !(defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define TCB_SPAN_NO_EXCEPTIONS
#endif
#endif

#ifndef TCB_SPAN_NO_EXCEPTIONS
#include <cstdio>
#include <stdexcept>
#endif

// Various feature test macros

#ifndef TCB_SPAN_NAMESPACE_NAME
#define TCB_SPAN_NAMESPACE_NAME tcb
#endif

#ifdef TCB_SPAN_STD_COMPLIANT_MODE
#define TCB_SPAN_NO_DEPRECATION_WARNINGS
#endif

#ifndef TCB_SPAN_NO_DEPRECATION_WARNINGS
#define TCB_SPAN_DEPRECATED_FOR(msg) [[deprecated(msg)]]
#else
#define TCB_SPAN_DEPRECATED_FOR(msg)
#endif

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#define TCB_SPAN_HAVE_CPP17
#endif

#if __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
#define TCB_SPAN_HAVE_CPP14
#endif

namespace TCB_SPAN_NAMESPACE_NAME {

// Establish default contract checking behavior
#if !defined(TCB_SPAN_THROW_ON_CONTRACT_VIOLATION) &&                          \
    !defined(TCB_SPAN_TERMINATE_ON_CONTRACT_VIOLATION) &&                      \
    !defined(TCB_SPAN_NO_CONTRACT_CHECKING)
#if defined(NDEBUG) || !defined(TCB_SPAN_HAVE_CPP14)
#define TCB_SPAN_NO_CONTRACT_CHECKING
#else
#define TCB_SPAN_TERMINATE_ON_CONTRACT_VIOLATION
#endif
#endif

#if defined(TCB_SPAN_THROW_ON_CONTRACT_VIOLATION)
struct contract_violation_error : std::logic_error {
    explicit contract_violation_error(const char* msg) : std::logic_error(msg)
    {}
};

inline void contract_violation(const char* msg)
{
    throw contract_violation_error(msg);
}

#elif defined(TCB_SPAN_TERMINATE_ON_CONTRACT_VIOLATION)
[[noreturn]] inline void contract_violation(const char* /*unused*/)
{
    std::terminate();
}
#endif

#if !defined(TCB_SPAN_NO_CONTRACT_CHECKING)
#define TCB_SPAN_STRINGIFY(cond) #cond
#define TCB_SPAN_EXPECT(cond)                                                  \
    cond ? (void) 0 : contract_violation("Expected " TCB_SPAN_STRINGIFY(cond))
#else
#define TCB_SPAN_EXPECT(cond)
#endif

#if defined(TCB_SPAN_HAVE_CPP17) || defined(__cpp_inline_variables)
#define TCB_SPAN_INLINE_VAR inline
#else
#define TCB_SPAN_INLINE_VAR
#endif

#if defined(TCB_SPAN_HAVE_CPP14) ||                                                 \
    (defined(__cpp_constexpr) && __cpp_constexpr >= 201304)
#define TCB_SPAN_CONSTEXPR14 constexpr
#else
#define TCB_SPAN_CONSTEXPR14
#endif

#if defined(TCB_SPAN_NO_CONTRACT_CHECKING)
#define TCB_SPAN_CONSTEXPR11 constexpr
#else
#define TCB_SPAN_CONSTEXPR11 TCB_SPAN_CONSTEXPR14
#endif

#if defined(TCB_SPAN_HAVE_CPP17) || defined(__cpp_deduction_guides)
#define TCB_SPAN_HAVE_DEDUCTION_GUIDES
#endif

#if defined(TCB_SPAN_HAVE_CPP17) || defined(__cpp_lib_byte) && !(defined(_HAS_STD_BYTE) && !_HAS_STD_BYTE)
#define TCB_SPAN_HAVE_STD_BYTE
#endif

#if defined(TCB_SPAN_HAVE_CPP17) || defined(__cpp_lib_array_constexpr)
#define TCB_SPAN_HAVE_CONSTEXPR_STD_ARRAY_ETC
#endif

#if defined(TCB_SPAN_HAVE_CONSTEXPR_STD_ARRAY_ETC)
#define TCB_SPAN_ARRAY_CONSTEXPR constexpr
#else
#define TCB_SPAN_ARRAY_CONSTEXPR
#endif

#ifdef TCB_SPAN_HAVE_STD_BYTE
using byte = std::byte;
#else
using byte = unsigned char;
#endif

TCB_SPAN_INLINE_VAR constexpr std::ptrdiff_t dynamic_extent = -1;

template <typename ElementType, std::ptrdiff_t Extent = dynamic_extent>
class span;

namespace detail {

template <typename E, std::ptrdiff_t S>
struct span_storage {
    constexpr span_storage() noexcept = default;

    constexpr span_storage(E* aptr, std::ptrdiff_t /*unused*/) noexcept
        : ptr(aptr)
    {}

    E* ptr = nullptr;
    static constexpr std::ptrdiff_t size = S;
};

template <typename E>
struct span_storage<E, dynamic_extent> {
    constexpr span_storage() noexcept = default;

    constexpr span_storage(E* aptr, std::size_t asize) noexcept
        : ptr(aptr), size(asize)
    {}

    E* ptr = nullptr;
    std::size_t size = 0;
};

// Reimplementation of C++17 std::size() and std::data()
#if defined(TCB_SPAN_HAVE_CPP17) ||                                            \
    defined(__cpp_lib_nonmember_container_access)
using std::data;
using std::size;
#else
template <class C>
constexpr auto size(const C& c) -> decltype(c.size())
{
    return c.size();
}

template <class T, std::size_t N>
constexpr std::size_t size(const T (&)[N]) noexcept
{
    return N;
}

template <class C>
constexpr auto data(C& c) -> decltype(c.data())
{
    return c.data();
}

template <class C>
constexpr auto data(const C& c) -> decltype(c.data())
{
    return c.data();
}

template <class T, std::size_t N>
constexpr T* data(T (&array)[N]) noexcept
{
    return array;
}

template <class E>
constexpr const E* data(std::initializer_list<E> il) noexcept
{
    return il.begin();
}
#endif // TCB_SPAN_HAVE_CPP17

#if defined(TCB_SPAN_HAVE_CPP17) || defined(__cpp_lib_void_t)
using std::void_t;
#else
template <typename...>
using void_t = void;
#endif

template <typename T>
using uncvref_t =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename>
struct is_span : std::false_type {};

template <typename T, std::ptrdiff_t S>
struct is_span<span<T, S>> : std::true_type {};

template <typename>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename, typename = void>
struct has_size_and_data : std::false_type {};

template <typename T>
struct has_size_and_data<T, void_t<decltype(detail::size(std::declval<T>())),
                                   decltype(detail::data(std::declval<T>()))>>
    : std::true_type {};

template <typename C, typename U = uncvref_t<C>>
struct is_container {
    static constexpr bool value =
        !is_span<U>::value && !is_std_array<U>::value &&
        !std::is_array<U>::value && has_size_and_data<C>::value;
};

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename, typename, typename = void>
struct is_container_element_type_compatible : std::false_type {};

template <typename T, typename E>
struct is_container_element_type_compatible<
    T, E, void_t<decltype(detail::data(std::declval<T>()))>>
    : std::is_convertible<
          remove_pointer_t<decltype(detail::data(std::declval<T>()))> (*)[],
          E (*)[]> {};

template <typename, typename = size_t>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};

} // namespace detail

template <typename ElementType, std::ptrdiff_t Extent>
class span {
    static_assert(Extent == dynamic_extent || Extent >= 0,
                  "A span must have an extent greater than or equal to zero, "
                  "or a dynamic extent");
    static_assert(std::is_object<ElementType>::value,
                  "A span's ElementType must be an object type (not a "
                  "reference type or void)");
    static_assert(detail::is_complete<ElementType>::value,
                  "A span's ElementType must be a complete type (not a forward "
                  "declaration)");
    static_assert(!std::is_abstract<ElementType>::value,
                  "A span's ElementType cannot be an abstract class type");

    using storage_type = detail::span_storage<ElementType, Extent>;

public:
    // constants and types
    using element_type = ElementType;
    using value_type = typename std::remove_cv<ElementType>::type;
    using index_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = ElementType*;
    using reference = ElementType&;
    using iterator = pointer;
    using const_iterator = const ElementType*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr index_type extent = static_cast<index_type>(Extent);

    // [span.cons], span constructors, copy, assignment, and destructor
    template <std::ptrdiff_t E = Extent,
              typename std::enable_if<E <= 0, int>::type = 0>
    constexpr span() noexcept
    {}

    TCB_SPAN_CONSTEXPR11 span(pointer ptr, index_type count)
        : storage_(ptr, count)
    {
        TCB_SPAN_EXPECT(extent == dynamic_extent || count == extent);
    }

    TCB_SPAN_CONSTEXPR11 span(pointer first_elem, pointer last_elem)
        : storage_(first_elem, last_elem - first_elem)
    {
        TCB_SPAN_EXPECT(extent == dynamic_extent ||
                        last_elem - first_elem == extent);
    }

    template <
        std::size_t N, std::ptrdiff_t E = Extent,
        typename std::enable_if<
            (E == dynamic_extent || static_cast<std::ptrdiff_t>(N) == E) &&
                detail::is_container_element_type_compatible<
                    element_type (&)[N], ElementType>::value,
            int>::type = 0>
    constexpr span(element_type (&arr)[N]) noexcept : storage_(arr, N)
    {}

    template <
        std::size_t N, std::ptrdiff_t E = Extent,
        typename std::enable_if<
            (E == dynamic_extent || static_cast<std::ptrdiff_t>(N) == E) &&
                detail::is_container_element_type_compatible<
                    std::array<value_type, N>&, ElementType>::value,
            int>::type = 0>
    TCB_SPAN_ARRAY_CONSTEXPR span(std::array<value_type, N>& arr) noexcept
        : storage_(arr.data(), N)
    {}

    template <
        std::size_t N, std::ptrdiff_t E = Extent,
        typename std::enable_if<
            (E == dynamic_extent || static_cast<std::ptrdiff_t>(N) == E) &&
                detail::is_container_element_type_compatible<
                    const std::array<value_type, N>&, ElementType>::value,
            int>::type = 0>
    TCB_SPAN_ARRAY_CONSTEXPR span(const std::array<value_type, N>& arr) noexcept
        : storage_(arr.data(), N)
    {}

    template <typename Container,
              typename std::enable_if<
                  detail::is_container<Container>::value &&
                      detail::is_container_element_type_compatible<
                          Container&, ElementType>::value,
                  int>::type = 0>
    TCB_SPAN_CONSTEXPR11 span(Container& cont)
        : storage_(detail::data(cont), detail::size(cont))
    {
        TCB_SPAN_EXPECT(extent == dynamic_extent ||
                        static_cast<std::ptrdiff_t>(detail::size(cont)) ==
                            extent);
    }

    template <typename Container,
              typename std::enable_if<
                  detail::is_container<Container>::value &&
                      detail::is_container_element_type_compatible<
                          const Container&, ElementType>::value,
                  int>::type = 0>
    TCB_SPAN_CONSTEXPR11 span(const Container& cont)
        : storage_(detail::data(cont), detail::size(cont))
    {
        TCB_SPAN_EXPECT(extent == dynamic_extent ||
                        static_cast<std::ptrdiff_t>(detail::size(cont)) ==
                            extent);
    }

    constexpr span(const span& other) noexcept = default;

    template <typename OtherElementType, std::ptrdiff_t OtherExtent,
              typename std::enable_if<
                  (Extent == OtherExtent || Extent == dynamic_extent) &&
                      std::is_convertible<OtherElementType (*)[],
                                          ElementType (*)[]>::value,
                  int>::type = 0>
    constexpr span(const span<OtherElementType, OtherExtent>& other) noexcept
        : storage_(other.data(), other.size())
    {}

    ~span() noexcept = default;

    span& operator=(const span& other) noexcept = default;

    // [span.sub], span subviews
    template <std::ptrdiff_t Count>
    TCB_SPAN_CONSTEXPR11 span<element_type, Count> first() const
    {
        TCB_SPAN_EXPECT(Count >= 0 && Count <= size());
        return {data(), Count};
    }

    template <std::ptrdiff_t Count>
    TCB_SPAN_CONSTEXPR11 span<element_type, Count> last() const
    {
        TCB_SPAN_EXPECT(Count >= 0 && Count <= size());
        return {data() + (size() - Count), Count};
    }

    template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent>
    using subspan_return_t =
        span<ElementType, Count != dynamic_extent
                              ? Count
                              : (Extent != dynamic_extent ? Extent - Offset
                                                          : dynamic_extent)>;

    template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent>
    TCB_SPAN_CONSTEXPR11 subspan_return_t<Offset, Count> subspan() const
    {
        TCB_SPAN_EXPECT((Offset >= 0 && Offset <= size()) &&
                        (Count == dynamic_extent ||
                         (Count >= 0 && Offset + Count <= size())));
        return {data() + Offset,
                Count != dynamic_extent
                    ? Count
                    : (Extent != dynamic_extent ? Extent - Offset
                                                : size() - Offset)};
    }

    TCB_SPAN_CONSTEXPR11 span<element_type, dynamic_extent>
    first(index_type count) const
    {
        TCB_SPAN_EXPECT(count >= 0 && count <= size());
        return {data(), count};
    }

    TCB_SPAN_CONSTEXPR11 span<element_type, dynamic_extent>
    last(index_type count) const
    {
        TCB_SPAN_EXPECT(count >= 0 && count <= size());
        return {data() + (size() - count), count};
    }

    TCB_SPAN_CONSTEXPR11 span<element_type, dynamic_extent>
    subspan(index_type offset, index_type count = static_cast<index_type>(dynamic_extent)) const
    {
        TCB_SPAN_EXPECT((offset >= 0 && offset <= size()) &&
                        (count == dynamic_extent ||
                         (count >= 0 && offset + count <= size())));
        return {data() + offset,
                count == dynamic_extent ? size() - offset : count};
    }

    // [span.obs], span observers
    constexpr index_type size() const noexcept { return storage_.size; }

    constexpr index_type size_bytes() const noexcept
    {
        return size() * sizeof(element_type);
    }

    constexpr bool empty() const noexcept { return size() == 0; }

    // [span.elem], span element access
    TCB_SPAN_CONSTEXPR11 reference operator[](index_type idx) const
    {
        TCB_SPAN_EXPECT(idx >= 0 && idx < size());
        return *(data() + idx);
    }

    /* Extension: not in P0122 */
#ifndef TCB_SPAN_STD_COMPLIANT_MODE
    TCB_SPAN_CONSTEXPR14 reference at(index_type idx) const
    {
#ifndef TCB_SPAN_NO_EXCEPTIONS
        if (idx < 0 || idx >= size()) {
            char msgbuf[64] = {
                0,
            };
            std::snprintf(msgbuf, sizeof(msgbuf),
                          "Index %td is out of range for span of size %td", idx,
                          size());
            throw std::out_of_range{msgbuf};
        }
#endif // TCB_SPAN_NO_EXCEPTIONS
        return this->operator[](idx);
    }

    TCB_SPAN_CONSTEXPR11 reference front() const
    {
        TCB_SPAN_EXPECT(!empty());
        return *data();
    }

    TCB_SPAN_CONSTEXPR11 reference back() const
    {
        TCB_SPAN_EXPECT(!empty());
        return *(data() + (size() - 1));
    }

#endif // TCB_SPAN_STD_COMPLIANT_MODE

#ifndef TCB_SPAN_NO_FUNCTION_CALL_OPERATOR
    TCB_SPAN_DEPRECATED_FOR("Use operator[] instead")
    constexpr reference operator()(index_type idx) const
    {
        return this->operator[](idx);
    }
#endif // TCB_SPAN_NO_FUNCTION_CALL_OPERATOR

    constexpr pointer data() const noexcept { return storage_.ptr; }

    // [span.iterators], span iterator support
    constexpr iterator begin() const noexcept { return data(); }

    constexpr iterator end() const noexcept { return data() + size(); }

    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr const_iterator cend() const noexcept { return end(); }

    TCB_SPAN_ARRAY_CONSTEXPR reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator(end());
    }

    TCB_SPAN_ARRAY_CONSTEXPR reverse_iterator rend() const noexcept
    {
        return reverse_iterator(begin());
    }

    TCB_SPAN_ARRAY_CONSTEXPR const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    TCB_SPAN_ARRAY_CONSTEXPR const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

private:
    storage_type storage_{};
};

#ifdef TCB_SPAN_HAVE_DEDUCTION_GUIDES

/* Deduction Guides */
template <class T, size_t N>
span(T (&)[N])->span<T, N>;

template <class T, size_t N>
span(std::array<T, N>&)->span<T, N>;

template <class T, size_t N>
span(const std::array<T, N>&)->span<const T, N>;

template <class Container>
span(Container&)->span<typename Container::value_type>;

template <class Container>
span(const Container&)->span<const typename Container::value_type>;

#endif // TCB_HAVE_DEDUCTION_GUIDES

template <typename ElementType, std::ptrdiff_t Extent>
constexpr span<ElementType, Extent>
make_span(span<ElementType, Extent> s) noexcept
{
    return s;
}

#define AS_SIGNED(N) static_cast<std::ptrdiff_t>(N)

template <typename T, std::size_t N>
constexpr span<T, AS_SIGNED(N)> make_span(T (&arr)[N]) noexcept
{
    return {arr};
}

template <typename T, std::size_t N>
TCB_SPAN_ARRAY_CONSTEXPR span<T, AS_SIGNED(N)> make_span(std::array<T, N>& arr) noexcept
{
    return {arr};
}

template <typename T, std::size_t N>
TCB_SPAN_ARRAY_CONSTEXPR span<const T, AS_SIGNED(N)>
make_span(const std::array<T, N>& arr) noexcept
{
    return {arr};
}

#undef AS_SIGNED

template <typename Container>
constexpr span<typename Container::value_type> make_span(Container& cont)
{
    return {cont};
}

template <typename Container>
constexpr span<const typename Container::value_type>
make_span(const Container& cont)
{
    return {cont};
}

/* Comparison operators */
// Implementation note: the implementations of == and < are equivalent to
// 4-legged std::equal and std::lexicographical_compare respectively

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator==(span<T, X> lhs, span<U, Y> rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::ptrdiff_t i = 0; i < lhs.size(); i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator!=(span<T, X> lhs, span<U, Y> rhs)
{
    return !(lhs == rhs);
}

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator<(span<T, X> lhs, span<U, Y> rhs)
{
    // No std::min to avoid dragging in <algorithm>
    const std::ptrdiff_t size =
        lhs.size() < rhs.size() ? lhs.size() : rhs.size();

    for (std::ptrdiff_t i = 0; i < size; i++) {
        if (lhs[i] < rhs[i]) {
            return true;
        }
        if (lhs[i] > rhs[i]) {
            return false;
        }
    }
    return lhs.size() < rhs.size();
}

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator<=(span<T, X> lhs, span<U, Y> rhs)
{
    return !(rhs < lhs);
}

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator>(span<T, X> lhs, span<U, Y> rhs)
{
    return rhs < lhs;
}

template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
TCB_SPAN_CONSTEXPR14 bool operator>=(span<T, X> lhs, span<U, Y> rhs)
{
    return !(lhs < rhs);
}

template <typename ElementType, std::ptrdiff_t Extent>
span<const byte, ((Extent == dynamic_extent)
                      ? dynamic_extent
                      : (static_cast<ptrdiff_t>(sizeof(ElementType)) * Extent))>
as_bytes(span<ElementType, Extent> s) noexcept
{
    return {reinterpret_cast<const byte*>(s.data()), s.size_bytes()};
}

template <
    class ElementType, ptrdiff_t Extent,
    typename std::enable_if<!std::is_const<ElementType>::value, int>::type = 0>
span<byte, ((Extent == dynamic_extent)
                ? dynamic_extent
                : (static_cast<ptrdiff_t>(sizeof(ElementType)) * Extent))>
as_writable_bytes(span<ElementType, Extent> s) noexcept
{
    return {reinterpret_cast<byte*>(s.data()), s.size_bytes()};
}

/* Extension: nonmember subview operations */

#ifndef TCB_SPAN_STD_COMPLIANT_MODE

template <std::ptrdiff_t Count, typename T>
TCB_SPAN_CONSTEXPR11 auto first(T& t)
    -> decltype(make_span(t).template first<Count>())
{
    return make_span(t).template first<Count>();
}

template <std::ptrdiff_t Count, typename T>
TCB_SPAN_CONSTEXPR11 auto last(T& t)
    -> decltype(make_span(t).template last<Count>())
{
    return make_span(t).template last<Count>();
}

template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent,
          typename T>
TCB_SPAN_CONSTEXPR11 auto subspan(T& t)
    -> decltype(make_span(t).template subspan<Offset, Count>())
{
    return make_span(t).template subspan<Offset, Count>();
}

template <typename T>
TCB_SPAN_CONSTEXPR11 auto first(T& t, std::ptrdiff_t count)
    -> decltype(make_span(t).first(count))
{
    return make_span(t).first(count);
}

template <typename T>
TCB_SPAN_CONSTEXPR11 auto last(T& t, std::ptrdiff_t count)
    -> decltype(make_span(t).last(count))
{
    return make_span(t).last(count);
}

template <typename T>
TCB_SPAN_CONSTEXPR11 auto subspan(T& t, std::ptrdiff_t offset,
                                  std::ptrdiff_t count = dynamic_extent)
    -> decltype(make_span(t).subspan(offset, count))
{
    return make_span(t).subspan(offset, count);
}

#endif // TCB_SPAN_STD_COMPLIANT_MODE

} // namespace TCB_SPAN_NAMESPACE_NAME

/* Extension: support for C++17 structured bindings */

#ifndef TCB_SPAN_STD_COMPLIANT_MODE

namespace TCB_SPAN_NAMESPACE_NAME {

template <std::ptrdiff_t N, typename E, std::ptrdiff_t S>
constexpr auto get(span<E, S> s) -> decltype(s[N])
{
    return s[N];
}

} // namespace TCB_SPAN_NAMESPACE_NAME

namespace std {

template <typename E, ptrdiff_t S>
class tuple_size<tcb::span<E, S>> : public integral_constant<size_t, static_cast<size_t>(S)> {};

template <typename E>
class tuple_size<tcb::span<E, tcb::dynamic_extent>>; // not defined

template <size_t N, typename E, ptrdiff_t S>
class tuple_element<N, tcb::span<E, S>> {
public:
    using type = E;
};

} // end namespace std

#endif // TCB_SPAN_STD_COMPLIANT_MODE

#endif // TCB_SPAN_HPP_INCLUDED
