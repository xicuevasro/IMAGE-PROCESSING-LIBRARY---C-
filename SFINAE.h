#ifndef SFINAE_H
#define SFINAE_H

template<typename Cond>
using EnableIfPolicy = typename std::enable_if_t<Cond::value>;

template<typename Cond1, typename Cond2>
using EnableIfOR2Policies = typename std::enable_if_t<Cond1::value || Cond2::value>;

template<typename Cond1, typename Cond2>
using EnableIfAND2Policies = typename std::enable_if_t<Cond1::value && Cond2::value>;

template<typename T>
using IsFloatingPoint = std::is_floating_point<T>;

template<typename T>
using IsIntegral = std::is_integral<T>;

// Test if type U can be casted to type T
/*
template <typename T, typename U, typename = void>
class IsSafelyCastable : std::false_type {};
template <typename T, typename U>
class IsSafelyCastable<T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};
*/
template <typename T, typename U>
class IsSafelyCastable
{
    typedef char one;
    typedef long two;

    template <typename V, typename W> static one test( decltype(static_cast<W>(std::declval<V>())) ) ;
    template <typename V, typename W> static two test(...);    

public:
    static bool const value = sizeof(test<T,U>(0)) == sizeof(char);
};

// Test if types are identical
template <typename T, typename U>
using IsSame = std::is_same<T, U>;

// Test if type T has subscript operator []
// https://stackoverflow.com/questions/31305894/how-to-check-for-the-existence-of-a-subscript-operator
// https://microeducate.tech/how-to-check-for-the-existence-of-a-subscript-operator/
/*
template <typename T>
class HasBrackets
{
    typedef char Yes[1];
    typedef char No[2];
    template <typename C> static Yes& test( decltype(&C::operator[]) ) ;
    template <typename C> static No& test(...);    
public:
    static bool const value = sizeof(test<T>(0)) == sizeof(Yes&);
};
*/
template <typename... >
using void_t = void;
template<class T, class Index, typename = void>
struct has_subscript_operator : std::false_type { };
template<class T, class Index>
struct has_subscript_operator<T, Index, void_t<
    decltype(std::declval<T>()[std::declval<Index>()])>> : std::true_type { };
template <class T, class Index>
using HasSubscriptOperator = typename has_subscript_operator<T, Index>::type;

// Test if type T has a member function size()
template <typename T>
struct HasSize 
{
    typedef char Yes[1];
    typedef char No[2];
    template <typename C> static Yes& test(typename std::enable_if<std::is_member_function_pointer<decltype(&C::size)>::value,bool>::type=0);
    template <typename C> static No& test(...);
public:
    static bool const value = sizeof(test<typename std::remove_cv<T>::type>(0)) == sizeof(Yes&);
};

// Test if type is integral
template <typename T>
using IntegralPolicy = EnableIfPolicy<IsIntegral<T>>;

// Test if type is floating point
template <typename T>
using FloatingPointPolicy = EnableIfPolicy<IsFloatingPoint<T>>;

// Test if two types are floating point
template <typename T, typename U>
using FloatingPoint2Policy = EnableIfAND2Policies<IsFloatingPoint<T>, IsFloatingPoint<U>>;

// Test if two types are the same or castable from one to the toher
template <typename T, typename U>
using SameORCastablePolicy = EnableIfOR2Policies<IsSame<T, U>, IsSafelyCastable<T, U>>;

// Test if type has operator [] and member function size()
template <typename T>
using BracketsANDSizePolicy = EnableIfAND2Policies<HasSubscriptOperator<T, int>, HasSize<T>>;

#endif