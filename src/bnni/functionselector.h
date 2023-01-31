#ifndef FUNCTIONSELECTOR_H
#define FUNCTIONSELECTOR_H

template<typename ... Args> struct SELECT {
        template<typename C, typename R>
        static constexpr auto OVERLOAD_OF(R (C::*pmf)(Args...)) -> decltype(pmf) {
                return pmf;
        }
};

#endif // FUNCTIONSELECTOR_H
