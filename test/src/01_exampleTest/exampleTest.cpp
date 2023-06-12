#include "Shuffle.h"

// TODO: use testing framework such as Boost test, google test or catch2

template<bool isCorrect>
consteval int returnsFour() {
    if constexpr (isCorrect) {
        return 4;
    } else {
        return 3;
    }
}

int main() {
    if (returnsFour<true>() == 4) {
        return 0;
    } else {
        return -1;
    }

}
