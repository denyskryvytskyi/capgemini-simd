#include <emmintrin.h>  // SSE2
#include <cpuid.h>
#include <iostream>

bool isSupportedSSE2();
bool isSupportedAVX();
void printArray(int* arr);
void addLooped(int* arrA, int* arrB);
void addSIMD(int* arrA, int* arrB);

constexpr int ARRAY_SIZE = 4;

int main()
{
    alignas(16) int32_t arrA[ARRAY_SIZE] = { 25, 41, -6, 80 };
    alignas(16) int32_t arrB[ARRAY_SIZE] = { 11, 2, 3, -50 };

    std::cout << "Array A: ";
    printArray(arrA);

    std::cout << "Array B: ";
    printArray(arrB);

    addLooped(arrA, arrB);
    addSIMD(arrA, arrB);

    return 0;
}

bool isSupportedSSE2()
{
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (edx & (1 << 26)) != 0; // Check SSE2 bit
}

bool isSupportedAVX()
{
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 28)) != 0; // Check AVX bit
}

void printArray(int* arr)
{
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

void addLooped(int* arrA, int* arrB)
{
    std::cout << "===== Looped-based addition =====\n";

    int R[4];

    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        R[i] = arrA[i] + arrB[i];
    }

    std::cout << "Result of A + B: ";
    printArray(R);
}

void addSIMD(int* arrA, int* arrB)
{
    std::cout << "===== SIMD-based addition =====\n";

    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;
    }

    __m128i xmmA = _mm_load_si128(reinterpret_cast<const __m128i*>(arrA));
    __m128i xmmB = _mm_load_si128(reinterpret_cast<const __m128i*>(arrB));

    __m128i xmmRes = _mm_add_epi32(xmmA, xmmB);

    alignas(16) int R[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(R), xmmRes); // Store result of xmm3 into R1

    std::cout << "Result of A + B: ";
    printArray(R);
}