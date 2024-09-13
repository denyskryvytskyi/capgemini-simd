/**
 * TASK: Data Alignment and Memory Access
 * TODO:
 *  - increase array size and process by step of 4
 *  - performance measure
 **/ 

#include <emmintrin.h>  // SSE2
#include <cpuid.h>

#include <iostream>
#include <cstdlib>

bool isSupportedSSE2();
bool isSupportedAVX();
void printArray(int*& arr);
void printResult(int* arr);

void allocArray(int*& pArr);
void initData(int*& pArrA, int*& pArrB);
void addLooped(int*& pArrA, int*& pArrB);
void addSIMD(int*& pArrA, int*& pArrB);

constexpr int ARRAY_SIZE = 4;
constexpr int ALIGNMENT = 16;
constexpr int OFFSET = 3; // offset used to init the second array

int main()
{
    int* pArrA = nullptr;
    int* pArrB = nullptr;

    allocArray(pArrA);
    allocArray(pArrB);

    initData(pArrA, pArrB);

    std::cout << "Array A: ";
    printArray(pArrA);

    std::cout << "Array B: ";
    printArray(pArrB);

    addLooped(pArrA, pArrB);
    addSIMD(pArrA, pArrB);

    free(pArrA);
    free(pArrB);

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

void printArray(int*& arr)
{
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

void printResult(int* res)
{
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << res[i] << " ";
    }

    std::cout << std::endl;
}

void allocArray(int*& pArr)
{
    if (posix_memalign(reinterpret_cast<void**>(&pArr), ALIGNMENT, ARRAY_SIZE * sizeof(int)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
    }
}

void initData(int*& pArrA, int*& pArrB)
{
    for (size_t i = 0; i < ARRAY_SIZE; ++i)
    {
        pArrA[i] = i;
        pArrB[i] = i + OFFSET;
    }
}

void addLooped(int*& pArrA, int*& pArrB)
{
    std::cout << "===== Looped-based addition =====\n";

    int R[4];

    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        R[i] = pArrA[i] + pArrB[i];
    }

    std::cout << "Result of A + B: ";
    printResult(R);
}

void addSIMD(int*& pArrA, int*& pArrB)
{
    std::cout << "===== SIMD-based addition =====\n";

    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;
    }

    __m128i xmmA = _mm_load_si128(reinterpret_cast<const __m128i*>(pArrA));
    __m128i xmmB = _mm_load_si128(reinterpret_cast<const __m128i*>(pArrB));

    __m128i xmmRes = _mm_add_epi32(xmmA, xmmB);

    alignas(16) int R[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(R), xmmRes); // Store result of xmm3 into R1

    std::cout << "Result of A + B: ";
    printResult(R);
}