/**
 * TASK: Basic SIMD Operations
 * RESULTS: (for static array size equals 10003)
 *  1. without optimizations flags:
 *      - Loop-based execution time: ~30-40 microseconds
 *      - SIMD-based: ~7-10 microseconds
 *  2. with optimization flags (-O3 -msse2):
 *      - Loop-based execution time: <1 microsecond (faster)
 *      - SIMD-based: ~2 microsecond
 **/

#include <emmintrin.h>  // SSE2
#include <cpuid.h>
#include <iostream>
#include <chrono>

constexpr int32_t ARRAY_SIZE = 10003;
constexpr bool PRINT_ARRAYS = false;
constexpr int32_t INT_AMOUNT_PER_SIMD_REG = 4;

void printArray(int32_t* arr);
void addLooped(int32_t* arrA, int32_t* arrB);
void addSIMD(int32_t* arrA, int32_t* arrB);

int main()
{
    alignas(16) int32_t arrA[ARRAY_SIZE];
    alignas(16) int32_t arrB[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arrA[i] = i;
        arrB[i] = i + 1;
    }

    if (PRINT_ARRAYS) {
        std::cout << "Array A: ";
        printArray(arrA);

        std::cout << "Array B: ";
        printArray(arrB);
    }

    addLooped(arrA, arrB);
    addSIMD(arrA, arrB);

    return 0;
}

bool isSupportedSSE2()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(1, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
    return (cpuInfo[3] & (1 << 26)) != 0; // Check SSE2 bit
}

bool isSupportedAVX()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(1, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
    return (cpuInfo[2] & (1 << 28)) != 0; // Check AVX bit
}

void printArray(int32_t* arr)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

void addLooped(int32_t* arrA, int32_t* arrB)
{
    std::cout << "===== Looped-based addition =====\n";

    int R[ARRAY_SIZE];
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        R[i] = arrA[i] + arrB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(R);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " microseconds.\n";
}

void addSIMD(int32_t* pArrA, int32_t* pArrB)
{
    std::cout << "===== SIMD-based addition =====\n";

    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;
    }

    const __m128i* pArrASIMD = reinterpret_cast<const __m128i*>(pArrA);
    const __m128i* pArrBSIMD = reinterpret_cast<const __m128i*>(pArrB);

    int result[ARRAY_SIZE]; // result array
    __m128i* pResSIMD = reinterpret_cast<__m128i*>(result); // pointer to result array with proper type for simd

    const int32_t resultArraySize = ARRAY_SIZE / INT_AMOUNT_PER_SIMD_REG;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (; i < resultArraySize; ++i) {
        pResSIMD[i] = _mm_add_epi32(pArrASIMD[i], pArrBSIMD[i]);
    }

    // remainder
    for (int j = i * INT_AMOUNT_PER_SIMD_REG; j < ARRAY_SIZE; ++j) {
        result[j] = pArrA[j] + pArrB[j];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(result);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " microseconds.\n";
}
