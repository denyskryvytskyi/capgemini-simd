/**
 * TASK: Basic SIMD Operations
 * RESULTS: (for static array size equals 100'000)
 *  1. with optimizations flags (-mavx2)):
 *      - Loop-based execution time: ~200-300 ms
 *      - SIMD-based: ~50-60 microseconds (overall faster)
 *  2. with optimization flags (-O3 -mavx2):
 *      - Loop-based execution time: ~100-200 microsecond
 *      - SIMD-based: ~30-40 microsecond (overall faster)
 **/

#include <immintrin.h>  // AVX, SSE, MMX
#include <cpuid.h>
#include <iostream>
#include <chrono>

constexpr int32_t ARRAY_SIZE = 100'000;
constexpr int32_t SIMD_AVX_WIDTH = 8;
constexpr int32_t SIMD_SSE_WIDTH = 4;
constexpr int32_t SIMD_MMX_WIDTH = 2;
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_ARRAYS = false;

bool isSupportedSSE2();
bool isSupportedAVX();
bool checkSIMDSupport();

void printArray(int32_t* arr);

void addLooped(int32_t* arrA, int32_t* arrB);
void addSIMD(int32_t* arrA, int32_t* arrB);

int main()
{
    const bool isSIMDSupport = checkSIMDSupport();

    alignas(ALIGNMENT) int32_t arrA[ARRAY_SIZE];
    alignas(ALIGNMENT) int32_t arrB[ARRAY_SIZE];

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
    if (isSIMDSupport) {
        addSIMD(arrA, arrB);
    }

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

bool checkSIMDSupport()
{
    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
        return false;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;

        return false;
    }

    return true;
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

    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " microseconds.\n";
}

void addSIMD(int32_t* pArrA, int32_t* pArrB)
{
std::cout << "===== SIMD-based addition =====\n";

    const __m256i* pArrASIMD = reinterpret_cast<const __m256i*>(pArrA);
    const __m256i* pArrBSIMD = reinterpret_cast<const __m256i*>(pArrB);

    alignas(ALIGNMENT) int result[ARRAY_SIZE]; // result array
    __m256i* pResSIMD = reinterpret_cast<__m256i*>(result); // pointer to result array with proper type for simd

    const int32_t resultArraySize = ARRAY_SIZE / SIMD_AVX_WIDTH;
    int i = 0;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (; i < resultArraySize; ++i) {
        pResSIMD[i] = _mm256_add_epi32(pArrASIMD[i], pArrBSIMD[i]);
    }

    // remainder calculation
    i *= SIMD_AVX_WIDTH; // convert index to normal mode
    // try using sse
    if (ARRAY_SIZE - i >= SIMD_SSE_WIDTH) {
        i /= SIMD_SSE_WIDTH; // convert index to sse mode
        const int32_t resultArraySize_SSE = ARRAY_SIZE / SIMD_SSE_WIDTH;

        const __m128i* pArrASIMD_SSE = reinterpret_cast<const __m128i*>(pArrA);
        const __m128i* pArrBSIMD_SSE = reinterpret_cast<const __m128i*>(pArrB);
        __m128i* pResSIMD_SSE = reinterpret_cast<__m128i*>(result);

        for (; i < resultArraySize_SSE; ++i) {
            pResSIMD_SSE[i] = _mm_add_epi32(pArrASIMD_SSE[i], pArrBSIMD_SSE[i]);
        }

        i *= SIMD_SSE_WIDTH; // convert back to normal index
    }

    // try using mmx
    if (ARRAY_SIZE - i >= SIMD_MMX_WIDTH) {
        i /= SIMD_MMX_WIDTH; // convert index to mmx mode
        const int32_t resultArraySize_MMX = ARRAY_SIZE / SIMD_MMX_WIDTH;

        const __m64* pArrASIMD_MMX = reinterpret_cast<const __m64*>(pArrA);
        const __m64* pArrBSIMD_MMX = reinterpret_cast<const __m64*>(pArrB);
        __m64* pResSIMD_MMX = reinterpret_cast<__m64*>(result);

        for (; i < resultArraySize_MMX; ++i) {
            pResSIMD_MMX[i] = _mm_add_pi16(pArrASIMD_MMX[i], pArrBSIMD_MMX[i]);
        }

        i *= SIMD_MMX_WIDTH; // convert back to normal index
    }

    // remaining elements addition
    for (; i < ARRAY_SIZE; ++i) {
        result[i] = pArrA[i] + pArrB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(result);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " microseconds.\n";
}
