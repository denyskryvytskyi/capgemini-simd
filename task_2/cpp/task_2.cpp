/**
 * TASK: Data Alignment and Memory Access
 * NOTE: my laptop doesn't support AVX512, so I have used 256-bit registers
 * RESULTS: (for dynamic array with size equals 100'000'000)
 *  - With optimization flag -mavx2 only:
 *      - Loop-based execution time: ~200-300 ms
 *      - SIMD-based execution time: ~70-80 ms (overall faster)
 *  - With optimization flags -O3 -mavx2:
 *      - Loop-based execution time: 100-300 ms
 *      - SIMD-based execution time: ~70 ms (overall faster)
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <chrono>
#include <cstring>

constexpr int32_t ARRAY_SIZE = 100'000'000;
constexpr int32_t SIMD_AVX_WIDTH = 8;
constexpr int32_t SIMD_SSE_WIDTH = 4;
constexpr int32_t SIMD_MMX_WIDTH = 2;
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_ARRAYS = false;

bool isSupportedSSE2();
bool isSupportedAVX();
bool isSupportedAVX512();
bool checkSIMDSupport();

int32_t* allocArray();
void initData(int32_t* pArrA, int32_t* pArrB);
void printArray(int* arr);

void add(int32_t* pArrA, int32_t* pArrB, int32_t* pRes);
void addSIMD(int32_t* pArrA, int32_t* pArrB, int32_t* pRes);

int main()
{
    const bool isSIMDSupport = checkSIMDSupport();

    int32_t* pArrA = allocArray();
    if (!pArrA) {
        std::cerr << "Failed to allocate memory for array A." << std::endl;
        return 1;
    }

    int32_t* pArrB = allocArray();
    if (!pArrB) {
        free(pArrA);
        std::cerr << "Failed to allocate memory for array B." << std::endl;
        return 1;
    }

    int32_t* pRes = allocArray();
    if (!pRes) {
        free(pArrA);
        free(pArrB);
        std::cerr << "Failed to allocate memory for result array." << std::endl;
        return 1;
    }

    initData(pArrA, pArrB);

    if (PRINT_ARRAYS) {
        std::cout << "Array A: ";
        printArray(pArrA);

        std::cout << "Array B: ";
        printArray(pArrB);
    }

    add(pArrA, pArrB, pRes);
    if (isSIMDSupport) {
        addSIMD(pArrA, pArrB, pRes);
    }

    free(pArrA);
    free(pArrB);
    free(pRes);

    return 0;
}

bool isSupportedSSE2()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(1, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
    return (cpuInfo[3] & (1 << 26)) != 0; // check SSE2 bit
}

bool isSupportedAVX()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(1, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
    return (cpuInfo[2] & (1 << 28)) != 0; // check AVX bit
}

bool isSupportedAVX512()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(0, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]); // check if CPUID supports function 7
    
    if (cpuInfo[0] >= 7) {
        __get_cpuid(7, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
        return (cpuInfo[1] & (1 << 16)) != 0; // check AVX-512 bit
    }
    
    return false; // AVX-512 is not supported
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
    if (!isSupportedAVX512()) {
        std::cerr << "AVX512 is not supported on this CPU." << std::endl;
    }
    
    return true;
}

int32_t* allocArray()
{
    int32_t* ptr = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&ptr), ALIGNMENT, ARRAY_SIZE * sizeof(int32_t)) != 0) {
        return nullptr;
    }

    return ptr;
}

void initData(int32_t* pArrA, int32_t* pArrB)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        pArrA[i] = i;
        pArrB[i] = i + 1;
    }
}

void printArray(int32_t* arr)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}

void add(int32_t* pArrA, int32_t* pArrB, int32_t* pRes)
{
    std::cout << "===== Loop-based addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        pRes[i] = pArrA[i] + pArrB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(pRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void addSIMD(int32_t* pArrA, int32_t* pArrB, int32_t* pRes)
{
    std::cout << "===== SIMD-based addition =====\n";

    const __m256i* pArrASIMD = reinterpret_cast<const __m256i*>(pArrA);
    const __m256i* pArrBSIMD = reinterpret_cast<const __m256i*>(pArrB);
    __m256i* pResSIMD = reinterpret_cast<__m256i*>(pRes);

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
        __m128i* pResSIMD_SSE = reinterpret_cast<__m128i*>(pRes);

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
        __m64* pResSIMD_MMX = reinterpret_cast<__m64*>(pRes);

        for (; i < resultArraySize_MMX; ++i) {
            pResSIMD_MMX[i] = _mm_add_pi16(pArrASIMD_MMX[i], pArrBSIMD_MMX[i]);
        }

        i *= SIMD_MMX_WIDTH; // convert back to normal index
    }

    // remaining elements addition
    for (; i < ARRAY_SIZE; ++i) {
        pRes[i] = pArrA[i] + pArrB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(pRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}