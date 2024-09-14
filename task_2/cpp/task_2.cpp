/**
 * TASK: Data Alignment and Memory Access
 * NOTE: my laptop doesn't support AVX512, so I have used 256-bit registers
 * RESULTS: (for dynamic array with size equals 100'000'000)
 *  - With optimization flag -mavx2 only:
 *      - Loop-based execution time: ~200-300 ms
 *      - SIMD-based execution time: ~70-80 ms (overall faster)
 *  - With optimization flags -O3 -mavx2:
 *      - Loop-based execution time: 80-300 ms
 *      - SIMD-based execution time: ~70 ms (overall faster)
 * TODO:
 *  - use smaller SIMD registers for remainder calculation
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <chrono>
#include <cstring>

constexpr int32_t ARRAY_SIZE = 100'000'000;
constexpr int32_t INT_AMOUNT_PER_SIMD_REG = 8;
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_ARRAYS = false;

void printArray(int* arr);

void allocArray(int*& pArr);
void initData(int* pArrA, int* pArrB);
void addLooped(int* pArrA, int* pArrB, int* pRes);
void addSIMD(int* pArrA, int* pArrB, int* pRes);

int main()
{
    int* pArrA = nullptr;
    int* pArrB = nullptr;
    int* pRes = nullptr;

    allocArray(pArrA);
    allocArray(pArrB);
    allocArray(pRes);

    initData(pArrA, pArrB);

    if (PRINT_ARRAYS) {
        std::cout << "Array A: ";
        printArray(pArrA);

        std::cout << "Array B: ";
        printArray(pArrB);
    }

    addLooped(pArrA, pArrB, pRes);
    addSIMD(pArrA, pArrB, pRes);

    free(pArrA);
    free(pArrB);

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

bool isSupportedAVX512()
{
    unsigned int cpuInfo[4] = { 0 }; // eax, ebx, ecx, edx registers
    __get_cpuid(0, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]); // check if CPUID supports function 7
    
    if (cpuInfo[0] >= 7) // ensure function 7 is supported
    {
        __get_cpuid(7, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
        return (cpuInfo[1] & (1 << 16)) != 0; // Check AVX-512 Foundation (bit 16 of EBX)
    }
    
    return false; // AVX-512 is not supported
}

void printArray(int* arr)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}


void allocArray(int*& pArr)
{
    if (posix_memalign(reinterpret_cast<void**>(&pArr), ALIGNMENT, ARRAY_SIZE * sizeof(int)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
    }
}

void initData(int* pArrA, int* pArrB)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        pArrA[i] = i;
        pArrB[i] = i + 1;
    }
}

void addLooped(int* pArrA, int* pArrB, int* pRes)
{
    std::cout << "===== Looped-based addition =====\n";

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
    std::cout << "Calculation time: " << duration.count() << " ms.\n";
}

void addSIMD(int* pArrA, int* pArrB, int* pRes)
{
    std::cout << "===== SIMD-based addition =====\n";

    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;
    }
    if (!isSupportedAVX512()) {
        // std::cerr << "AVX512 is not supported on this CPU." << std::endl;
    }

    const __m256i* pArrASIMD = reinterpret_cast<const __m256i*>(pArrA);
    const __m256i* pArrBSIMD = reinterpret_cast<const __m256i*>(pArrB);
    __m256i* pResSIMD = reinterpret_cast<__m256i*>(pRes);

    const int32_t resultArraySize = ARRAY_SIZE / INT_AMOUNT_PER_SIMD_REG;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (; i < resultArraySize; ++i) {
        pResSIMD[i] = _mm256_add_epi32(pArrASIMD[i], pArrBSIMD[i]);
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // remainder calculation
    for (int j = i * INT_AMOUNT_PER_SIMD_REG; j < ARRAY_SIZE; ++j) {
        pRes[j] = pArrA[j] + pArrB[j];
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Calculation time: " << duration.count() << " ms.\n";

    if (PRINT_ARRAYS) {
        std::cout << "Result of A + B: ";
        printArray(pRes);
    }
}