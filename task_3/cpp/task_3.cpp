/**
 * TASK: Vector Addition and Dot Product Calculation Using SIMD
 * RESULTS: (for vector with size equals 100'000'000) 
 *      ADDITION:
 *      - With optimization flag -mavx2 only:
 *          - Loop-based execution time: ~300-400 ms
 *          - SIMD-based execution time: ~75 ms (overall faster)
 *      - With optimization flags -O3 -mavx2 -mfma:
 *          - Loop-based execution time: ~100-180 ms
 *          - SIMD-based execution time: ~60-80 ms (overall faster)
 *      DOT PRODUCT:
 *      - With optimization flag -mavx2 only:
 *          - Loop-based execution time: ~200-300 ms
 *          - SIMD-based execution time: ~70-80 ms (overall faster)
 *      - With optimization flags -O3 -mavx2 -mfma:
 *          - Loop-based execution time: ~130 ms
 *          - SIMD-based execution time: ~40 ms (overall faster)
 * * TODO:
 *  - try loop-unrolling for simd loop
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <chrono>       // timestamp

constexpr int32_t VEC_SIZE = 100'000'000;
constexpr int32_t SIMD_AVX_WIDTH = 8;
constexpr int32_t SIMD_SSE_WIDTH = 4;
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_VEC = false;
constexpr float VEC_A_OFFSET = 0.2f;
constexpr float VEC_B_OFFSET = 1.3f;

bool isSupportedSSE2();
bool isSupportedAVX();
bool isSupportedAVX512();
bool checkSIMDSupport();

void allocVec(float*& pVec);
void initData(float* pVecA, float* pVecB);
void printVec(float* pVec);

float sumM128(__m128 value); // helper function to find sum of the 4 float numbers in sse register
void add(float* pVecA, float* pVecB, float* pRes);
void addSIMD(float* pVecA, float* pVecB, float* pRes);
void dotProduct(float* pVecA, float* pVecB);
void dotProductSIMD(float* pVecA, float* pVecB);

int main()
{
    const bool isSIMDSupport = checkSIMDSupport();

    float* pVecA = nullptr;
    float* pVecB = nullptr;
    float* pVecRes = nullptr;

    allocVec(pVecA);
    allocVec(pVecB);
    allocVec(pVecRes);

    initData(pVecA, pVecB);

    if (PRINT_VEC) {
        std::cout << "Vector A: ";
        printVec(pVecA);

        std::cout << "Vector B: ";
        printVec(pVecB);
    }

    add(pVecA, pVecB, pVecRes);
    if (isSIMDSupport) {
        addSIMD(pVecA, pVecB, pVecRes);
    }

    dotProduct(pVecA, pVecB);
    if (isSIMDSupport) {
        dotProductSIMD(pVecA, pVecB);
    }

    free(pVecA);
    free(pVecB);
    free(pVecRes);

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

    // ensure function 7 is supported
    if (cpuInfo[0] >= 7) {
        __get_cpuid(7, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
        return (cpuInfo[1] & (1 << 16)) != 0; // Check AVX-512 Foundation (bit 16 of EBX)
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


void allocVec(float*& pVec)
{
    if (posix_memalign(reinterpret_cast<void**>(&pVec), ALIGNMENT, VEC_SIZE * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
    }
}

void initData(float* pVecA, float* pVecB)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecA[i] = static_cast<float>(i) + VEC_A_OFFSET;
        pVecB[i] = static_cast<float>(i) + VEC_B_OFFSET;
    }
}

void printVec(float* pVec)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

float sumM128(__m128 value)
{
    value = _mm_add_ps(value, _mm_movehl_ps(value, value)); // 2 higher values ([2][3]) + 2 lower values ([0][1])
    value = _mm_add_ss(value, _mm_shuffle_ps(value, value, 1)); // [0] + [1] values
    return _mm_cvtss_f32(value);
}

void add(float* pVecA, float* pVecB, float* pVecRes)
{
    std::cout << "===== Loop-based addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_VEC) {
        std::cout << "Result of A + B: ";
        printVec(pVecRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void addSIMD(float* pVecA, float* pVecB, float* pVecRes)
{
    std::cout << "===== SIMD-based addition =====\n";

    const __m256* pVecASIMD = reinterpret_cast<const __m256*>(pVecA);
    const __m256* pVecBSIMD = reinterpret_cast<const __m256*>(pVecB);
    __m256* pVecResSIMD = reinterpret_cast<__m256*>(pVecRes);

    const int32_t vecAvxSize = VEC_SIZE / SIMD_AVX_WIDTH;

    int i = 0;
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (; i < vecAvxSize; ++i) {
        pVecResSIMD[i] = _mm256_add_ps(pVecASIMD[i], pVecBSIMD[i]);
    }

    // remainder calculation
    i *= SIMD_AVX_WIDTH; // convert index to normal mode
    // try to use sse
    if (VEC_SIZE - i >= SIMD_SSE_WIDTH) {
        i /= SIMD_SSE_WIDTH; // convert index to sse mode
        const __m128* pVecASIMDSSE = reinterpret_cast<const __m128*>(pVecA);
        const __m128* pVecBSIMDSSE = reinterpret_cast<const __m128*>(pVecB);
        __m128* pVecResSIMDSSE = reinterpret_cast<__m128*>(pVecRes);
        const int vecSseSize = VEC_SIZE / SIMD_SSE_WIDTH; // find size of vector in sse mode
        for (; i < vecSseSize; ++i) {
            pVecResSIMDSSE[i] = _mm_add_ps(pVecASIMDSSE[i], pVecBSIMDSSE[i]);
        }

        i *= SIMD_SSE_WIDTH; // convert index back to normal mode
    }

    // final remainder
    for (; i < VEC_SIZE; ++i) {
        pVecRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_VEC) {
        std::cout << "Result of A + B: ";
        printVec(pVecRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void dotProduct(float* pVecA, float* pVecB)
{
    std::cout << "===== Loop-based dot product =====\n";
    double result = 0.0f;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        result += pVecA[i] * pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Dot product: " <<  result << std::endl;
}

void dotProductSIMD(float* pVecA, float* pVecB)
{
    std::cout << "===== SIMD-based dot product =====\n";
    // we need to multiply 8 elements at time and then add to the 8 element result array
    // then we need to add all elements in result 8-element array

    const __m256* pVecASIMD = reinterpret_cast<const __m256*>(pVecA);
    const __m256* pVecBSIMD = reinterpret_cast<const __m256*>(pVecB);
    const int32_t resultArraySize = VEC_SIZE / SIMD_AVX_WIDTH;

    // accumulator that hold sum up of every 8 float elements multiplication
    __m256 pResult = _mm256_setzero_ps();

    int i = 0; // index in avx mode
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
        const auto tstartTimePoint = std::chrono::high_resolution_clock::now();
    for (; i < resultArraySize; ++i) {
        pResult = _mm256_fmadd_ps(pVecASIMD[i], pVecBSIMD[i], pResult);
    }
        const auto tendTimePoint = std::chrono::high_resolution_clock::now();

    // to avoid horizontal adding extract 128 bits from 256 and sum the up
    __m128 highRes = _mm256_extractf128_ps(pResult, 1); // first 128 bits of data
    __m128 lowRes = _mm256_castps256_ps128(pResult);    // second 128 bits of data
    __m128 sumRes = _mm_add_ps(highRes, lowRes);        // sum up 4 float numbers with 4 float numbers

    double dotProduct = sumM128(sumRes);

    // remainder calculation
    i *= SIMD_AVX_WIDTH; // convert index to normal mode (final unprocessed element index)
    // try using sse first
    if (VEC_SIZE - i >= SIMD_SSE_WIDTH) {
        i /= SIMD_SSE_WIDTH; // convert index to sse mode
        const __m128* pVecASIMDSSE = reinterpret_cast<const __m128*>(pVecA);
        const __m128* pVecBSIMDSSE = reinterpret_cast<const __m128*>(pVecB);
        __m128 res = _mm_setzero_ps();
        const int vecSseSize = VEC_SIZE / SIMD_SSE_WIDTH; // find size of vector in sse mode
        for (; i < vecSseSize; ++i) {
            res = _mm_fmadd_ps(pVecASIMDSSE[i], pVecBSIMDSSE[i], res);
        }
        dotProduct += sumM128(res);

        i *= SIMD_SSE_WIDTH; // convert index back to normal mode
    }
    // final remainder
    for (i; i < VEC_SIZE; ++i) {
        dotProduct += pVecA[i] * pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Dot product: "  << dotProduct << std::endl;

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}