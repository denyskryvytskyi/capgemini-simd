/**
 * TASK: Vector Addition and Dot Product Calculation Using SIMD
 * RESULTS: (for vector with size equals 100'000'000) 
 *      ADDITION:
 *      - With optimization flag -mavx2 only:
 *          - Loop-based execution time: ~300-400 ms
 *          - SIMD-based execution time: ~75 ms (overall faster)
 *      - With optimization flags -O3 -mavx2:
 *          - Loop-based execution time: ~100-180 ms
 *          - SIMD-based execution time: ~60-80 ms (overall faster)
 *      DOT PRODUCT:
 *      - With optimization flag -mavx2 only:
 *          - Loop-based execution time: ~200-300 ms
 *          - SIMD-based execution time: ~70-80 ms (overall faster)
 *      - With optimization flags -O3 -mavx2:
 *          - Loop-based execution time: ~130 ms
 *          - SIMD-based execution time: ~40 ms (overall faster)
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <chrono>       // timestamp
#include <iomanip>      // setprecision

constexpr int32_t VEC_SIZE = 100'000'000;
constexpr int32_t FLOAT_AMOUNT_PER_SIMD_REG = 8;
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_VEC = false;
constexpr float VEC_A_OFFSET = 0.2f;
constexpr float VEC_B_OFFSET = 1.3f;
constexpr int32_t FLOAT_PRECISION = 4;

bool checkSIMDSupport();

void printArray(float* arr);
void allocArray(float*& pVec);
void initData(float* pVecA, float* pVecB);
void add(float* pVecA, float* pVecB, float* pRes);
void addSIMD(float* pVecA, float* pVecB, float* pRes);
// dot product functions here
void dotProduct(float* pVecA, float* pVecB);
void dotProductSIMD(float* pVecA, float* pVecB);

int main()
{
    if (checkSIMDSupport())
    {
        return -1;
    }

    float* pVecA = nullptr;
    float* pVecB = nullptr;
    float* pRes = nullptr;

    allocArray(pVecA);
    allocArray(pVecB);
    allocArray(pRes);

    initData(pVecA, pVecB);

    if (PRINT_VEC) {
        std::cout << "Vector A: ";
        printArray(pVecA);

        std::cout << "Vector B: ";
        printArray(pVecB);
    }

    add(pVecA, pVecB, pRes);
    addSIMD(pVecA, pVecB, pRes);

    dotProduct(pVecA, pVecB);
    dotProductSIMD(pVecA, pVecB);

    free(pVecA);
    free(pVecB);
    free(pRes);

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

bool checkSIMDSupport()
{
    if (!isSupportedSSE2()) {
        std::cerr << "SSE2 is not supported on this CPU." << std::endl;
        return -1;
    }
    if (!isSupportedAVX()) {
        std::cerr << "AVX is not supported on this CPU." << std::endl;

        return -1;
    }
    if (!isSupportedAVX512()) {
        // std::cerr << "AVX512 is not supported on this CPU." << std::endl;
    }
    
    return 0;
}

void printArray(float* arr)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}


void allocArray(float*& pVec)
{
    if (posix_memalign(reinterpret_cast<void**>(&pVec), ALIGNMENT, VEC_SIZE * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
    }
}

void initData(float* pVecA, float* pVecB)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecA[i] = VEC_A_OFFSET;
        pVecB[i] = VEC_B_OFFSET;
    }
}

void add(float* pVecA, float* pVecB, float* pRes)
{
    std::cout << "===== Loop-based addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        pRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_VEC) {
        std::cout << "Result of A + B: ";
        printArray(pRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void addSIMD(float* pVecA, float* pVecB, float* pRes)
{
    std::cout << "===== SIMD-based addition =====\n";

    const __m256* pVecASIMD = reinterpret_cast<const __m256*>(pVecA);
    const __m256* pVecBSIMD = reinterpret_cast<const __m256*>(pVecB);
    __m256* pResSIMD = reinterpret_cast<__m256*>(pRes);

    const int32_t resultArraySize = VEC_SIZE / FLOAT_AMOUNT_PER_SIMD_REG;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (; i < resultArraySize; ++i)
    {
        pResSIMD[i] = _mm256_add_ps(pVecASIMD[i], pVecBSIMD[i]);
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // remainder calculation
    for (int j = i * FLOAT_AMOUNT_PER_SIMD_REG; j < VEC_SIZE; ++j) {
        pRes[j] = pVecA[j] + pVecB[j];
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";

    if (PRINT_VEC) {
        std::cout << "Result of A + B: ";
        printArray(pRes);
    }
}

void dotProduct(float* pVecA, float* pVecB)
{
    std::cout << "===== Loop-based dot product =====\n";
    double result = 0.0f;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        // result += static_cast<double>(pVecA[i]) * static_cast<double>(pVecB[i]);
        result += pVecA[i] * pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Dot product: " <<  result << std::endl;

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void dotProductSIMD(float* pVecA, float* pVecB)
{
    std::cout << "===== SIMD-based dot product =====\n";
    // we need to multiply 8 elements at time and then add to the 8 element result array
    // then we need to add all elements in result 8-element array

    const __m256* pVecASIMD = reinterpret_cast<const __m256*>(pVecA);
    const __m256* pVecBSIMD = reinterpret_cast<const __m256*>(pVecB);
    const int32_t resultArraySize = VEC_SIZE / FLOAT_AMOUNT_PER_SIMD_REG;

    // accumulator that hold sum up of every 8 float elements multiplication
    __m256 pResult = _mm256_setzero_ps();

    int i = 0;
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (; i < resultArraySize; ++i)
    {
        pResult = _mm256_add_ps(pResult, _mm256_mul_ps(pVecASIMD[i], pVecBSIMD[i]));
    }

    __m128 highRes = _mm256_extractf128_ps(pResult, 1); // first 128 bits of data
    __m128 lowRes = _mm256_castps256_ps128(pResult);    // second 128 bits of data
    __m128 sumRes = _mm_add_ps(highRes, lowRes);        // sum up 4 float numbers with 4 float numbers

    float dotProductArray[4];
    _mm_storeu_ps(dotProductArray, sumRes);             // convert to float array

    double dotProduct = 0.0f;
    dotProduct += dotProductArray[0] + dotProductArray[1]  + dotProductArray[2] + dotProductArray[3];

    // remainder calculation
    for (int j = i * FLOAT_AMOUNT_PER_SIMD_REG; j < VEC_SIZE; ++j) {
        dotProduct += pVecA[j] * pVecB[j];
    }

    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Dot product: "  << dotProduct << std::endl;

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}