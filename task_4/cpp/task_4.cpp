/**
 * TASK: Matrix Multiplication Using SIMD
 * RESULTS: (for square matrix with dimension = 1000) 
 *      - With optimization flags -O3 -mavx2 -mfma:
 *          - Loop-based execution time: ~1300 ms
 *          - SIMD-based execution time: ~200 ms
 * TODO:
 *  - support of the rectangular matrix with different rows and columns size
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <chrono>       // timestamp
#include <iomanip>      // setprecision

constexpr int32_t MAT_DIM = 1000; // dimension of the matrix
constexpr int32_t MAT_SIZE = MAT_DIM * MAT_DIM;
constexpr int32_t SIMD_AVX_WIDTH = 8;   // 8 bytes - 256 bits
constexpr int32_t SIMD_SSE_WIDTH = 4;   // 4 bytes - 128 bits
constexpr int32_t ALIGNMENT = 32;
constexpr bool PRINT_MAT = false;
constexpr float MAT_A_OFFSET = 0.0f;
constexpr float MAT_B_OFFSET = 1.0f;

bool checkSIMDSupport();

void allocMat(float*& pMat);
void initData(float* pMatA, float* pMatB);
void printMat(float* pMat);
void matMul(float* pMatA, float* pMatB, float* pMatRes);
void matMulSIMD(float* pMatA, float* pMatB, float* pMatRes);

int main()
{
    if (checkSIMDSupport()) {
        return -1;
    }

    float* pMatA = nullptr;
    float* pMatB = nullptr;
    float* pMatRes = nullptr;

    allocMat(pMatA);
    allocMat(pMatB);
    allocMat(pMatRes);

    initData(pMatA, pMatB);

    if (PRINT_MAT) {
        std::cout << "Mat A:\n";
        printMat(pMatA);

        std::cout << "Mat B:\n";
        printMat(pMatB);
    }

    matMul(pMatA, pMatB, pMatRes);
    matMulSIMD(pMatA, pMatB, pMatRes);

    free(pMatA);
    free(pMatB);
    free(pMatRes);

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

void allocMat(float*& pMat)
{
    if (posix_memalign(reinterpret_cast<void**>(&pMat), ALIGNMENT, MAT_SIZE * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
    }
}

void initData(float* pMatA, float* pMatB)
{
    for (int i = 0; i < MAT_SIZE; ++i) {
        pMatA[i] = static_cast<float>(i) + MAT_A_OFFSET;
        pMatB[i] = static_cast<float>(i) + MAT_B_OFFSET;
    }
}

void printMat(float* pMat)
{
    for (int i = 0; i < MAT_DIM; ++i) {
        for (int j = 0; j < MAT_DIM; ++j)
        {
            std::cout << pMat[i * MAT_DIM + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void matMul(float* pMatA, float* pMatB, float* pMatRes)
{
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < MAT_DIM; ++row)
    {
        for (int col = 0; col < MAT_DIM; ++col)
        {
            // find sum for the one element of the result matrix by multipling row and col 
            // of the two matrix per element
            float elementRes = 0.0f;
            for (int i = 0; i < MAT_DIM; ++i)
            {
                elementRes += pMatA[row * MAT_DIM + i] * pMatB[i * MAT_DIM + col];
            }

            pMatRes[row * MAT_DIM + col] = elementRes;
        }
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void matMulSIMD(float* pMatA, float* pMatB, float* pMatRes)
{
    int remainder = MAT_DIM % SIMD_AVX_WIDTH;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < MAT_DIM; ++row)
    {
        for (int col = 0; col < MAT_DIM - remainder; col += SIMD_AVX_WIDTH)
        {
            // find sum for the one element of the result matrix by multipling row and col 
            // of the two matrix per element
            __m256 c = _mm256_setzero_ps(); // Initialize accumulator register to zero
            for (int i = 0; i < MAT_DIM; ++i)
            {
                __m256 a = _mm256_set1_ps(pMatA[row * MAT_DIM + i]);  // Broadcast A[i, k] across all elements
                __m256 b = _mm256_loadu_ps(&pMatB[i * MAT_DIM + col]); // Load 8 floats from B[k, j:j+7]
                c = _mm256_fmadd_ps(a, b, c);             // Multiply and add (fused multiply-add)
            }

            _mm256_storeu_ps(&pMatRes[row * MAT_DIM + col], c);
        }

        // handle the remaining columns using SSE2 if possible
        if (remainder >= SIMD_SSE_WIDTH)
        {
            // // process using sse2
    
            // for (int col = MAT_DIM - remainder; col < MAT_DIM; ++col) {
            //     float sum = 0.0f;
            //     for (int i = 0; i < MAT_DIM; ++i) {
            //         sum += pMatA[row * MAT_DIM + i] * pMatB[i * MAT_DIM + col];
            //     }
            //     pMatRes[row * MAT_DIM + col] = sum;
            // }
            
            // remainder = remainder % SIMD_SSE_WIDTH;
        }

        // handle the remaining columns (remainder part)
        for (int col = MAT_DIM - remainder; col < MAT_DIM; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < MAT_DIM; ++i) {
                sum += pMatA[row * MAT_DIM + i] * pMatB[i * MAT_DIM + col];
            }
            pMatRes[row * MAT_DIM + col] = sum;
        }
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}