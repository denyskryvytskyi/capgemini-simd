/**
 * TASK: String Processing Using SIMD
 * NOTE: Implementation support any substring length less then 16
 * RESULTS: (for string with length = 1'000'000)
 *      - With optimization flags -O3 -msse2 -mfma:
 *          - Loop-based execution time: ~3-3.5 ms
 *          - SIMD-based execution time: ~0.7 ms
 * TODO:
 *  - add support for the substring length greater then 16
 **/ 

#include <immintrin.h>  // AVX
#include <cpuid.h>      // __get_cpuid
#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include <cstring>

#include <bitset>

constexpr bool PRINT_STR = false;
constexpr int32_t ALIGNMENT = 16;
constexpr int32_t MAX_SUB_LENGTH = 16;  // max length of substring that is supported for now
constexpr int32_t STR_LENGTH = 1'000'000;
const char* const PATTERN = "pattern"; 
constexpr int32_t PATTERN_INTERAL = 100; // interval of pattern appearance in generated string

bool isSupportedSSE2();
bool isSupportedAVX();
bool isSupportedAVX512();
bool checkSIMDSupport();

bool allocStr(char*& pStr);
void initStr(char*& pStr);
unsigned int buildMask(int length);
int countSubstringLoop(const char* str, int strLen, const char* sub, int subLen);
int countSubstringSIMD(const char* str, int strLen, const char* sub, int subLen);

int main()
{
    // data definition
    const std::string substring = "pattern";
    char* pStr = nullptr;
    if (allocStr(pStr)) {
        // failed to allocate
        return 0;
    }
    initStr(pStr);

    if (PRINT_STR) {
        std::cout << pStr <<std::endl;
    }

    // Loop-based counting
    auto startTime = std::chrono::high_resolution_clock::now();
    int loopCount = countSubstringLoop(pStr, STR_LENGTH, substring.c_str(), substring.length());
    auto endTime = std::chrono::high_resolution_clock::now();
    const float loopDuration = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    // SIMD-based counting
    startTime = std::chrono::high_resolution_clock::now();
    int simdCount = countSubstringSIMD(pStr, STR_LENGTH, substring.c_str(), substring.length());
    endTime = std::chrono::high_resolution_clock::now();
    const float simdDuration = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    free(pStr);

    // results
    std::cout << "SIMD-based count: " << simdCount << std::endl;
    std::cout << "Loop-based count: " << loopCount << std::endl;
    std::cout << "Loop execution time: " << loopDuration<< " ms" << std::endl;
    std::cout << "SIMD execution time: " << simdDuration << " ms" << std::endl;

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

bool allocStr(char*& pStr)
{
    if (posix_memalign(reinterpret_cast<void**>(&pStr), ALIGNMENT, STR_LENGTH * sizeof(char)) != 0) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
        return 1;
    }

    return 0;
}

void initStr(char*& pStr) {
    std::srand(static_cast<unsigned int>(std::time(0))); // seed random number generator

    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const size_t patternLen = strlen(PATTERN);
    const size_t charsetLen = strlen(charset);
    int patternCount = 0;

    for (size_t i = 0; i < STR_LENGTH; ++i) {
        if (patternCount == PATTERN_INTERAL) {
            // insert the pattern at regular interval
            strcat(pStr, PATTERN);
            i += patternLen;
            patternCount = 0; // reset the counter
        } else {
            // generate a random character from the charset
            char c = charset[std::rand() % charsetLen];
            strncat(pStr, &c, 1); // append a single character
            ++patternCount;
        }
    }
}

unsigned int buildMask(int length) {
    return (1U << length) - 1; // creates a mask of 'length' bits set to 1
}

int countSubstringSIMD(const char* str, int strLen, const char* sub, int subLen) {
    int count = 0;
    if (subLen > strLen){
        std::cout << "Error: substring is longer then string\n";
        return 0;
    }
    if (subLen > MAX_SUB_LENGTH) {
        std::cout << "Error: substring is longer then 16 characters. It isn't supported for now\n";
        return 0;
    }

    unsigned int maskSub = buildMask(subLen); // build bit mask based on substring length

    __m128i pattern;
    if (subLen < MAX_SUB_LENGTH) {
        // if length of the substring less then 16, we need to fill other bytes with 0 values
        char temp[16] = {0}; // initialize a temporary array with 16 bytes
        memcpy(temp, sub, subLen); // copy the substring into the temporary array
        pattern = _mm_loadu_si128(reinterpret_cast<const __m128i*>(temp));
    }
    else {
        pattern = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sub));
    }

    for (int i = 0; i <= strLen - subLen; ++i) {
        if (str[i] == sub[0]) { // check character if it the same as the first character of the substring
            // compare substring and 16 characters of string
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str + i));
            __m128i cmp = _mm_cmpeq_epi8(chunk, pattern);
            unsigned int mask = _mm_movemask_epi8(cmp);

            if (mask == maskSub) { // check whether all needed bits match
                ++count;
                i += subLen - 1; // skip found substring
            }
        }
    }

    return count;
}

int countSubstringLoop(const char* str, int strLen, const char* sub, int subLen) {
    int count = 0;
    for (int i = 0; i <= strLen - subLen; ++i) {
        if (strncmp(str + i, sub, subLen) == 0) {
            count++;
        }
    }
    return count;
}
