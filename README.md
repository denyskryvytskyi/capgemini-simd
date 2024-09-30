# Capgemini SIMD tasks
Assembly tasks are implemented for the Linux environment (WSL) and NASM assembler.

C++ tasks are implemented for the Linux environment (WSL) and gcc compiler (g++).

## Getting Started
### Work environment preparation
`git clone https://github.com/denyskryvytskyi/capgemini-simd`

`sudo apt update`

`sudo apt install nasm gdb gcc g++`

### Pure assembly program compilation and linking
`nasm -f elf64 task_<n>.asm`

`ld -o task_<n> task_<n>.o`

### Assembly with extern C function program compilation and linking
`nasm -f elf64 task_<n>.asm`

`gcc task_<n>.o -o task_<n> -no-pie`

### C++ program compilation
#### Without optimization flags:
`g++ -o task_<n> task_<n>.cpp`
#### With optimization flags:
**SSE**
`g++ -Wall -o task_<n> task_<n>.cpp -O3 -msse2`

**AVX integer calculations**
`g++ -Wall -o task_<n> task_<n>.cpp -O3 -mavx2`

**AVX float calculations**
`g++ -Wall -o task_<n> task_<n>.cpp -O3 -mavx2 -mfma`

### Run
`./task_<n>`
