# Capgemini SIMD tasks
Assembly tasks are implemented for the Linux environment (WSL) and NASM assembler.

C++ tasks are implemented for the Linux environment (WSL) and g++ compiler.

## Getting Started
### Work environment preparation
`git clone https://github.com/denyskryvytskyi/capgemini-simd`

`sudo apt update`

`sudo apt install nasm gdb gcc g++`

### Pure assembly program compilation and linking
`nasm -f elf64 task_<n>.asm`

`ld -o task_<n> task_<n>.o`

`./task_<n>`

### Assembly with extern C function program compilation and linking
`nasm -f elf64 task_<n>.asm`

`gcc task_<n>.o -o task_<n> -no-pie`

`./task_<n>`

### C++ program compilation
`g++ -o task_<n> task_<n>.cpp`

`./task_<n>`
