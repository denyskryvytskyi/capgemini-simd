; TASK: Matrix Multiplication Using SIMD
; NOTE: we align dynamically alloted matrices, but due to the multiplication algorithm there is an edge case if we have matrix dimension that is not multiple of 8. Then we have situation when we try to load unaligned memory to simd registers. So to handle this I have used instructions for unaligned memory location for now. Possible solutions also mentioned in TODO.
; RESULTS (for matrices with dimension 1000):
;       - Loop-based execution time: ~1700-1800 ms
;       - SIMD-based execution time: ~450 ms
; TODO (possible improvements):
;   - reactangle matrix processing (with different row and column dimensions)
;   - usage of simd instructions for aligned memory when matrix dimension is multiple of 8:
;       - add conditional logic depending on dimension directly in current algorithm
;       - make seperate implementation of the algorithm in other procedure
;       - other algorithm for multiplication approach (e.g. transpose second matrix)

SYS_WRITE equ 1
SYS_EXIT equ 60
STDOUT equ 1

MAT_DIM equ 1000                              ; matrix dimension
FLOAT_SIZE equ 4                            ; size in bytes
MAT_LENGTH equ MAT_DIM * MAT_DIM            ; amount of numbers in matrix
MAT_SIZE equ MAT_LENGTH * FLOAT_SIZE        ; size in bytes
SIMD_AVX_WIDTH equ 8                        ; 8 float numbers
ALIGNMENT equ 32

PRINT_MAT equ 0                             ; flag to print vectors (1 - print, 0 - don't print)
CPU_FREQ equ 2808000000                     ; CPU frequency of my CPU for execution time calculation
MS_IN_SEC equ 1000                          ; ms in one sec

section .data
    start_time dq 0
    end_time dq 0

section .rodata
    msg_SSE2 db "SSE2 is not supported on this CPU.", 0
    msg_SSE2_len equ $ - msg_SSE2
    msg_AVX db "AVX is not supported on this CPU.", 0
    msg_AVX_len equ $ - msg_AVX

    msg_A db "Matrix A: ", 0xa
    msg_A_len equ $ - msg_A
    msg_B db "Matrix B: ", 0xa
    msg_B_len equ $ - msg_B
    msg_res db "Result Matrix: ", 0xa
    msg_res_len equ $ - msg_res
    msg_test db "== test ==", 0xa
    msg_test_len equ $ - msg_test

    msg_loop_res db "====== Loop-based results ======", 0xa
    msg_loop_res_len equ $ - msg_loop_res

    msg_simd_res db "====== SIMD-based results ======", 0xa
    msg_simd_res_len equ $ - msg_simd_res

    alloc_failed db "Allocation is failed ", 0xa
    alloc_failed_len equ $ - alloc_failed

    msg_timer db "Exectuion time (ms): ", 0
    msg_timer_len equ $ - msg_timer

    fmt_space db "%f ", 0                           ; format string for float with space
    fmt_newline db "%f", 10, 0                      ; format string for last float with newline
    fmt_timer db "Execution time: %d ms.", 10, 0    ; format string for integer (execution time in ms)

    newline_ascii db 0xa                            ; newline character

    START_MAT_VAL dd 1.0                            ; start value for the first vector element
    INIT_STEP dd 1.0                                ; step for vector element value on every iteration of init

section .bss
    matA_ptr resq 1
    matB_ptr resq 1
    mat_res_ptr resq 1

section .text
    extern posix_memalign
    extern free
    extern printf
    global main

main:
    call check_simd_support
    cmp rax, 1
    je .exit

    ; allocate matrix A
    lea rdi, [matA_ptr]     ; (1st arg) load address of matrix (pointer to the pointer)
    call alloc
    test rax, rax           ; check if rax (return value) is 0 (success)
    jnz .alloc_failed       ; if not zero, allocation failed

    ; allocate matrix B
    lea rdi, [matB_ptr]     ; (1st arg) load address of matrix (pointer to the pointer)
    call alloc
    test rax, rax           ; check if rax (return value) is 0 (success)
    jnz .alloc_failed       ; if not zero, allocation failed

    ; allocate result matrix
    lea rdi, [mat_res_ptr]  ; (1st arg) load address of matrix (pointer to the pointer)
    call alloc
    test rax, rax           ; check if rax (return value) is 0 (success)
    jnz .alloc_failed       ; if not zero, allocation failed

    ; init matrix A
    mov rdi, [matA_ptr]     ; load the aligned memory pointer
    call init_data

    ; init matrix B
    mov rdi, [matB_ptr]     ; load the aligned memory pointer
    call init_data

    ; check if we need to print matrices
    mov ax, PRINT_MAT
    test ax, ax
    jz .calculation

    ; print A
    mov rsi, msg_A
    mov rdx, msg_A_len
    call print_string
    mov rbx, [matA_ptr]
    call print_mat

    ; print B
    mov rsi, msg_B
    mov rdx, msg_B_len
    call print_string
    mov rbx, [matB_ptr]
    call print_mat

    .calculation:
        ; Loop
        mov rsi, msg_loop_res
        mov rdx, msg_loop_res_len
        call print_string   ; print header
        call multiply

        ; SIMD
        mov rsi, msg_simd_res
        mov rdx, msg_simd_res_len
        call print_string   ; print header
        call multiply_simd

    ; free memory
    mov rdi, [matA_ptr]
    call free
    mov rdi, [matB_ptr]
    call free
    mov rdi, [mat_res_ptr]
    call free

    jmp .exit

    .alloc_failed:
        mov rsi, alloc_failed
        mov rdx, alloc_failed_len
        call print_string

    .exit:
        mov eax, SYS_EXIT                   ; sys_exit system call
        xor edi, edi                        ; exit status 0
        syscall

; =================== addition functions ===================

multiply:
    push rbp
    mov rbp, rsp

    mov rcx, MAT_DIM
    mov rsi, [matA_ptr]      ; pointer to matrix A
    mov rdi, [matB_ptr]      ; pointer to matrix B
    mov rbx, [mat_res_ptr]   ; pointer to result matrix

    xor r12, r12             ; row index
    xor r13, r13             ; column index
    xor r14, r14             ; element index

    call timer_start            ; get the start time
    .row_loop:
        cmp r12, rcx
        jge .done

        xor r13, r13
        .col_loop:
            cmp r13, rcx
            jge .done_col_loop  ; next row if column loop is done

            xor r14, r14
            xorps xmm0, xmm0
            .element_loop:
                cmp r14, rcx
                jge .done_element_loop                  ; next row if column loop is done

                ; find index of element from matrix A
                mov rax, r12
                mul rcx
                add rax, r14
                shl rax, 2  ; multiply by 4 (size of float)
                movss xmm1, [rsi + rax]    ; mov element from matrix A

                ; find index of element from matrix B
                mov rax, r14
                mul rcx
                add rax, r13
                shl rax, 2  ; multiply by 4 (size of float)
                movss xmm2, [rdi + rax]    ; add element from matrix B
                mulss xmm1, xmm2
                addss xmm0, xmm1
                inc r14
                jmp .element_loop

                .done_element_loop:
                    ; find index of element for result matrix
                    mov rax, r12
                    mul rcx
                    add rax, r13
                    shl rax, 2  ; multiply by 4 (size of float)
                    movss [rbx + rax], xmm0
                    inc r13
                    jmp .col_loop

            .done_col_loop:
                inc r12
                jmp .row_loop

    .done:
        call timer_end              ; get the end time

        mov ax, PRINT_MAT
        test ax, ax
        jz .exit
        ; print result matrix
        mov rsi, msg_res
        mov rdx, msg_res_len
        call print_string

        mov rbx, [mat_res_ptr]
        call print_mat

    .exit:
        call timer_result           ; print timer results
        mov rsp, rbp
        pop rbp
ret

multiply_simd:
    push rbp
    mov rbp, rsp

    mov rsi, [matA_ptr]      ; pointer to matrix A
    mov rdi, [matB_ptr]      ; pointer to matrix B
    mov rbx, [mat_res_ptr]   ; pointer to result matrix

    xor r12, r12             ; matrix A index
    xor r13, r13             ; matrix B index
    xor r14, r14             ; element index

    ; find remainder
    mov eax, MAT_DIM
    mov ecx, SIMD_AVX_WIDTH
    xor edx, edx            ; clear edx for division
    div ecx                 ; divide eax by ecx, now edx contains the remainder

    ; Algorithm of multiplication:
    ; 1. mat_A_loop (outer) - loop through the matrix A by row per iteration
    ; 2. mat_B_loop (inner) - loop through the matrix A by row per iteration
    ; 3. row loop (inner for matrix B loop) -> process 8 pack numbers per iteration.
    ; One iteration of the outer loop -> final row values for result matrix
    ; Remainder check every iteration for outer loop (if there are remaining elements in matrix B)
    call timer_start         ; get the start time
    .mat_A_loop:
        cmp r12, MAT_DIM
        jge .done

        xor r13, r13         ; reset index
        .mat_B_loop:
            cmp r13, MAT_DIM
            jge .next_B_row  ; next row of the matrix B

            xor r14, r14
            vxorps ymm0, ymm0
            .row_loop:
                cmp r14, MAT_DIM
                jge .store_element              ; next 8 float pack

                ; find index of element from matrix A
                mov rax, MAT_DIM                ; matrix dimension
                mul r12                         ; multiplied by current row from matrix A
                add rax, r14                    ; add current row from matrix B
                shl rax, 2                      ; multiply by 4 (size of float)
                vbroadcastss ymm1, [rsi + rax]  ; mov 1 elements from matrix A to all values of ymm

                ; find index of pack of elements from matrix B
                mov rax, MAT_DIM                ; matrix dimension
                mul r14                         ; multiplied by current row from matrix A
                add rax, r13                    ; add current row from matrix B
                shl rax, 2                      ; multiply by 4 (size of float)
                vmovups ymm2, [rdi + rax]       ; move 8 element from matrix B
                vfmadd231ps ymm0, ymm1, ymm2    ; multiply element from A with 8 elements from B and add to accumulator
                inc r14
                jmp .row_loop

                .store_element:
                    ; find index of element for result matrix
                    mov rax, MAT_DIM
                    mul r12
                    add rax, r13
                    shl rax, 2  ; multiply by 4 (size of float)
                    vmovups [rbx + rax], ymm0
                    add r13, 8  ; next 8 block numbers
                    jmp .mat_B_loop

            .next_B_row:
                ; check remainder
                test rdx, rdx
                jz .next_iteration

                ; calculate remainder
                mov r13, MAT_DIM  ; start index
                sub r13, rdx
                .remainder_loop:
                    cmp r13, MAT_DIM
                    jge .next_iteration
                    xor r14, r14
                    xorps xmm0, xmm0
                    .inner_loop:
                        cmp r14, MAT_DIM
                        jge .done_inner_loop
                        ; find index of element from matrix A
                        mov rax, MAT_DIM
                        mul r12
                        add rax, r14
                        shl rax, 2  ; multiply by 4 (size of float)
                        movss xmm1, [rsi + rax]    ; mov element from matrix A

                        ; find index of element from matrix B
                        mov rax, MAT_DIM
                        mul r14
                        add rax, r13
                        shl rax, 2  ; multiply by 4 (size of float)
                        movss xmm2, [rdi + rax]    ; add element from matrix B
                        mulss xmm1, xmm2
                        addss xmm0, xmm1
                        inc r14
                        jmp .inner_loop

                        .done_inner_loop:
                            ; find index of element for result matrix
                            mov rax, MAT_DIM
                            mul r12
                            add rax, r13
                            shl rax, 2  ; multiply by 4 (size of float)
                            movss [rbx + rax], xmm0
                            inc r13
                            jmp .remainder_loop
                .next_iteration:
                    inc r12
                    jmp .mat_A_loop

    .done:
        call timer_end              ; get the end time

        mov ax, PRINT_MAT
        test ax, ax
        jz .exit
        ; print result matrix
        mov rsi, msg_res
        mov rdx, msg_res_len
        call print_string

        mov rbx, [mat_res_ptr]
        call print_mat

    .exit:
        call timer_result           ; print timer results
        mov rsp, rbp
        pop rbp
ret

; =================== helpers ===================
check_simd_support:
    ; check for SSE2 support
    mov eax, 1                   ; CPUID function 1
    cpuid
    test edx, 1 << 26            ; check if SSE2 (bit 26 of edx) is set
    jz .no_sse2

    ; check for AVX support
    test ecx, 1 << 28            ; check if AVX (bit 28 of ecx) is set
    jnz .exit

    ; if no SSE2 support, print error message
    mov esi, msg_SSE2
    mov edx, msg_SSE2_len
    call print_string
    mov rax, 1              ; error code
    jmp .exit

    .no_sse2:
        ; if no AVX support, print error message
        mov esi, msg_AVX
        mov edx, msg_AVX_len
        call print_string
        mov rax, 1              ; error code

    .exit:
ret

alloc:
    mov rsi, ALIGNMENT              ; (2d arg) alignment
    mov rdx, MAT_SIZE               ; (3d arg) size of the memory block to allocate
    call posix_memalign             ; call posix_memalign
ret

init_data:
    xor rsi, rsi ; index
    movss xmm0, [START_MAT_VAL]
    movss xmm1, [INIT_STEP]

    .init:
        cmp rsi, MAT_LENGTH
        jge .done_filling                      ; if index >= VECTOR_LENGTH, stop filling

        movss [rdi + rsi * FLOAT_SIZE], xmm0   ; store index as the value in the vector

        addss xmm0, xmm1

        inc rsi                                ; move to the next index
        jmp .init

    .done_filling:
ret

print_mat:
    xor r12, r12    ; row index, non-volatile for printf function call
    xor r14, r14
    xor rsi, rsi

    .print_row_loop:
        cmp r12, MAT_DIM
        jge .end_loop
        xor r13, r13    ; column index, non-volatile for printf function call
        .print_col_loop:
            ; find index of element
            mov r14, MAT_DIM
            imul r14, r12
            add r14, r13

            movss xmm0, [rbx + r14 * FLOAT_SIZE]    ; move float number to register
            cvtss2sd xmm0, xmm0                     ; convert to double as printf expects double
            mov rax, 1                              ; arguments amount (1 float number)

            cmp r13, MAT_DIM - 1                    ; compare index with matrix dimension
            jge .print_last_in_row                  ; jump if index >= matrix dimension - 1

            mov rdi, fmt_space                      ; format string
            call printf                             ; call printf to print the value

            inc r13                                 ; increment the column index
            jmp .print_col_loop                     ; repeat the loop

            .print_last_in_row:
                mov rdi, fmt_newline                ; format string
                call printf                         ; call printf to print the value
                inc r12
                jmp .print_row_loop                  ; next row

    .end_loop:
ret

print_string:
push rbx
    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall
    pop rbx
ret

print_newline:
    mov rsi, newline_ascii      ; address of newline character
    mov rdx, 1                  ; length
    mov rdi, STDOUT
    mov rax, SYS_WRITE
    syscall
ret

timer_start:
    push rdx

    rdtsc                       ; read time-stamp counter into EDX:EAX
    shl rdx, 32                 ; shift RDX left by 32 bits
    or rax, rdx                 ; combine into a 64-bit value (RAX)
    mov [start_time], rax       ; store start time

    pop rdx
ret

timer_end:
    rdtsc                       ; read time-stamp counter into EDX:EAX
    shl rdx, 32                 ; shift RDX left by 32 bits
    or rax, rdx                 ; combine into a 64-bit value (RAX)
    mov [end_time], rax         ; store end time
ret

timer_result:
    ; calculate elapsed CPU cycles
    mov rax, [end_time]         ; load end time
    sub rax, [start_time]       ; subtract start time to get elapsed cycles

    ; convert cycles to milliseconds: elapsed * 1000 / CPU frequency
    mov rbx, CPU_FREQ           ; load CPU frequency
    mov rcx, MS_IN_SEC          ; conversion factor (1000 ms)
    mul rcx                     ; rax = elapsed * 1000
    div rbx                     ; rax = (elapsed * 1000) / CPU frequency

    ; print result
    mov rsi, rax                                ; move the integer to be printed into rsi
    mov rdi, fmt_timer                          ; format string for integer
    xor rax, rax                                ; printf uses rax to count floating-point args, set it to 0
    call printf                                 ; call printf to print the integer
ret