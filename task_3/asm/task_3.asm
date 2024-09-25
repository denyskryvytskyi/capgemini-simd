; TASK: Vector Addition and Dot Product Calculation Using SIMD
; RESULTS (for vectors with size 100'000'00):
;   ADDITION:
;       - Loop-based execution time: ~27-30 ms
;       - SIMD-based execution time: ~8 ms
;   DOT PRODUCT:
;       - Loop-based execution time: ~20 ms
;       - SIMD-based execution time: ~3-5 ms
; TODO:
;   - SSE for the remainder if remainder size >= 4

SYS_WRITE equ 1
SYS_EXIT equ 60
STDOUT equ 1

VECTOR_LENGTH equ 10000000                  ; length of the vectors
FLOAT_SIZE equ 4                            ; size in bytes
VECTOR_SIZE equ VECTOR_LENGTH * FLOAT_SIZE  ; 4 float * 4-byte
ALIGNMENT equ 32

PRINT_VEC equ 0                             ; flag to print vectors (1 - print, 0 - don't print)
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

    msg_A db "Vector A: ", 0
    msg_A_len equ $ - msg_A
    msg_B db "Vector B: ", 0
    msg_B_len equ $ - msg_B

    msg_loop_res db "====== Loop-based results ======", 0xa
    msg_loop_res_len equ $ - msg_loop_res

    msg_simd_res db "====== SIMD-based results ======", 0xa
    msg_simd_res_len equ $ - msg_simd_res

    msg_vec_sum_res db "A + B: ", 0
    msg_vec_sum_res_len equ $ - msg_vec_sum_res

    msg_vec_dot_product_res db "A * B: ", 0
    msg_vec_dot_product_res_len equ $ - msg_vec_dot_product_res

    alloc_failed db "Allocation is failed ", 0xa
    alloc_failed_len equ $ - alloc_failed

    msg_timer db "Exectuion time (ms): ", 0
    msg_timer_len equ $ - msg_timer

    fmt_space db "%f ", 0                           ; format string for float with space
    fmt_newline db "%f", 10, 0                      ; format string for last float with newline
    fmt_timer db "Execution time: %d ms.", 10, 0    ; format string for integer (execution time in ms)

    newline_ascii db 0xa                                ; newline character

    START_VECTOR_VAL dd 1.0                         ; start value for the first vector element
    INIT_STEP dd 1.0                                ; step for vector element value on every iteration of init

section .bss
    result resd 1
    vecA_ptr resq 1
    vecB_ptr resq 1
    result_arr_ptr resq 1
    simd_res_vec resd 4     ; array that hold 4 float vector after simd dot product calculations

section .text
    extern posix_memalign
    extern free
    extern printf
    global main

main:
    call check_simd_support
    cmp rax, 1
    je .exit
    ; allocate vector A
    lea rdi, [vecA_ptr]  ; (1st arg) load address of vector (pointer to the pointer)
    call alloc
    test rax, rax        ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; allocate vector B
    lea rdi, [vecB_ptr]  ; (1st arg) load address of vector (pointer to the pointer)
    call alloc
    test rax, rax        ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; allocate result
    lea rdi, [result_arr_ptr] ; (1st arg) load address of vector (pointer to the pointer)
    call alloc
    test rax, rax        ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; init vector A
    mov rdi, [vecA_ptr]            ; load the aligned memory pointer
    call init_data

    ; init vector B
    mov rdi, [vecB_ptr]            ; load the aligned memory pointer
    call init_data

    ; check if we need to print vectors
    mov ax, PRINT_VEC
    test ax, ax
    jz .calculation

    ; print A
    mov rsi, msg_A
    mov rdx, msg_A_len
    call print_string
    mov rbx, [vecA_ptr]
    call print_vector

    ; print B
    mov rsi, msg_B
    mov rdx, msg_B_len
    call print_string
    mov rbx, [vecB_ptr]
    call print_vector

    .calculation:
        ; Loop
        mov rsi, msg_loop_res
        mov rdx, msg_loop_res_len
        call print_string   ; print header
        call add
        call dot_product

        ; SIMD
        mov rsi, msg_simd_res
        mov rdx, msg_simd_res_len
        call print_string   ; print header
        call add_simd
        call dot_product_simd

    ; free memory
    mov rdi, [vecA_ptr]
    call free
    mov rdi, [vecB_ptr]
    call free
    mov rdi, [result_arr_ptr]
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

add:
    mov rcx, VECTOR_LENGTH
    mov rsi, [vecA_ptr]         ; pointer to vector A
    mov rdi, [vecB_ptr]         ; pointer to vector B
    mov rbx, [result_arr_ptr]   ; pointer to result vector

    call timer_start            ; get the start time

    .vector_loop:
        movss xmm0, [rsi]
        movss xmm1, [rdi]
        addss xmm0, xmm1
        movss [rbx], xmm0
        add rsi, FLOAT_SIZE
        add rdi, FLOAT_SIZE
        add rbx, FLOAT_SIZE
    loop .vector_loop

    call timer_end      ; get the end time

    mov ax, PRINT_VEC
    test ax, ax
    jz .exit
    ; print result
    mov rsi, msg_vec_sum_res
    mov rdx, msg_vec_sum_res_len
    call print_string
    mov rbx, [result_arr_ptr]
    call print_vector

    .exit:
        call timer_result   ; print timer results
ret

add_simd:
    mov rcx, VECTOR_LENGTH      ; vector size
    shr rcx, 3                  ; divide by 8 to find amount of 256-register values
    mov rsi, [vecA_ptr]         ; pointer to vector A
    mov rdi, [vecB_ptr]         ; pointer to vector B
    mov rbx, [result_arr_ptr]   ; pointer to result vector
    xor rdx, rdx                ; index of loop

    call timer_start            ; get the start time

    .loop_process_pack:
        cmp rdx, rcx
        jge .check_remainder
        ; load arr_A and arrB into xmm registers
        vmovaps ymm0, [rsi]          ; load vector A into xmm0
        vmovaps ymm1, [rdi]          ; load vector B into xmm1

        ; perform SIMD addition (A + B)
        vaddps ymm0, ymm0, ymm1            ; packed addition of 32-bit integers

        ; store the result in memory
        vmovaps [rbx], ymm0          ; store the result of the addition
        inc rdx
        add rsi, ALIGNMENT
        add rdi, ALIGNMENT
        add rbx, ALIGNMENT
        jmp .loop_process_pack

    .check_remainder:
        shl rdx, 3                     ; multiple by 8 to receive final processed element index
        .loop_process_remainder:
            cmp rdx, VECTOR_LENGTH     ; check index
            jge .done
            movss xmm0, [rsi]          ; element from the vector A
            movss xmm1, [rdi]          ; element from the vector B
            addss xmm0, xmm1           ; A[ebp] + B[ebp]
            movss [rbx], xmm0          ; move to result vector
            inc rdx
            add rsi, FLOAT_SIZE
            add rdi, FLOAT_SIZE
            add rbx, FLOAT_SIZE
            jmp .loop_process_remainder

    .done:
        call timer_end      ; get the end time

        mov ax, PRINT_VEC
        test ax, ax
        jz .exit
        ; print result vector
        mov rsi, msg_vec_sum_res
        mov rdx, msg_vec_sum_res_len
        call print_string
        mov rbx, [result_arr_ptr]
        call print_vector
        jmp .exit

    .exit:
        call timer_result   ; print timer results
ret

; =================== dot product functions ===================

dot_product:
    mov rsi, [vecA_ptr]         ; pointer to vector A
    mov rdi, [vecB_ptr]         ; pointer to vector B
    mov rcx, VECTOR_LENGTH      ; vector size
    xorps xmm0, xmm0            ; result

    call timer_start            ; get the start time
    .process_vectors:
        movss xmm1, [rsi]
        movss xmm2, [rdi]
        vfmadd231ss xmm0, xmm1, xmm2
        add rsi, FLOAT_SIZE
        add rdi, FLOAT_SIZE
        loop .process_vectors
    call timer_end              ; get the end time

    test:
    ; print result
    mov rsi, msg_vec_dot_product_res
    mov rdx, msg_vec_dot_product_res_len
    call print_string
    call print_dot_product
    call timer_result           ; print timer results
ret

dot_product_simd:
    mov rsi, [vecA_ptr]         ; pointer to vector A
    mov rdi, [vecB_ptr]         ; pointer to vector B
    mov rcx, VECTOR_LENGTH      ; vector size
    shr rcx, 3                  ; amount of 8 number blocks
    xor rdx, rdx                ; index
    vxorps ymm0, ymm0           ; result vector

    call timer_start            ; get the start time
    .process_vectors:
        cmp rdx, rcx
        jge .check_remainder
        vmovaps ymm1, [rsi]
        vmovaps ymm2, [rdi]
        vfmadd231ps ymm0, ymm1, ymm2
        inc rdx
        add rsi, ALIGNMENT
        add rdi, ALIGNMENT
        jmp .process_vectors

    ; remainder
    .check_remainder:
        vextractf128 xmm1, ymm0, 0    ; extract the lower 128 bit 
        vextractf128 xmm2, ymm0, 1    ; extract the higher 128 bit
        vaddps xmm1, xmm1, xmm2       ; sum lower 128 bit with the higher one
        xorps xmm0, xmm0              ; reset result
        movaps xmm0, xmm1
        haddps xmm0, xmm0             ; horizontal addition 
        haddps xmm0, xmm0             ; horizontal addition one more. now sum is in xmm0[0]
        shl rdx, 3
        .loop_remainder:
            cmp rdx, VECTOR_LENGTH
            jge .done
            movss xmm1, [rsi]
            movss xmm2, [rdi]
            vfmadd231ss xmm0, xmm1, xmm2
            inc rdx
            add rsi, FLOAT_SIZE
            add rdi, FLOAT_SIZE
            jmp .loop_remainder

    .done:
        call timer_end              ; get the end time

        ; print result
        mov rsi, msg_vec_dot_product_res
        mov rdx, msg_vec_dot_product_res_len
        call print_string
        call print_dot_product
        call timer_result           ; print timer results
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
    mov rdx, VECTOR_SIZE             ; (3d arg) size of the memory block to allocate
    call posix_memalign             ; call posix_memalign
ret

init_data:
    xor rsi, rsi ; index
    movss xmm0, [START_VECTOR_VAL]
    movss xmm1, [INIT_STEP]

    .init:
        cmp rsi, VECTOR_LENGTH
        jge .done_filling                       ; if index >= VECTOR_LENGTH, stop filling

        movss [rdi + rsi * FLOAT_SIZE], xmm0   ; store index as the value in the vector

        addss xmm0, xmm1

        inc rsi                                 ; move to the next index
        jmp .init

    .done_filling:
ret

print_vector:
    ; align the stack (needed for calling printf)
    mov r15, rsp         ; copy rsp to r9 for alignment calculation
    test r15, 15          ; check if rsp is already 16-byte aligned
    jz .aligned         ; if aligned, skip the next instruction

    sub rsp, 8          ; adjust stack pointer if not aligned (subtract 8 to align)

    .aligned:

    xor r12, r12    ; index, non-volatile for printf function call
    xor rsi, rsi

    print_loop:
        cmp r12, VECTOR_LENGTH                  ; compare index with vector size
        jge .end_print_loop                      ; jump if index >= vector size

        movss xmm0, [rbx + r12 * FLOAT_SIZE]    ; move float number to register
        cvtss2sd xmm0, xmm0                     ; convert to double as printf expects double
        mov rax, 1                              ; arguments amount (1 float number)

        cmp r12, VECTOR_LENGTH - 1              ; compare index with vector size
        jge .print_last                         ; jump if index >= vector size - 1

        mov rdi, fmt_space                      ; format string
        call printf                             ; call printf to print the value

        inc r12                                 ; increment the index
        jmp print_loop                          ; repeat the loop

    .print_last:
        mov rdi, fmt_newline                    ; format string
        call printf                             ; call printf to print the value

    .end_print_loop:
        ; check stack pointer alignment
        cmp r15, rsp         ; check if we adjusted the stack (from earlier alignment check)
        je .adjusted        ; jump if rsp was adjusted

        add rsp, 8          ; restore the stack pointer (undo the alignment adjustment)

        .adjusted:
ret

print_dot_product:
    ; align the stack (needed for calling printf)
    mov r15, rsp         ; copy rsp to r9 for alignment calculation
    test r15, 15          ; check if rsp is already 16-byte aligned
    jz .aligned         ; if aligned, skip the next instruction

    sub rsp, 8          ; adjust stack pointer if not aligned (subtract 8 to align)

    .aligned:

    cvtss2sd xmm0, xmm0         ; convert to double as printf expects double
    mov rdi, fmt_newline        ; format string
    mov rax, 1
    call printf                 ; call printf to print the value

    ; check stack pointer alignment
    cmp r15, rsp         ; check if we adjusted the stack (from earlier alignment check)
    je .adjusted        ; jump if rsp was adjusted

    add rsp, 8          ; restore the stack pointer (undo the alignment adjustment)

    .adjusted:
ret

print_string:
    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall
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
    ; align the stack (needed for calling printf)
    mov r15, rsp         ; copy rsp to r9 for alignment calculation
    test r15, 15          ; check if rsp is already 16-byte aligned
    jz .aligned         ; if aligned, skip the next instruction

    sub rsp, 8          ; adjust stack pointer if not aligned (subtract 8 to align)

    .aligned:
    
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

    ; check stack pointer alignment
    cmp r15, rsp         ; check if we adjusted the stack (from earlier alignment check)
    je .adjusted        ; jump if rsp was adjusted

    add rsp, 8          ; restore the stack pointer (undo the alignment adjustment)

    .adjusted:
ret