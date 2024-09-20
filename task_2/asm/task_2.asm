; TASK: Data Alignment and Memory Access
; RESULTS (for arrays with size 100'000'000)):
;  - Loop-based execution time: ~200-300 ms
;  - SIMD-based execution time: ~60-70 ms
; TODO:
;   - SSE and MMX for remainder if remainder size >= 4 or >= 2

SYS_WRITE equ 1
SYS_EXIT equ 60
STDOUT equ 1

ARRAY_LENGTH equ 100000000                  ; length of the arrays
INT_SIZE equ 4                              ; size in bytes
ARRAY_SIZE equ ARRAY_LENGTH * INT_SIZE      ; 4 integers * 4-byte
ALIGNMENT equ 32
ITOA_BUFFER_SIZE equ 10                     ; size in bytes

PRINT_ARRAYS equ 0                          ; flag to print arrays (1 - print, 0 - don't print)
CPU_FREQ equ 2808000000                     ; CPU frequency for execution time calculation
MS_IN_SEC equ 1000

section .data
    start_time dq 0
    end_time dq 0

section .rodata
    msg_SSE2 db "SSE2 is not supported on this CPU.", 0
    msg_SSE2_len equ $ - msg_SSE2
    msg_AVX db "AVX is not supported on this CPU.", 0
    msg_AVX_len equ $ - msg_AVX

    msg_A db "Array A: ", 0
    msg_A_len equ $ - msg_A
    msg_B db "Array B: ", 0
    msg_B_len equ $ - msg_B

    msg_loop_res db "====== Loop-based results ======", 0
    msg_loop_res_len equ $ - msg_loop_res

    msg_simd_res db "====== SIMD-based results ======", 0
    msg_simd_res_len equ $ - msg_simd_res

    msg_arr_sum_res db "A + B: ", 0
    msg_arr_sum_res_len equ $ - msg_arr_sum_res

    alloc_failed db "Allocation is failed ", 0
    alloc_failed_len equ $ - alloc_failed

    alloc_success db "Allocation is succesfull ", 0
    alloc_success_len equ $ - alloc_success

    msg_timer db "Exectuion time (ms): ", 0
    msg_timer_len equ $ - msg_timer

    newline_ascii db 0xa                                ; newline character
    space_ascii db 0x20                                 ; space character

section .bss
    itoa_result_buffer resb 20  ; buffer to store the number digits in string
    arrA_ptr resq 1
    arrB_ptr resq 1
    result_arr_ptr resq 1

section .text
    extern posix_memalign
    extern free
    global main

main:
    ; allocate array A
    lea rdi, [arrA_ptr] ; (1st arg) load address of array (pointer to the pointer)
    call alloc
    test rax, rax       ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; allocate array B
    lea rdi, [arrB_ptr] ; (1st arg) load address of array (pointer to the pointer)
    call alloc
    test rax, rax       ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; allocate result
    lea rdi, [result_arr_ptr] ; (1st arg) load address of array (pointer to the pointer)
    call alloc
    test rax, rax       ; check if rax (return value) is 0 (success)
    jnz .alloc_failed    ; if not zero, allocation failed

    ; init array A
    mov rdi, [arrA_ptr]            ; Load the aligned memory pointer
    call init_data

    ; init array B
    mov rdi, [arrB_ptr]            ; Load the aligned memory pointer
    call init_data

    ; print array A
    mov ax, PRINT_ARRAYS
    test ax, ax
    jz .calculation
    mov rsi, msg_A
    mov rdx, msg_A_len
    call print_string
    mov rbx, [arrA_ptr]
    call print_array

    ; print array B
    mov rsi, msg_B
    mov rdx, msg_B_len
    call print_string
    mov rbx, [arrB_ptr]
    call print_array

    .calculation:
        call add_loop
        call add_simd

    ; free memory
    mov rdi, [arrA_ptr]
    call free
    mov rdi, [arrB_ptr]
    call free
    mov rdi, [result_arr_ptr]
    call free

    jmp .exit

    .alloc_failed:
        mov rsi, alloc_failed
        mov rdx, alloc_failed_len
        call print_string
        call print_newline

    .exit:
        mov eax, SYS_EXIT                   ; sys_exit system call
        xor edi, edi                        ; exit status 0
        syscall

; =================== addition functions ===================
add_loop:
    mov rcx, ARRAY_LENGTH
    mov rsi, [arrA_ptr]         ; pointer to array A
    mov rdi, [arrB_ptr]         ; pointer to array B
    mov rbx, [result_arr_ptr]   ; pointer to result array

    call timer_start            ; get the start time

    .array_loop:
        mov rax, [rsi]
        add rax, [rdi]
        mov [rbx], rax
        add rsi, INT_SIZE
        add rdi, INT_SIZE
        add rbx, INT_SIZE
    loop .array_loop

    call timer_end              ; get the end time

    mov rsi, msg_loop_res
    mov rdx, msg_loop_res_len
    call print_string   ; print header
    call print_newline
    call timer_result   ; print timer results

    mov ax, PRINT_ARRAYS
    test ax, ax
    jz .exit
    ; print result
    mov rsi, msg_arr_sum_res
    mov rdx, msg_arr_sum_res_len
    call print_string
    mov rbx, [result_arr_ptr]
    call print_array

    .exit:
ret

add_simd:
    ; check for SSE2 support
    mov eax, 1                   ; CPUID function 1
    cpuid
    test edx, 1 << 26            ; check if SSE2 (bit 26 of edx) is set
    jz .no_sse2

    ; check for AVX support
    test ecx, 1 << 28            ; check if AVX (bit 28 of ecx) is set
    jz .no_avx

    ; calculate sum
    mov rcx, ARRAY_LENGTH       ; array size
    shr rcx, 3                  ; divide by 8 to find amount of 256-register values
    mov rsi, [arrA_ptr]         ; pointer to array A
    mov rdi, [arrB_ptr]         ; pointer to array B
    mov rbx, [result_arr_ptr]   ; pointer to result array
    mov rbp, 0                  ; index of loop

    call timer_start            ; get the start time

    .loop_process_pack:
        cmp rbp, rcx
        jge .check_remainder
        ; load arr_A and arrB into xmm registers
        vmovaps ymm0, [rsi]          ; load array A into xmm0
        vmovaps ymm1, [rdi]          ; load array B into xmm1

        ; perform SIMD addition (A + B)
        vpaddd ymm0, ymm1            ; packed addition of 32-bit integers

        ; store the result in memory
        vmovaps [rbx], ymm0          ; store the result of the addition
        inc rbp
        add rsi, ALIGNMENT
        add rdi, ALIGNMENT
        add rbx, ALIGNMENT
        jmp .loop_process_pack

    .check_remainder:
        shl ebp, 3                  ; multiple by 8 to receive final processed element index
        .loop_process_remainder:
            cmp ebp, ARRAY_LENGTH   ; check index
            jge .done
            mov eax, [esi]          ; element from the array A
            add eax, [edi]          ; add element from the array B
            mov [ebx], eax          ; move to result array
            inc ebp
            add esi, INT_SIZE
            add edi, INT_SIZE
            add ebx, INT_SIZE
            jmp .loop_process_remainder

    .done:
        call timer_end      ; get the end time
        mov rsi, msg_simd_res
        mov rdx, msg_simd_res_len
        call print_string   ; print header
        call print_newline
        call timer_result   ; print timer results

        mov ax, PRINT_ARRAYS
        test ax, ax
        jz .exit
        ; print result array
        mov rsi, msg_arr_sum_res
        mov rdx, msg_arr_sum_res_len
        call print_string
        mov rbx, [result_arr_ptr]
        call print_array
        jmp .exit

    .no_sse2:
        ; if no SSE2 support, print error message
        mov esi, msg_SSE2
        mov edx, msg_SSE2_len
        call print_string
        jmp .exit

    .no_avx:
        ; if no AVX support, print error message
        mov esi, msg_AVX
        mov edx, msg_AVX_len
        call print_string

    .exit:
ret

; =================== helpers ===================

alloc:
    mov rsi, ALIGNMENT              ; (2d arg) alignment
    mov rdx, ARRAY_SIZE             ; (3d arg) size of the memory block to allocate
    call posix_memalign             ; call posix_memalign
ret

init_data:
    xor rsi, rsi ; index

    .init:
        cmp rsi, ARRAY_LENGTH
        jge .done_filling                       ; if index >= ARRAY_LENGTH, stop filling
        mov dword [rdi + rsi * INT_SIZE], esi   ; store index as the value in the array
        inc rsi                                 ; move to the next index
        jmp .init

    .done_filling:
ret

print_string:
    push rcx
    push rax

    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall

    pop rax
    pop rcx
ret

itoa:
    push rcx

    mov rdi, itoa_result_buffer     ; point edi to the end of the buffer

    add rdi, ITOA_BUFFER_SIZE - 1   ; fill it from right to left
    mov rcx, 10                     ; divisor for extracting digits

    .convert_loop:
        xor rdx, rdx                ; clear edx for division
        div rcx                     ; divide rax by 10, remainder in edx
        add dl, '0'                 ; convert remainder to ASCII
        dec rdi                     ; move buffer pointer left
        mov [rdi], dl               ; store the digit
        test rax, rax               ; check if quotient is zero
        jnz .convert_loop           ; if not, continue loop

    .done:
    pop rcx
ret


print_newline:
    mov rsi, newline_ascii      ; address of newline character
    mov rdx, 1                  ; length
    mov rdi, STDOUT
    mov rax, SYS_WRITE
    syscall
ret

print_int:
    push rcx

    mov rdx, itoa_result_buffer
    add rdx, ITOA_BUFFER_SIZE
    sub rdx, rdi                 ; rdx now holds the string length

    mov rsi, rdi
    mov rdi, STDOUT
    mov rax, SYS_WRITE
    syscall

    pop rcx
ret

print_array:
    mov ecx, ARRAY_LENGTH         ; size of array for loop counter
    .loop_array:
        mov eax, [ebx]        ; next number
        call itoa
        call print_int

        ; print space
        mov esi, space_ascii
        mov edx, 1
        call print_string

        add ebx, INT_SIZE  ; move pointer to the next element
        loop .loop_array

    call print_newline
ret

timer_start:
    rdtsc                       ; read time-stamp counter into EDX:EAX
    shl rdx, 32                 ; shift RDX left by 32 bits
    or rax, rdx                 ; combine into a 64-bit value (RAX)
    mov [start_time], rax       ; store start time
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
    mov rbx, CPU_FREQ         ; load CPU frequency
    mov rcx, MS_IN_SEC        ; conversion factor (1000 ms)
    mul rcx                   ; rax = elapsed * 1000
    div rbx                   ; rax = (elapsed * 1000) / CPU frequency

    ; print result
    mov esi, msg_timer
    mov edx, msg_timer_len
    call print_string
    call itoa
    call print_int
    call print_newline
ret