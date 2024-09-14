; TASK: Data Alignment and Memory Access
; TODO:
;  - performance measure
; 

SYS_WRITE equ 1
SYS_EXIT equ 60
STDOUT equ 1

ARRAY_LENGTH equ 18                         ; length of the arrays
INT_SIZE equ 4                              ; size in bytes
ARRAY_SIZE equ ARRAY_LENGTH * INT_SIZE      ; 4 integers * 4-byte
ALIGNMENT equ 16
ITOA_BUFFER_SIZE equ 10                     ; size in bytes

section .data
    msg_SSE2 db "SSE2 is not supported on this CPU.", 0
    msg_SSE2_len equ $ - msg_SSE2
    msg_AVX db "AVX is not supported on this CPU.", 0
    msg_AVX_len equ $ - msg_AVX

    msg_A db "Array A: ", 0
    msg_A_len equ $ - msg_A
    msg_B db "Array B: ", 0
    msg_B_len equ $ - msg_B

    msg_loop_res db "Result of A + B (Loop): ", 0
    msg_loop_res_len equ $ - msg_loop_res

    msg_simd_res db "Result of A + B (SIMD): ", 0
    msg_simd_res_len equ $ - msg_simd_res

    alloc_failed db "Allocation is failed ", 0
    alloc_failed_len equ $ - alloc_failed

    alloc_success db "Allocation is succesfull ", 0
    alloc_success_len equ $ - alloc_success

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
    mov rsi, [arrA_ptr]                  ; pointer to array A
    mov rdi, [arrB_ptr]                  ; pointer to array B
    mov rbx, [result_arr_ptr]                  ; pointer to result array
    .array_loop:
        mov rax, [rsi]
        add rax, [rdi]
        mov [rbx], rax
        add rsi, INT_SIZE
        add rdi, INT_SIZE
        add rbx, INT_SIZE
    loop .array_loop

    ; print result
    mov rsi, msg_loop_res
    mov rdx, msg_loop_res_len
    call print_string
    mov rbx, [result_arr_ptr]
    call print_array
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
    mov ecx, ARRAY_LENGTH   ; array size
    shr ecx, 2              ; divide by 4 to find amount of 128-register values
    mov esi, [arrA_ptr]     ; pointer to array A
    mov edi, [arrB_ptr]     ; pointer to array B
    mov ebx, [result_arr_ptr]         ; pointer to result array
    mov ebp, 0              ; index of loop
    .loop_process_pack:
        cmp ebp, ecx
        jge .check_remainder
        ; load arr_A and arrB into xmm registers
        movups xmm0, [esi]          ; load array A into xmm0
        movups xmm1, [edi]          ; load array B into xmm1

        ; perform SIMD addition (A + B)
        paddd xmm0, xmm1            ; packed addition of 32-bit integers

        ; store the result in memory
        movups [ebx], xmm0          ; store the result of the addition
        inc ebp
        add esi, 16
        add edi, 16
        add ebx, 16
        jmp .loop_process_pack

    .check_remainder:
        shl ebp, 2                  ; multiple by 4 to receive final processed element index
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
        ; print result
        mov rsi, msg_simd_res
        mov rdx, msg_simd_res_len
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
    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall
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