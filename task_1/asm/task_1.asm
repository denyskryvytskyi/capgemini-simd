SYS_READ equ 0
SYS_WRITE equ 1
SYS_EXIT equ 60
STDIN equ 0
STDOUT equ 1

ARRAY_SIZE equ 4
INT_SIZE equ 4   ; size in bytes

section .data
    arr_A dd 25, 41, 6, 80      ; Array A
    arr_B dd 11, 2, 3, 50       ; Array B
    result dq 0, 0, 0, 0        ; Result array initialized to 0

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

    newline_ascii db 0xa                                ; newline character
    space_ascii db 0x20                                 ; space character

section .bss
    itoa_result_buffer resb 20  ; buffer to store the number digits in string

section .text
    global _start

_start:
    ; print array A
    mov rsi, msg_A
    mov rdx, msg_A_len
    call print_string
    mov rbx, arr_A
    call print_array

    ; print array B
    mov rsi, msg_B
    mov rdx, msg_B_len
    call print_string
    mov rbx, arr_B
    call print_array

    call add_loop
    call add_simd

    mov eax, SYS_EXIT                   ; sys_exit system call
    xor edi, edi                        ; exit status 0
    
    syscall

; =================== addition functions ===================
add_loop:
    mov ecx, ARRAY_SIZE
    mov esi, arr_A      ; pointer to array A
    mov edi, arr_B      ; pointer to array B
    mov ebx, result     ; pointer to result array
    .array_loop:
        mov eax, [esi]
        add eax, [edi]
        mov [ebx], eax
        add esi, INT_SIZE
        add edi, INT_SIZE
        add ebx, INT_SIZE
    loop .array_loop

    ; print result
    mov rsi, msg_loop_res
    mov rdx, msg_loop_res_len
    call print_string
    mov rbx, result
    call print_array
ret

add_simd:
    ; check for SSE2 support
    mov eax, 1                   ; CPUID function 1
    cpuid
    test edx, 1 << 26            ; Check if SSE2 (bit 26 of edx) is set
    jz .no_sse2

    ; check for AVX support
    test ecx, 1 << 28            ; Check if AVX (bit 28 of ecx) is set
    jz .no_avx

    ; load arr_A and arrB into xmm registers
    movaps xmm0, [arr_A]          ; Load array A into xmm0
    movaps xmm1, [arr_B]          ; Load array B into xmm1

    ; perform SIMD addition (A + B)
    paddd xmm0, xmm1             ; Packed addition of 32-bit integers

    ; store the result in memory
    movaps [result], xmm0        ; Store the result of the addition

    ; print result
    mov rsi, msg_simd_res
    mov rdx, msg_simd_res_len
    call print_string
    mov rbx, result
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

print_string:
    push rcx
    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall
    pop rcx
ret

itoa:
    push rcx

    mov rdi, itoa_result_buffer ; point edi to the end of the buffer

    add rdi, 19                 ; fill it from right to left
    mov rcx, 10                 ; divisor for extracting digits

    .convert_loop:
        xor rdx, rdx            ; clear edx for division
        div rcx                 ; divide rax by 10, remainder in edx
        add dl, '0'             ; convert remainder to ASCII
        dec rdi                 ; move buffer pointer left
        mov [rdi], dl           ; store the digit
        test rax, rax           ; check if quotient is zero
        jnz .convert_loop       ; if not, continue loop

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
    add rdx, 20
    sub rdx, rdi                 ; rdx now holds the string length

    mov rsi, rdi
    mov rdi, STDOUT
    mov rax, SYS_WRITE
    syscall

    pop rcx
ret

print_array:
    mov ecx, ARRAY_SIZE         ; size of array for loop counter
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