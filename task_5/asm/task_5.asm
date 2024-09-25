; TASK: String Processing Using SIMD
; NOTE: 
;    - I have implemented string reading from file to test on really large string.
;    - Implementation support search only for substring with length <= 16.
; RESULTS (with the string length equals 50'000'000):
;       - Loop-based execution time: ~60-50 ms
;       - SIMD-based execution time: ~50-55 ms (approximately 10 ms less than loop-based)
; CONCLUSION: According to the performance results improvement isn't too big. So, there is a bottleneck in simd substring search implementation because we compare the first character of the string with the first character of the substring on every iteration of the main loop, and only then (when the first characters are the same) use simd for substring check.
; TODO (improvements/optimizations):
;   - loop unrolling: split the big string of the smaller parts and search in all of them in one iteration of the main loop iteration.
;   - reduce the amount of branching in simd implementation
;   - AVS register usage (if the substring is less than 16, then we can check in 32 bytes of string at once)

SYS_READ equ 0
SYS_WRITE equ 1
SYS_EXIT equ 60
SYS_OPEN equ 2      ; open file
SYS_CLOSE equ 3     ; close file
STDOUT equ 1

PRINT_STRING equ 0         ; flag to print string and substring
CPU_FREQ equ 2808000000     ; CPU frequency of my CPU for execution time calculation
MS_IN_SEC equ 1000          ; ms in one sec
STRING_SIZE equ 50000000    ; size in bytes of string to read from file)
ERROR_FILE_NOT_FOUND equ -2

section .data
    start_time dq 0
    end_time dq 0
    string_length dq 0

section .rodata
    align 16
    substring db "lorem"
    substr_len equ $ - substring

    INPUT_FILE_NAME db "input.txt", 0
    
    msg_loop_res db "====== Loop-based results ======", 0xa
    msg_loop_res_len equ $ - msg_loop_res

    msg_simd_res db "====== SIMD-based results ======", 0xa
    msg_simd_res_len equ $ - msg_simd_res

    error_file_msg db "Error: file not found", 0xa
    error_file_msg_len equ $ - error_file_msg

    fmt_count db "Substring count: %d.", 0xa, 0     ; format string for substring count
    fmt_timer db "Execution time: %d ms.", 10, 0    ; format string for execution time in ms

section .bss
    file_content_buffer resb STRING_SIZE

section .text
    extern printf
    global main

main:
    call init_string_from_file
    cmp rax, ERROR_FILE_NOT_FOUND
    je .exit

    mov esi, file_content_buffer ; pointer to string
    mov edi, substring           ; pointer to substring
    mov ecx, [string_length]

    call count_substring
    call count_substring_simd

    .exit:
    mov eax, SYS_EXIT         ; exit system call
    xor edi, edi              ; return code 0
    syscall

; =================== substring match functions ===================

count_substring:
    push rdi
    push rsi
    push rcx

    ; align the stack (needed for calling printf)
    mov r9, rsp         ; copy rsp to r9 for alignment calculation
    test r9, 15          ; check if rsp is already 16-byte aligned
    jz .aligned         ; if aligned, skip the next instruction

    sub rsp, 8          ; adjust stack pointer if not aligned (subtract 8 to align)

    .aligned:

    xor eax, eax       ; init counter of the substring
    xor r10, r10       ; init counter of the chars in one substring search check

    call timer_start
    .substring_search:
        ; find starting char of substring
        mov r8b, [edi]              ; load first character of substring into low byte of r8
        mov r9b, [esi]              ; load first character of string into low byte of r9b
        cmp r8b, r9b                ; compare the characters
        jne .next_sub               ; if chars aren't equal go to the next iteration

        inc r10                     ; increment chars count
        ; now compare all subsequent characters to find substring
        mov ebx, edi                ; copy pointer to the substring
        inc ebx                     ; next char
        mov edx, esi                ; copy pointer to the string
        inc edx                     ; next char
        mov r11, substr_len         ; counter for the compare_loop (substring length - 1)
        sub r11, 1
        .compare_loop:
            mov r8b, [ebx]              ; load character of string1 into low byte of r8
            mov r9b, [edx]              ; load character of string2 into low byte of r9b

            cmp r8b, r9b                ; check if there was a match
            jne .next_sub               ; if no match - next iteration

            inc r10                     ; if match, increment chars count

            ; check if we have already found substring
            cmp r10, substr_len
            jne .next_char

            ; increment substring count
            inc eax
            jmp .next_sub

            .next_char:
                dec r11
                test r11, r11
                jz .next_sub
                inc ebx
                inc edx
                jmp .compare_loop


        .next_sub:
            xor r10, r10
            add esi, 1                ; move pointer to next position in large string
            dec ecx                   ; decrease remaining length
            cmp ecx, substr_len       ; stop if there’s no room for substring
            jge .substring_search     ; repeat loop 

    ; result is now stored in rax
    call timer_end

    ; print result
    mov rsi, msg_loop_res
    mov rdx, msg_loop_res_len
    call print_string

    mov rsi, rax                ; move the integer to be printed into rsi
    mov rdi, fmt_count          ; format string for integer
    xor rax, rax                ; printf uses rax to count floating-point args, set it to 0
    call printf

    call timer_result

    ; check stack pointer alignment
    cmp r9, rsp         ; check if we adjusted the stack (from earlier alignment check)
    je .exit            ; jump if rsp was adjusted

    add rsp, 8          ; restore the stack pointer (undo the alignment adjustment)

    .exit:

    pop rcx
    pop rsi
    pop rdi
ret

count_substring_simd:
    xor ebx, ebx              ; substring mask
    call build_mask           ; ebx now containes substring mask

    xor eax, eax              ; init counter
    xor edx, edx              ; compare mask

    movdqu xmm1, [edi]        ; load substring (assuming length <= 16)
    call timer_start
    .substring_search:
        ; find starting char of substring
        mov r8b, [esi]             ; load first character of substring
        mov r9b, [edi]             ; load first character of string
        cmp r8b, r9b               ; compare the characters
        jne .next                  ; if chars aren't equal go to the next iteration

        movdqu xmm0, [esi]        ; load 16 bytes from larger string

        pcmpeqb xmm0, xmm1        ; compare bytes in xmm0 and xmm1
        pmovmskb edx, xmm0        ; move result to edx as a mask

        and edx, ebx              ; bitwise AND to get only needed bits
        cmp edx, ebx              ; check if there was a match
        jne .next                 ; if no match - next iteration

        inc eax                   ; if match, increment the count

        .next:
            add esi, 1                ; move pointer to next position in larger string
            dec ecx                   ; decrease remaining length
            cmp ecx, substr_len       ; stop if there’s no room for substring
            jge .substring_search     ; repeat loop

    ; result is now stored in rax
    call timer_end

    ; print result
    mov rsi, msg_simd_res
    mov rdx, msg_simd_res_len
    call print_string

    mov rsi, rax                                ; move the integer to be printed into rsi
    mov rdi, fmt_count                          ; format string for integer
    xor rax, rax                                ; printf uses rax to count floating-point args, set it to 0
    call printf

    call timer_result
ret

; =================== helpers ===================

init_string_from_file:
    ; open input file
    mov rax, SYS_OPEN
    mov rdi, INPUT_FILE_NAME
    xor rsi, rsi                        ; read-only
    xor rdx, rdx                        ; no flags
    syscall                             ; return: rax contains file desriptor
    mov rdi, rax                        ; save file descriptor

    ; check errors
    cmp rax, ERROR_FILE_NOT_FOUND
    je .error_file_not_found

    ; get input file content
    mov rax, SYS_READ
    mov rsi, file_content_buffer
    mov rdx, STRING_SIZE
    syscall                             ; return: rax contains number of reading bytes
    mov [string_length], rax            ; length of the reading bytes

    ; close input file
    mov rax, SYS_CLOSE
    mov rdi, rdi                        ; use saved file descriptor
    syscall

    ret

    .error_file_not_found:
        mov esi, error_file_msg
        mov edx, error_file_msg_len
        call print_string
ret

build_mask:
    mov ebx, 1
    shl ebx, substr_len     ; bitwise shift left by length of the substring
    sub ebx, 1              ; flips all lowest 'length' 0 bits to 1
ret

print_string:
    push rax

    mov rax, SYS_WRITE
    mov rdi, STDOUT
    syscall

    pop rax
ret

timer_start:
    push rdx
    push rax

    rdtsc                       ; read time-stamp counter into EDX:EAX
    shl rdx, 32                 ; shift RDX left by 32 bits
    or rax, rdx                 ; combine into a 64-bit value (RAX)
    mov [start_time], rax       ; store start time

    pop rax
    pop rdx
ret

timer_end:
    push rdx
    push rax

    rdtsc                       ; read time-stamp counter into EDX:EAX
    shl rdx, 32                 ; shift RDX left by 32 bits
    or rax, rdx                 ; combine into a 64-bit value (RAX)
    mov [end_time], rax         ; store end time

    pop rax
    pop rdx
ret

timer_result:
    ; align the stack (needed for calling printf)
    mov r9, rsp         ; copy rsp to r9 for alignment calculation
    test r9, 15          ; check if rsp is already 16-byte aligned
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
    cmp r9, rsp         ; check if we adjusted the stack (from earlier alignment check)
    je .exit            ; jump if rsp was adjusted

    add rsp, 8          ; restore the stack pointer (undo the alignment adjustment)

    .exit:
ret