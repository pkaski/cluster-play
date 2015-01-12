#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<immintrin.h> 
#include<omp.h>

#include"matrix_mynative.h"

JNIEXPORT void JNICALL Java_matrix_mynative_ping(JNIEnv *env, jobject obj) 
{
    printf("Greetings from native library!\n");
}

//#define INNER_TEST    // test the performance of the inner loop only
#define DO_TRANSPOSE  // transpose input operands and result
#define PREFETCH      // run prefetch for next mid block

JNIEXPORT void JNICALL Java_matrix_mynative_mulpar(JNIEnv *env, jobject obj, jint size, jdoubleArray result, jdoubleArray left, jdoubleArray right)
{
    double wstart = omp_get_wtime();

    jsize len = (*env)->GetArrayLength(env, result);

    assert((size % 128) == 0);
    assert(size*size == len);
    int size_inner = 32;
    int size_mid = 128/32;     // size in inner blocks
    int size_outer = size/128; // size in mid blocks

    int len_inner = size_inner*size_inner;
    int len_mid = size_mid*size_mid*len_inner;

    double *result_elem = (double *) (*env)->GetPrimitiveArrayCritical(env, result, 0);
    double *left_elem   = (double *) (*env)->GetPrimitiveArrayCritical(env, left, 0);
    double *right_elem  = (double *) (*env)->GetPrimitiveArrayCritical(env, right, 0);
    double *result_a = (double *) 0;
    double *left_a   = (double *) 0;
    double *right_a  = (double *) 0;
    posix_memalign((void**)&result_a, 32, sizeof(double)*len);
    posix_memalign((void**)&left_a,   32, sizeof(double)*len);
    posix_memalign((void**)&right_a,  32, sizeof(double)*len);

#ifdef DO_TRANSPOSE
    /* Transpose left input to aligned & cache-blocked form. */
    {       
        int dim[8];
        int perm[8];
        int rank = 6;
        dim[0] = size_inner;
        dim[1] = size_mid;
        dim[2] = size_outer;
        dim[3] = size_inner;
        dim[4] = size_mid;
        dim[5] = size_outer;
        // newdim[j] == dim[perm[j]]
        perm[0] = 0;
        perm[1] = 3;
        perm[2] = 1;
        perm[3] = 4;
        perm[4] = 2;
        perm[5] = 5;
        long below[8];
        for(int d = 0; d < rank; d++) {
            int pd = perm[d];
            long b = 1;
            for(int j = 0; j < pd; j++)
                b *= dim[j];
            below[d] = b;
        }
        #pragma omp parallel for
        for(long u = 0; u < len; u++) {
            long uu = u;
            long v = 0;
            for(int d = 0; d < rank; d++) {
                v += (uu%dim[perm[d]])*below[d];
                uu /= dim[perm[d]];
            }       
            left_a[u] = left_elem[v];
        }
    }
#else
    #pragma omp parallel for
    for(long u = 0; u < len; u++)
        left_a[u] = left_elem[u];    
#endif

#ifdef DO_TRANSPOSE
    /* Transpose right input to aligned & cache-blocked form. */
    {       
        int dim[8];
        int perm[8];
        int rank = 6;
        dim[0] = size_inner;
        dim[1] = size_mid;
        dim[2] = size_outer;
        dim[3] = size_inner;
        dim[4] = size_mid;
        dim[5] = size_outer;
        // newdim[j] == dim[perm[j]]
        perm[0] = 0;
        perm[1] = 3;
        perm[2] = 1;
        perm[3] = 4;
        perm[4] = 2;
        perm[5] = 5;
        long below[8];
        for(int d = 0; d < rank; d++) {
            int pd = perm[d];
            long b = 1;
            for(int j = 0; j < pd; j++)
                b *= dim[j];
            below[d] = b;
        }
        #pragma omp parallel for
        for(long u = 0; u < len; u++) {
            long uu = u;
            long v = 0;
            for(int d = 0; d < rank; d++) {
                v += (uu%dim[perm[d]])*below[d];
                uu /= dim[perm[d]];
            }       
            right_a[u] = right_elem[v];
        }
    }
#else
    #pragma omp parallel for
    for(long u = 0; u < len; u++)
        right_a[u] = right_elem[u];
#endif

    #pragma omp parallel for
    for(long i = 0; i < len; i++)
        result_a[i] = 0.0;    

    double wstart_inner = omp_get_wtime();

    double *c = result_a;
    double *a = left_a;
    double *b = right_a;

    #pragma omp parallel for
    for(int q = 0; q < size_outer*size_outer; q++) {
        int i_outer = q / size_outer;
        int k_outer = q % size_outer;
        double *c_outer = c + (i_outer*size_outer + k_outer)*len_mid;
        for(int j_outer = 0; j_outer < size_outer; j_outer++) {
            double *a_outer = a + (i_outer*size_outer + j_outer)*len_mid;
            double *b_outer = b + (j_outer*size_outer + k_outer)*len_mid;
#ifdef PREFETCH
            double *a_outer_p = a + (i_outer*size_outer + j_outer + 1)*len_mid;
            double *b_outer_p = b + ((j_outer+1)*size_outer + k_outer)*len_mid;
#endif
            for(int p = 0; p < size_mid*size_mid; p++) {
                int i_mid = p / size_mid;
                int k_mid = p % size_mid;
                double *c_mid = c_outer + (i_mid*size_mid + k_mid)*len_inner;
                for(int j_mid = 0; j_mid < size_mid; j_mid++) {
                    double *a_mid = a_outer + (i_mid*size_mid + j_mid)*len_inner;
                    double *b_mid = b_outer + (j_mid*size_mid + k_mid)*len_inner;
                    for(int i = 0; i < 30; i+=3) { 
#ifdef PREFETCH
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
#endif
                        for(int k = 0; k < size_inner; k += 16) {
#ifdef INNER_TEST
                            double *a_inner = a_outer + i*size_inner;
                            double *b_inner = b_outer + k;
                            double *c_inner = c_outer + i*size_inner + k;
#else
                            double *a_inner = a_mid + i*size_inner;
                            double *b_inner = b_mid + k;
                            double *c_inner = c_mid + i*size_inner + k;
#endif
                            int iter = size_inner;
                            __asm__ volatile
                                (  /* Assembler template */
                                 
                                 "  mov            %[c], %%rbx                  \n\t"
                                 "  mov            %[wd], %%rcx                 \n\t"
                                 "  shl            $0x3, %%rcx                  \n\t"
                                 "  vmovapd        (%%rbx), %%ymm4              \n\t"
                                 "  vmovapd        (%%rbx,%%rcx), %%ymm5        \n\t"
                                 "  vmovapd        (%%rbx,%%rcx,2), %%ymm6      \n\t"
                                 "  vmovapd        0x20(%%rbx), %%ymm7          \n\t"
                                 "  vmovapd        0x20(%%rbx,%%rcx), %%ymm8    \n\t"
                                 "  vmovapd        0x20(%%rbx,%%rcx,2), %%ymm9  \n\t"
                                 "  vmovapd        0x40(%%rbx), %%ymm10         \n\t"
                                 "  vmovapd        0x40(%%rbx,%%rcx), %%ymm11   \n\t"
                                 "  vmovapd        0x40(%%rbx,%%rcx,2), %%ymm12 \n\t"
                                 "  vmovapd        0x60(%%rbx), %%ymm13         \n\t"
                                 "  vmovapd        0x60(%%rbx,%%rcx), %%ymm14   \n\t"
                                 "  vmovapd        0x60(%%rbx,%%rcx,2), %%ymm15 \n\t"
                                 "  mov            %[a], %%rdx                  \n\t"
                                 "  mov            %[b], %%rax                  \n\t"
                                 "  mov            %[iter], %%rbx               \n\t"
                                 
                                 ".testinner10:                                 \n\t"                         
                                 "  vbroadcastsd   (%%rdx), %%ymm0              \n\t"
                                 "  vbroadcastsd   (%%rdx,%%rcx), %%ymm1        \n\t"
                                 "  vbroadcastsd   (%%rdx,%%rcx,2), %%ymm2      \n\t"
                                 "  add            $0x08, %%rdx                 \n\t"
                                 "  vmovapd        (%%rax), %%ymm3              \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm4       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm5       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm6       \n\t"
                                 "  vmovapd        0x20(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm7       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm8       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm9       \n\t"
                                 "  vmovapd        0x40(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm10      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm11      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm12      \n\t"
                                 "  vmovapd        0x60(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm13      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm14      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm15      \n\t"
                                 "  add            %%rcx, %%rax                 \n\t"
                                 "  dec            %%rbx                        \n\t"
                                 "  jnz            .testinner10                 \n\t"
                                 
                                 "  mov            %[c], %%rbx                  \n\t"
                                 "  vmovapd        %%ymm4, (%%rbx)              \n\t"
                                 "  vmovapd        %%ymm5, (%%rbx,%%rcx)        \n\t"
                                 "  vmovapd        %%ymm6, (%%rbx,%%rcx,2)      \n\t"
                                 "  vmovapd        %%ymm7, 0x20(%%rbx)          \n\t"
                                 "  vmovapd        %%ymm8, 0x20(%%rbx,%%rcx)    \n\t"
                                 "  vmovapd        %%ymm9, 0x20(%%rbx,%%rcx,2)  \n\t"
                                 "  vmovapd        %%ymm10, 0x40(%%rbx)         \n\t"
                                 "  vmovapd        %%ymm11, 0x40(%%rbx,%%rcx)   \n\t"
                                 "  vmovapd        %%ymm12, 0x40(%%rbx,%%rcx,2) \n\t"
                                 "  vmovapd        %%ymm13, 0x60(%%rbx)         \n\t"
                                 "  vmovapd        %%ymm14, 0x60(%%rbx,%%rcx)   \n\t"
                                 "  vmovapd        %%ymm15, 0x60(%%rbx,%%rcx,2) \n\t"
                                 
                                 /* 
                                  * Format for operands:
                                  *   [{asm symbolic name}] "{constraint}" ({C variable name})
                                  * Reference with "%[{asm symbolic name}]" in assembler template
                                  * Constraints:
                                  *   =  ~ overwrite               [for output operands]
                                  *   +  ~ both read and write     [for output operands]
                                  *   r  ~ register
                                  *   m  ~ memory
                                  */
                                 : /* Output operands (comma-separated list) */
                                 : /* Input operands (comma-separated list) */
                                 [iter] "r" ((long) iter),
                                 [wd] "r" ((long) size_inner),
                                 [c]  "r" (c_inner),
                                 [a]  "r" (a_inner),
                                 [b]  "r" (b_inner)
                                 : /* Clobbers (comma-separated list of registers, e.g. "ymm12", 
                                    *           "memory" for universal mem clobber) */
                                 "rax",
                                 "rbx",
                                 "rcx",
                                 "rdx",
                                 "ymm0", 
                                 "ymm1", 
                                 "ymm2", 
                                 "ymm3", 
                                 "ymm4", 
                                 "ymm5", 
                                 "ymm6", 
                                 "ymm7", 
                                 "ymm8", 
                                 "ymm9", 
                                 "ymm10", 
                                 "ymm11", 
                                 "ymm12", 
                                 "ymm13", 
                                 "ymm14", 
                                 "ymm15",
                                 "memory"
                                   );
                        }
                    }

                    for(int i = 30; i < 32; i+=3) {
#ifdef PREFETCH
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
#endif
                        for(int k = 0; k < size_inner; k += 16) {
#ifdef INNER_TEST
                            double *a_inner = a_outer + i*size_inner;
                            double *b_inner = b_outer + k;
                            double *c_inner = c_outer + i*size_inner + k;
#else
                            double *a_inner = a_mid + i*size_inner;
                            double *b_inner = b_mid + k;
                            double *c_inner = c_mid + i*size_inner + k;
#endif                          
                            int iter = size_inner;
                            __asm__ volatile
                                (  /* Assembler template */
                                 
                                 "  mov            %[c], %%rbx                  \n\t"
                                 "  mov            %[wd], %%rcx                 \n\t"
                                 "  shl            $0x3, %%rcx                  \n\t"
                                 "  vmovapd        (%%rbx), %%ymm4              \n\t"
                                 "  vmovapd        (%%rbx,%%rcx), %%ymm5        \n\t"
//                                 "  vmovapd        (%%rbx,%%rcx,2), %%ymm6      \n\t"
                                 "  vmovapd        0x20(%%rbx), %%ymm7          \n\t"
                                 "  vmovapd        0x20(%%rbx,%%rcx), %%ymm8    \n\t"
//                                 "  vmovapd        0x20(%%rbx,%%rcx,2), %%ymm9  \n\t"
                                 "  vmovapd        0x40(%%rbx), %%ymm10         \n\t"
                                 "  vmovapd        0x40(%%rbx,%%rcx), %%ymm11   \n\t"
//                                 "  vmovapd        0x40(%%rbx,%%rcx,2), %%ymm12 \n\t"
                                 "  vmovapd        0x60(%%rbx), %%ymm13         \n\t"
                                 "  vmovapd        0x60(%%rbx,%%rcx), %%ymm14   \n\t"
//                                 "  vmovapd        0x60(%%rbx,%%rcx,2), %%ymm15 \n\t"
                                 "  mov            %[a], %%rdx                  \n\t"
                                 "  mov            %[b], %%rax                  \n\t"
                                 "  mov            %[iter], %%rbx               \n\t"
                                 
                                 ".testinner1:                                  \n\t"                         
                                 "  vbroadcastsd   (%%rdx), %%ymm0              \n\t"
                                 "  vbroadcastsd   (%%rdx,%%rcx), %%ymm1        \n\t"
//                                 "  vbroadcastsd   (%%rdx,%%rcx,2), %%ymm2      \n\t"
                                 "  add            $0x08, %%rdx                 \n\t"
                                 "  vmovapd        (%%rax), %%ymm3              \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm4       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm5       \n\t"
//                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm6       \n\t"
                                 "  vmovapd        0x20(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm7       \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm8       \n\t"
//                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm9       \n\t"
                                 "  vmovapd        0x40(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm10      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm11      \n\t"
//                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm12      \n\t"
                                 "  vmovapd        0x60(%%rax), %%ymm3          \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm13      \n\t"
                                 "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm14      \n\t"
//                                 "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm15      \n\t"
                                 "  add            %%rcx, %%rax                 \n\t"
                                 "  dec            %%rbx                        \n\t"
                                 "  jnz            .testinner1                  \n\t"
                                 
                                 "  mov            %[c], %%rbx                  \n\t"
                                 "  vmovapd        %%ymm4, (%%rbx)              \n\t"
                                 "  vmovapd        %%ymm5, (%%rbx,%%rcx)        \n\t"
//                                 "  vmovapd        %%ymm6, (%%rbx,%%rcx,2)      \n\t"
                                 "  vmovapd        %%ymm7, 0x20(%%rbx)          \n\t"
                                 "  vmovapd        %%ymm8, 0x20(%%rbx,%%rcx)    \n\t"
//                                 "  vmovapd        %%ymm9, 0x20(%%rbx,%%rcx,2)  \n\t"
                                 "  vmovapd        %%ymm10, 0x40(%%rbx)         \n\t"
                                 "  vmovapd        %%ymm11, 0x40(%%rbx,%%rcx)   \n\t"
//                                 "  vmovapd        %%ymm12, 0x40(%%rbx,%%rcx,2) \n\t"
                                 "  vmovapd        %%ymm13, 0x60(%%rbx)         \n\t"
                                 "  vmovapd        %%ymm14, 0x60(%%rbx,%%rcx)   \n\t"
//                                 "  vmovapd        %%ymm15, 0x60(%%rbx,%%rcx,2) \n\t"                         
                                 
                                 /* 
                                  * Format for operands:
                                  *   [{asm symbolic name}] "{constraint}" ({C variable name})
                                  * Reference with "%[{asm symbolic name}]" in assembler template
                                  * Constraints:
                                  *   =  ~ overwrite               [for output operands]
                                  *   +  ~ both read and write     [for output operands]
                                  *   r  ~ register
                                  *   m  ~ memory
                                  */
                                 : /* Output operands (comma-separated list) */
                                 : /* Input operands (comma-separated list) */
                                 [iter] "r" ((long) iter),
                                 [wd] "r" ((long) size_inner),
                                 [c]  "r" (c_inner),
                                 [a]  "r" (a_inner),
                                 [b]  "r" (b_inner)
                                 : /* Clobbers (comma-separated list of registers, e.g. "ymm12", 
                                    *           "memory" for universal mem clobber) */
                                 "rax",
                                 "rbx",
                                 "rcx",
                                 "rdx",
                                 "ymm0", 
                                 "ymm1", 
                                 "ymm2", 
                                 "ymm3", 
                                 "ymm4", 
                                 "ymm5", 
                                 "ymm6", 
                                 "ymm7", 
                                 "ymm8", 
                                 "ymm9", 
                                 "ymm10", 
                                 "ymm11", 
                                 "ymm12", 
                                 "ymm13", 
                                 "ymm14", 
                                 "ymm15",
                                 "memory"
                                   );
                        }
                    }
                }
            }
#ifdef PREFETCH
            assert(a_outer_p == a + (i_outer*size_outer + j_outer + 2)*len_mid);
            assert(b_outer_p == b + ((j_outer + 1)*size_outer + k_outer + 1)*len_mid);
#endif
        }
    }

    double wstop_inner = omp_get_wtime();
    double wtime_inner = (double) (1000.0*(wstop_inner-wstart_inner));

#ifdef DO_TRANSPOSE
    /* Transpose result from cache-block to original form. */
    {       
        int dim[8];
        int perm[8];
        int rank = 6;
        dim[0] = size_inner;
        dim[1] = size_inner;
        dim[2] = size_mid;
        dim[3] = size_mid;
        dim[4] = size_outer;
        dim[5] = size_outer;
        // newdim[j] == dim[perm[j]]
        perm[0] = 0;
        perm[1] = 2;
        perm[2] = 4;
        perm[3] = 1;
        perm[4] = 3;
        perm[5] = 5;
        long below[8];
        for(int d = 0; d < rank; d++) {
            int pd = perm[d];
            long b = 1;
            for(int j = 0; j < pd; j++)
                b *= dim[j];
            below[d] = b;
        }
        #pragma omp parallel for
        for(long u = 0; u < len; u++) {
            long uu = u;
            long v = 0;
            for(int d = 0; d < rank; d++) {
                v += (uu%dim[perm[d]])*below[d];
                uu /= dim[perm[d]];
            }       
            result_elem[u] = result_a[v];
        }
    }
#else
    #pragma omp parallel for
    for(long u = 0; u < len; u++)
        result_elem[u] = result_a[u];   
#endif 

    free(result_a);
    free(left_a);
    free(right_a);  
   
    (*env)->ReleasePrimitiveArrayCritical(env, result, result_elem, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, left, left_elem, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, right, right_elem, 0);

    double wstop = omp_get_wtime();
    double wtime = (double) (1000.0*(wstop-wstart));

    fprintf(stdout, 
            "wtime = %9.3f ms, perf = %6.2f GFLOPS [ %6.2f GFLOPS/core];   wtime_inner = %9.3f ms, perf_inner = %6.2f GFLOPS [ %6.2f GFLOPS/core]\n",
            wtime,
            (((double) size)*size*size*2.0)/(wtime*1e6),
            (((double) size)*size*size*2.0)/(wtime*1e6)/omp_get_max_threads(),
            wtime_inner,
            (((double) size)*size*size*2.0)/(wtime_inner*1e6),
            (((double) size)*size*size*2.0)/(wtime_inner*1e6)/omp_get_max_threads());
    fflush(stdout);
}

