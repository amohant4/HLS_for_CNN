
#ifndef CONV_BLOCK_SIZE
#define CONV_BLOCK_SIZE 16
#endif

#ifndef CONV_SIMD_ITEMS
#define CONV_SIMD_ITEMS 2
#endif

#ifndef NORM_BLOCK_SIZE
#define NORM_BLOCK_SIZE 3
#endif

#ifndef POOL_SIMD_ITEMS
#define POOL_SIMD_ITEMS 1
#endif

#define LRN 5
#define alpha 0.0001
#define beta 0.75

#ifndef IP_BLOCK_SIZE
#define IP_BLOCK_SIZE 21
#endif

#ifndef QN
#define QN 0
#define QN_IP 3
#endif

#define K_pool 3u
#define S_pool 2u
#define TOP_K 5
