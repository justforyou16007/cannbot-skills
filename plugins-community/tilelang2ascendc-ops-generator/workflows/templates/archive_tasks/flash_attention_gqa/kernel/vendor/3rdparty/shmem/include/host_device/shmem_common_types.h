/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @private
*/
#define ACLSHMEM_GLOBAL __global__ __aicore__
/**
 * @addtogroup group_macros
 * @{
*/
/**
 * @def ACLSHMEM_DEVICE
 * @brief A macro that identifies a function on the device side.
*/
#define ACLSHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__
/**@} */ // end of group_enums
/**
 * @addtogroup group_enums
 * @{
*/

/**
* @brief The state of the ACLSHMEM host OP type.
*/
enum aclshmemi_op_t {
    ACLSHMEMI_OP_PUT = 0,
    ACLSHMEMI_OP_P,
    ACLSHMEMI_OP_PUT_SIGNAL,
    ACLSHMEMI_OP_GET,
    ACLSHMEMI_OP_G,
    // ACLSHMEMI_OP_FENCE,
    // ACLSHMEMI_OP_AMO,
    // ACLSHMEMI_OP_QUIET,
    // ACLSHMEMI_OP_SENTINEL = INT_MAX,
};

/**
 * @brief Team's index.
*/
enum aclshmem_team_index_t {
    ACLSHMEM_TEAM_INVALID = -1,
    ACLSHMEM_TEAM_WORLD = 0
};

/**
 * @brief Data operation engine type.
*/
enum data_op_engine_type_t {
    ACLSHMEM_DATA_OP_MTE = 0x01,
    ACLSHMEM_DATA_OP_SDMA = 0x02,
    ACLSHMEM_DATA_OP_ROCE = 0x04,
};

/**
 * @brief Memory type of NPU or host
 */
enum aclshmem_mem_type_t {
    HOST_SIDE = 0, // The memory address allocated by shmem is on the host side
    DEVICE_SIDE    // The memory address allocated by shmem is on the device side
};

/**
 * @brief Signal operations, used by the signaler in peer-to-peer synchronization
 */
enum {
    ACLSHMEM_SIGNAL_SET,
    ACLSHMEM_SIGNAL_ADD
};

/**
 * @brief Signal comparison operations, used by the signalee in peer-to-peer synchronization
 */
enum {
    ACLSHMEM_CMP_EQ = 0,
    ACLSHMEM_CMP_NE,
    ACLSHMEM_CMP_GT,
    ACLSHMEM_CMP_GE,
    ACLSHMEM_CMP_LT,
    ACLSHMEM_CMP_LE
};

/**@} */ // end of group_enums

/**
 * @defgroup group_typedef Typedef
 * @{
*/
/**
 * @brief A typedef of int
*/
typedef int aclshmem_team_t;

/**
 * @brief A typedef of uint64_t
*/
typedef uint64_t aclshmemx_team_uniqueid_t;

/**@} */ // end of group_typedef


/**
 * @addtogroup group_macros
 * @{
 */

/// \def ACLSHMEM_MAX_PES
/// \brief Maximum number of Processing Elements (PE) supported by ACLSHMEM
#define ACLSHMEM_MAX_PES 16384

/// \def ACLSHMEM_MAX_TEAMS
/// \brief Maximum number of teams supported by ACLSHMEM
#define ACLSHMEM_MAX_TEAMS 2048

/// \def ACLSHMEM_MAX_LOCAL_SIZE
/// \brief Maximum capacity of ACLSHMEM local memory (40GB)
#define ACLSHMEM_MAX_LOCAL_SIZE (40UL * 1024 * 1024 * 1024)

/// \def TEAM_CONFIG_PADDING
/// \brief Padding bytes for the team configuration structure, used for memory alignment
#define TEAM_CONFIG_PADDING 48

/* arch related */
/// \def SCALAR_DATA_CACHELINE_SIZE
/// \brief Size of the scalar data cache line (64 bytes), used for cache alignment optimization
#define SCALAR_DATA_CACHELINE_SIZE 64

/// \def L2_CACHELINE_SIZE
/// \brief Size of the L2 cache line (512 bytes), used for advanced cache alignment optimization
#define L2_CACHELINE_SIZE 512

/// \def ACLSHMEM_PAGE_SIZE
/// \brief ACLSHMEM memory page size (2MB), complying with system memory paging specifications
#define ACLSHMEM_PAGE_SIZE (1024UL * 1024UL * 2)

/// \def ALIGH_TO
/// \brief Memory address/size alignment macro
/// \param size Original size/address to be aligned
/// \param page Target alignment granularity (usually page size)
/// \details Aligns the size up to an integer multiple of the page to avoid memory fragmentation and out-of-bounds access
#define ALIGH_TO(size, page) (((size) + (page) - 1) / (page) * (page))

/* synchronization related */
/// \def ACLSHMEMI_SYNCBIT_SIZE
/// \brief Memory size of the ACLSHMEM synchronization bit, consistent with the scalar data cache line size
#define ACLSHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
/// \def SYNC_LOG_MAX_PES
/// \brief Number of log bits required for NPU-level synchronization (5 bits), calculated as: ceil(log₈(16384)) = 5
#define SYNC_LOG_MAX_PES 5

/// \def ACLSHMEM_BARRIER_TG_DISSEM_KVAL
/// \brief Task group dissemination coefficient (8) for ACLSHMEM barrier synchronization, used for synchronization algorithm optimization
#define ACLSHMEM_BARRIER_TG_DISSEM_KVAL 8

/// \def SYNC_ARRAY_SIZE
/// \brief Total size of the NPU-level synchronization array, calculated from sync bit size, log bits and dissemination coefficient
#define SYNC_ARRAY_SIZE (ACLSHMEMI_SYNCBIT_SIZE * SYNC_LOG_MAX_PES * ACLSHMEM_BARRIER_TG_DISSEM_KVAL)

/// \def SYNC_COUNTER_SIZE
/// \brief Memory size of the NPU-level synchronization counter, consistent with the sync bit size
#define SYNC_COUNTER_SIZE ACLSHMEMI_SYNCBIT_SIZE

/// \def SYNC_POOL_SIZE
/// \brief Total size of the NPU-level synchronization pool, expanding the sync array by the maximum number of teams
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * ACLSHMEM_MAX_TEAMS)

/// \def SYNC_COUNTERS_SIZE
/// \brief Total size of the NPU-level synchronization counter pool, expanding the counter by the maximum number of teams
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * ACLSHMEM_MAX_TEAMS)

// core level sync
/// \def ACLSHMEM_MAX_AIV_PER_NPU
/// \brief Maximum number of AIV (AI Core) supported per NPU (48)
#define ACLSHMEM_MAX_AIV_PER_NPU 48

/// \def ACLSHMEM_LOG_MAX_AIV_PER_NPU
/// \brief Number of log bits required for core-level synchronization (6 bits), calculated as: ceil(log₂(48)) = 6
#define ACLSHMEM_LOG_MAX_AIV_PER_NPU 6

/// \def ACLSHMEM_CORE_SYNC_POOL_SIZE
/// \brief Total size of the core-level synchronization pool, calculated from AIV count, log bits and sync bit size
#define ACLSHMEM_CORE_SYNC_POOL_SIZE (ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_LOG_MAX_AIV_PER_NPU * ACLSHMEMI_SYNCBIT_SIZE)

/// \def ACLSHMEM_CORE_SYNC_COUNTER_SIZE
/// \brief Memory size of the core-level synchronization counter, consistent with the sync bit size
#define ACLSHMEM_CORE_SYNC_COUNTER_SIZE ACLSHMEMI_SYNCBIT_SIZE

// Total extra
/// \def ACLSHMEM_EXTRA_SIZE_UNALIGHED
/// \brief Unaligned size of the ACLSHMEM extra memory (only includes NPU sync pool)
#define ACLSHMEM_EXTRA_SIZE_UNALIGHED SYNC_POOL_SIZE

/// \def ACLSHMEM_EXTRA_SIZE
/// \brief Aligned size of the ACLSHMEM extra memory, aligned to the memory page size
#define ACLSHMEM_EXTRA_SIZE ALIGH_TO(ACLSHMEM_EXTRA_SIZE_UNALIGHED, ACLSHMEM_PAGE_SIZE)

/**@} */ // end of group_macros

/**
 * @addtogroup group_constants
 * @{
 */
// global_state
/// \var SVM_START_ADDR
/// \brief Start address of the Shared Virtual Memory (SVM) (0x100000000000)
constexpr uint64_t SVM_START_ADDR = 0x100000000000ULL;

/// \var SVM_END_ADDR
/// \brief End address of the Shared Virtual Memory (SVM), calculated as: start address + 0x80000000000 - 1GB
constexpr uint64_t SVM_END_ADDR = 0x100000000000ULL + 0x80000000000ULL - (1UL << 30UL); // svm end

/// \var GLOBAL_STATE_SIZE
/// \brief Fixed memory size of the global state structure (4MB), used to store system global configuration information
constexpr uint64_t GLOBAL_STATE_SIZE = 4UL * 1024UL * 1024UL; // global_state fixed length
/**@} */
// synchronization
/**
 * @addtogroup group_typedef
 * @{
 */
/// \brief ACLSHMEM synchronization bit array type, cache line-aligned storage structure for synchronization flags
typedef int32_t aclshmemi_sync_bit[ACLSHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];
/**@} */ // end of group_typedef
/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct aclshmem_handle_t
 * @brief Handle information used for non-blocking API synchronization.
 *
 * - aclshmem_team_t team_id: Team ID used for synchronization.
*/
struct aclshmem_handle_t {
    aclshmem_team_t team_id;
};

/**@} */ // end of group_structs

// devmm constants reference
constexpr uint64_t ACLSHMEM_HEAP_ALIGNMENT_SIZE = (1UL << 30UL);     // same definition with DEVMM_HEAP_SIZE
#ifndef ALIGN_UP
#define ALIGN_UP(val, al) (((val) + ((al) - 1)) & ~((al) - 1))
#endif

/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct aclshmem_team_config_t
 * @brief Group management configuration structure, used to store core configuration information of the team
 */
typedef struct {
    int32_t version;                      /// Version number
    int32_t num_contexts;                 /// Reserved field
    aclshmemx_team_uniqueid_t uniqueid;   /// Unique ID within the group
    char padding[TEAM_CONFIG_PADDING];    /// Used to pad the structure to 64 bytes
} aclshmem_team_config_t;
static_assert(sizeof(aclshmem_team_config_t) == 64, "aclshmem_team_config_t must be 64 bytes.");

// Team
/**
 * @struct aclshmemx_team_t
 * @brief Core data structure of a team, storing the topology and mapping relationship of the team
 * @details Contains the local view of PEs in the team, global view mapping, as well as team configuration and PE mapping table
 */
typedef struct {
    int mype;           ///< Local number of the current PE within the team, range [0, size)
    int start;          ///< Global number of the starting PE of the team, range [0, npes)
    int stride;         ///< Global stride of PEs in the team, range [1, npes-1]
    int size;           ///< Total number of PEs in the team (team size)
    int team_idx;       ///< Global index number of the team
    aclshmem_team_config_t config;  ///< Configuration information structure of the team
    int32_t pe_mapping[2 * ACLSHMEM_MAX_PES];  ///< PE mapping table, storing the correspondence between team and global PEs
} aclshmemx_team_t;

// mte_config
/**
 * @struct aclshmem_mte_config_t
 * @brief Memory Transfer Engine (MTE) configuration structure
 * @details Stores buffer address, size and synchronization event ID related to memory transfer
 */
typedef struct {
    int64_t aclshmem_ub;        ///< __ubuf__ buffer pointer, used as source/destination address for Shmem memory copy
    uint32_t ub_size;        ///< Size of the UB buffer in bytes (Bytes)
    uint32_t event_id;       ///< TEventID for memory copy synchronization, used for asynchronous transfer event synchronization
} aclshmem_mte_config_t;

// state
/**
 * @struct aclshmem_device_host_state_t
 * @brief Global state structure of the ACLSHMEM device and host
 * @details Stores core global data such as system initialization state, memory heap information, team pool, sync pool, etc., which is the core state carrier of ACLSHMEM
 */
typedef struct {
    int version;                  ///< Version number of the state structure
    int mype;                     ///< Global number of the local PE
    int npes;                     ///< Total number of PEs in the system
    void *heap_base;              ///< Start address of the local memory heap
    void *host_heap_base;         ///< heap_base for host side memory type

    // Store All Devices' heap_base.
    void **p2p_device_heap_base;  ///< Array of P2P memory heap base addresses for all devices
    void **rdma_device_heap_base; ///< Array of RDMA memory heap base addresses for all devices
    void **sdma_device_heap_base; ///< Array of SDMA memory heap base addresses for all devices

    // Store All Host' heap_base.
    void **p2p_host_heap_base;    ///< Array of P2P memory heap base addresses for all hosts
    void **rdma_host_heap_base;   ///< Array of RDMA memory heap base addresses for all hosts
    void **sdma_host_heap_base;   ///< Array of SDMA memory heap base addresses for all hosts

    uint8_t topo_list[ACLSHMEM_MAX_PES]; ///< PE topology list, storing the topological relationship of PEs in the system
    size_t heap_size;                   ///< Total size of the local memory heap in bytes

    aclshmemx_team_t *team_pools[ACLSHMEM_MAX_TEAMS]; ///< Team pool array, storing all created team instances

    // Using aclshmemi_sync_bit instead of basic types to aclshmemi_store flag,
    // avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    // These members are 'shmemi_sync_bit *' types actually, but are defined as 'uint64_t' due to compiler restriction.
    uint64_t sync_pool;          ///< NPU-level sync pool pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t sync_counter;       ///< NPU-level sync counter pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t core_sync_pool;     ///< Core-level sync pool pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t core_sync_counter;  ///< Core-level sync counter pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)

    bool is_aclshmem_initialized; ///< Flag indicating whether ACLSHMEM has completed initialization
    bool is_aclshmem_created;     ///< Flag indicating whether ACLSHMEM has been created

    aclshmem_mte_config_t mte_config; ///< Configuration information of the MTE memory transfer engine
    uint64_t qp_info;                 ///< Queue Pair (QP) information, used for communication mechanisms such as RDMA
} aclshmem_device_host_state_t;

// host only state
/**
 * @struct aclshmem_host_state_t
 * @brief Host-side exclusive state structure
 * @details Stores configuration information such as streams, events, and block counts used only by the host side
 */
typedef struct {
    // typedef void *aclrtStream; as in https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/appdevgapi/aclcppdevg_03_1355.html
    void *default_stream;       ///< Default ACL runtime stream (aclrtStream type) on the host, used for asynchronous task scheduling
    int8_t default_event_id;    ///< Default event ID on the host, used for stream synchronization
    uint32_t default_block_num; ///< Default block count on the host, used for memory/task block management
} aclshmem_host_state_t;

/**@} */ // end of group_structs

#ifdef __cplusplus
}
#endif

#endif /* ACLSHMEM_TYPES_H */