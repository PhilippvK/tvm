# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
from types import MappingProxyType
from typing import Optional
from typing import TYPE_CHECKING, Callable, List
import traceback
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
# from tvm.contrib import graph_executor
from tvm.contrib import utils
from tvm import meta_schedule as ms
from tvm.tir import stmt_functor
from tvm.runtime import ndarray
from tvm.driver import tvmc

###
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig

###
# from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

import logging
logging.basicConfig(level=logging.ERROR)

from tvm.meta_schedule.logging import get_logger
get_logger("xgb_model").setLevel(logging.ERROR)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR / "../../../../"


MS_DISPATCH = 1  # silent?
# MS_DISPATCH = 2  # verbose
# MS_DISPATCH = ?  # error


import numpy as np
import pytest
from types import MappingProxyType
import pathlib
import json
import tvm
import tvm.testing
from tvm import relay
from tvm import transform
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor, utils
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Schedule, Trace
from tvm.tir.tensor_intrin.rocm import AMDGPU_SDOT4_INTRIN
from tvm.tir.tensor_intrin.arm_cpu import DP4A_S8S8S32_INTRIN
from tvm.tir.tensor_intrin.cfu import CFU_32X_INTRIN, CFU_24X_INTRIN, CFU_16X_INTRIN, CFU_8X_INTRIN
from tvm.tir.tensor_intrin.cfu import CFU_40X_INTRIN, CFU_48X_INTRIN, CFU_56X_INTRIN, CFU_64X_INTRIN
# from tvm.tir.tensor_intrin.arm_cpu import DP4A_S8S8S32_INIT_INTRIN
from tvm.tir.tensor_intrin.hexagon import VRMPY_i8i8i32_INTRIN
from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_NEON_INTRIN
# from tvm.tir.tensor_intrin.cfu import CFU_MAC_i8i8i32_INTRIN

from tvm.tir.schedule.analysis import has_block


def _gen_cfu_kernel_code(num_clusters: int, cfu_mode: str, channel_count: int):
    assert num_clusters in [2, 4, 16]
    assert cfu_mode in ["MODE_EMUL", "MODE_CFU"]
    assert channel_count in [8, 16, 24, 32, 40, 48, 56, 64]
    return """
#ifndef CFU_KERNEL_CODE
#define CFU_KERNEL_CODE
#include <stdint.h>

asm(".set regnum_x0  ,  0");
asm(".set regnum_x1  ,  1");
asm(".set regnum_x2  ,  2");
asm(".set regnum_x3  ,  3");
asm(".set regnum_x4  ,  4");
asm(".set regnum_x5  ,  5");
asm(".set regnum_x6  ,  6");
asm(".set regnum_x7  ,  7");
asm(".set regnum_x8  ,  8");
asm(".set regnum_x9  ,  9");
asm(".set regnum_x10 , 10");
asm(".set regnum_x11 , 11");
asm(".set regnum_x12 , 12");
asm(".set regnum_x13 , 13");
asm(".set regnum_x14 , 14");
asm(".set regnum_x15 , 15");
asm(".set regnum_x16 , 16");
asm(".set regnum_x17 , 17");
asm(".set regnum_x18 , 18");
asm(".set regnum_x19 , 19");
asm(".set regnum_x20 , 20");
asm(".set regnum_x21 , 21");
asm(".set regnum_x22 , 22");
asm(".set regnum_x23 , 23");
asm(".set regnum_x24 , 24");
asm(".set regnum_x25 , 25");
asm(".set regnum_x26 , 26");
asm(".set regnum_x27 , 27");
asm(".set regnum_x28 , 28");
asm(".set regnum_x29 , 29");
asm(".set regnum_x30 , 30");
asm(".set regnum_x31 , 31");

asm(".set regnum_zero,  0");
asm(".set regnum_ra  ,  1");
asm(".set regnum_sp  ,  2");
asm(".set regnum_gp  ,  3");
asm(".set regnum_tp  ,  4");
asm(".set regnum_t0  ,  5");
asm(".set regnum_t1  ,  6");
asm(".set regnum_t2  ,  7");
asm(".set regnum_s0  ,  8");
asm(".set regnum_s1  ,  9");
asm(".set regnum_a0  , 10");
asm(".set regnum_a1  , 11");
asm(".set regnum_a2  , 12");
asm(".set regnum_a3  , 13");
asm(".set regnum_a4  , 14");
asm(".set regnum_a5  , 15");
asm(".set regnum_a6  , 16");
asm(".set regnum_a7  , 17");
asm(".set regnum_s2  , 18");
asm(".set regnum_s3  , 19");
asm(".set regnum_s4  , 20");
asm(".set regnum_s5  , 21");
asm(".set regnum_s6  , 22");
asm(".set regnum_s7  , 23");
asm(".set regnum_s8  , 24");
asm(".set regnum_s9  , 25");
asm(".set regnum_s10 , 26");
asm(".set regnum_s11 , 27");
asm(".set regnum_t3  , 28");
asm(".set regnum_t4  , 29");
asm(".set regnum_t5  , 30");
asm(".set regnum_t6  , 31");

asm(".set CUSTOM0  , 0x0B");
asm(".set CUSTOM1  , 0x2B");

#ifdef ISSUE_582_WORKAROUND
#define CUSTOM_INSTRUCTION_NOP "nop\\n"
#else
#define CUSTOM_INSTRUCTION_NOP
#endif

#define opcode_R(opcode, func3, func7, rs1, rs2)   \\
({                                                 \\
    register unsigned long result;                 \\
    asm volatile(                                  \\
     ".word ((" #opcode ") |                       \\
     (regnum_%[result] << 7) |                     \\
     (regnum_%[arg1] << 15) |                      \\
     (regnum_%[arg2] << 20) |                      \\
     ((" #func3 ") << 12) |                        \\
     ((" #func7 ") << 25));\\n"                    \\
     CUSTOM_INSTRUCTION_NOP                        \\
     : [result] "=r" (result)                      \\
     : [arg1] "r" (rs1), [arg2] "r" (rs2)          \\
    );                                             \\
    result;                                        \\
})

// generic name for each custom instruction - via hardware
#define cfu_op_hw(funct3, funct7, rs1, rs2) \\
  opcode_R(CUSTOM0, funct3, funct7, (rs1), (rs2))
#define cfu_op0_hw(funct7, rs1, rs2) cfu_op_hw(0, funct7, rs1, rs2)
#define cfu_op1_hw(funct7, rs1, rs2) cfu_op_hw(1, funct7, rs1, rs2)
#define cfu_op2_hw(funct7, rs1, rs2) cfu_op_hw(2, funct7, rs1, rs2)
#define cfu_op3_hw(funct7, rs1, rs2) cfu_op_hw(3, funct7, rs1, rs2)
#define cfu_op4_hw(funct7, rs1, rs2) cfu_op_hw(4, funct7, rs1, rs2)
#define cfu_op5_hw(funct7, rs1, rs2) cfu_op_hw(5, funct7, rs1, rs2)
#define cfu_op6_hw(funct7, rs1, rs2) cfu_op_hw(6, funct7, rs1, rs2)
#define cfu_op7_hw(funct7, rs1, rs2) cfu_op_hw(7, funct7, rs1, rs2)

// generic name for each custom instruction - via software
#define cfu_op_sw(funct3, funct7, rs1, rs2) \\
  software_cfu(funct3, funct7, rs1, rs2)
#define cfu_op0_sw(funct7, rs1, rs2) cfu_op_sw(0, funct7, rs1, rs2)
#define cfu_op1_sw(funct7, rs1, rs2) cfu_op_sw(1, funct7, rs1, rs2)
#define cfu_op2_sw(funct7, rs1, rs2) cfu_op_sw(2, funct7, rs1, rs2)
#define cfu_op3_sw(funct7, rs1, rs2) cfu_op_sw(3, funct7, rs1, rs2)
#define cfu_op4_sw(funct7, rs1, rs2) cfu_op_sw(4, funct7, rs1, rs2)
#define cfu_op5_sw(funct7, rs1, rs2) cfu_op_sw(5, funct7, rs1, rs2)
#define cfu_op6_sw(funct7, rs1, rs2) cfu_op_sw(6, funct7, rs1, rs2)
#define cfu_op7_sw(funct7, rs1, rs2) cfu_op_sw(7, funct7, rs1, rs2)

// generic name for each custom instruction - switchable
#define cfu_op0(funct7, rs1, rs2) cfu_op(0, funct7, rs1, rs2)
#define cfu_op1(funct7, rs1, rs2) cfu_op(1, funct7, rs1, rs2)
#define cfu_op2(funct7, rs1, rs2) cfu_op(2, funct7, rs1, rs2)
#define cfu_op3(funct7, rs1, rs2) cfu_op(3, funct7, rs1, rs2)
#define cfu_op4(funct7, rs1, rs2) cfu_op(4, funct7, rs1, rs2)
#define cfu_op5(funct7, rs1, rs2) cfu_op(5, funct7, rs1, rs2)
#define cfu_op6(funct7, rs1, rs2) cfu_op(6, funct7, rs1, rs2)
#define cfu_op7(funct7, rs1, rs2) cfu_op(7, funct7, rs1, rs2)

// =============== Switch HW vs SW

#ifdef CFU_SOFTWARE_DEFINED
#define cfu_op(funct3, funct7, rs1, rs2) cfu_op_sw(funct3, funct7, rs1, rs2)
#else
#define cfu_op(funct3, funct7, rs1, rs2) cfu_op_hw(funct3, funct7, rs1, rs2)
#endif

#define MODE_CPU      1
#define MODE_EMUL     2
#define MODE_CFU      3

#define CFU_OPCODE_PUSH_WEIGHTS        0b0000000
#define CFU_OPCODE_PUSH_WEIGHTS_4B     0b0001000
#define CFU_OPCODE_SET_CODEBOOK_2B     0b0100000
#define CFU_OPCODE_SET_CODEBOOK_4B     0b0101000
#define CFU_OPCODE_SET_CODEBOOK_16B_LO 0b0111000
#define CFU_OPCODE_SET_CODEBOOK_16B_HI 0b0110000
#define CFU_OPCODE_ALU_MAC             0b1000000
#define CFU_OPCODE_ALU_RST             0b1001000

#define MODE """ + cfu_mode + """
#define NUM_CLUSTERS """ + str(num_clusters) + """
#define COUNT """ + str(channel_count) + """

#if MODE == MODE_EMUL
typedef struct {
    uint32_t word0;
    uint32_t word1;
} weights_t;

typedef struct {
    int8_t x[4];
} codebook_t;

static weights_t current_weights = {.word0 = 0, .word1 = 0};
// static codebook_t current_codebook = {.byte0 = 0, .byte1 = 0, .byte2 = 0, .byte3 = 0};
static codebook_t current_codebook = {.x = {0, 0, 0, 0}};

static int32_t acc = 0;
#endif  // MODE

#if MODE == MODE_EMUL
static void __attribute__((always_inline)) inline push_weights_4b(uint32_t word0, uint32_t word1) {
    // printf("push_weights_4b\\n");
    current_weights.word0 = word0;
    current_weights.word1 = word1;
}
#elif MODE == MODE_CFU
static int32_t __attribute__((always_inline)) inline push_weights_4b(uint32_t word0, uint32_t word1) {
    // printf("push_weights_4b\\n");
#ifdef SEAL5
    return __builtin_riscv_xcfu_cfu0_push_weights_4b(word0);
#else
    cfu_op0_hw(CFU_OPCODE_PUSH_WEIGHTS_4B, word0, word1);
#endif  // SEAL5
}
#endif  // MODE

#define push_weights_16b push_weights_4b

static int32_t __attribute__((always_inline)) inline alu_mac(uint32_t word0, uint32_t word1) {
    // printf("alu_mac\\n");
#if MODE == MODE_EMUL
    acc += current_weights.word0 * word0;
    acc += current_weights.word1 * word1;
    return acc;
#elif MODE == MODE_CFU
#ifdef SEAL5
    int32_t acc = __builtin_riscv_xcfu_cfu0_alu_mac(word0, word1);
#else
    int32_t acc = cfu_op0_hw(CFU_OPCODE_ALU_MAC, word0, word1);
#endif  // SEAL5
    return acc;
#endif  // MODE
    // TODO: if non-zero?
}

static void __attribute__((always_inline)) inline alu_rst() {
    // printf("alu_rst\\n");
#if MODE == MODE_EMUL
    acc = 0;
#elif MODE == MODE_CFU
#ifdef SEAL5
    __builtin_riscv_xcfu_cfu0_alu_rst();
#else
    cfu_op0_hw(CFU_OPCODE_ALU_RST, 0, 0);
#endif  // SEAL5
#endif  // MODE
}

static int32_t __attribute__((always_inline)) inline get_acc() {
    // printf("get_acc\\n");
#if MODE == MODE_EMUL
    return acc;
#elif MODE == MODE_CFU
#ifdef SEAL5
    return __builtin_riscv_xcfu_cfu0_alu_mac(0, 0);
#else
    return cfu_op0_hw(CFU_OPCODE_ALU_MAC, 0, 0);  // TODO: opcode for load?
#endif  // SEAL5
#endif  // MODE
}

static int32_t __attribute__((always_inline)) inline cfu_kernel_""" + str(channel_count) + """x(int8_t* data_ptr, int8_t* weights_ptr, int32_t* acc) {

    alu_rst();
#if NUM_CLUSTERS == 2
#if COUNT == 64  // TODO: hardcode or dynamic?
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, code_word1);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 56
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1_lo = *((uint16_t*)(weights_ptr + 4));
    uint8_t code_word1_hi = *((uint8_t*)(weights_ptr + 6));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1 | ((uint32_t)code_word1_hi << 16));
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 48
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1 = *((uint16_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 40
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint8_t code_word1 = *((uint8_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 32
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 24
    uint16_t code_word0_lo = *((uint16_t*)(weights_ptr));
    uint8_t code_word0_hi = *((uint8_t*)(weights_ptr + 2));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0 | ((uint32_t)code_word0_hi << 16), 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 16
    uint16_t code_word0 = *((uint16_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0, 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 8
    uint8_t code_word0 = *((uint8_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0, 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#else
// TODO: error?
#endif  // COUNT
#elif NUM_CLUSTERS == 4
#if COUNT == 32  // TODO: hardcode or dynamic?
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    // cfu_op0(CFU_FUNCT7_PUSH_WEIGHTS, code_word0, code_word1);
    push_weights_4b(code_word0, code_word1);
    for (int i = 0; i < (COUNT / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 24
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1 = *((uint16_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < (COUNT / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 16
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b(code_word0, 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 8
    uint16_t code_word0 = *((uint16_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b((uint32_t)code_word0, 0);
    for (int i = 0; i < (COUNT / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#else
// TODO: error
#endif  // COUNT
#elif NUM_CLUSTERS == 16
#if COUNT == 16  // TODO: hardcode or dynamic?
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_16b(code_word0, code_word1); // rename?
    for (int i = 0; i < (COUNT / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#elif COUNT == 8
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_16b(code_word0, 0); // rename?
    for (int i = 0; i < (COUNT / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
#else
// TODO: error?
#endif  // COUNT
#endif  // NUM_CLUSTERS
    // *acc = get_acc();
    // return 42;
    // return *acc;
    return get_acc();
}

#if NUM_CLUSTERS == 2
void set_codebook_2(int8_t* data_ptr) {
    uint16_t* codebook_lo = *((int16_t*)data_ptr);
    cfu_op0_hw(CFU_OPCODE_SET_CODEBOOK_2B, codebook_lo, 0);
}
}
#elif NUM_CLUSTERS == 4
void set_codebook_4(int8_t* data_ptr) {
    uint32_t* codebook_lo = *((int32_t*)data_ptr);
    cfu_op0_hw(CFU_OPCODE_SET_CODEBOOK_4B, codebook_lo, 0);
}
#elif NUM_CLUSTERS == 16
void set_codebook_16(int8_t* data_ptr) {
    uint32_t* codebook_lo = *((int32_t*)data_ptr);
    uint32_t* codebook_hi = *(((int32_t*)data_ptr) + 1);
    cfu_op0_hw(CFU_OPCODE_SET_CODEBOOK_4B, codebook_lo, hi);
}
#else
// TODO: err?
#endif  // NUM_CLUSTERS
#endif  // CFU_KERNEL_CODE
"""

@derived_object
class ImportCPostprocess(ms.postproc.PyPostproc):
    """A postproc that always fails."""

    def __init__(
        self,
        num_clusters: int,
        mode: str,
        channel_count: int,
        # f_initialize_with_tune_context: Callable = None,
        # f_apply: Callable = None,
        # f_clone: Callable = None,
        # f_as_string: Callable = None,
    ):
        # print("ImportCPostprocess.__init__")
        super().__init__(
            # self,
            # f_initialize_with_tune_context,
            # f_apply,
            # f_clone,
            # f_as_string,
        )
        self.num_clusters = num_clusters
        self.mode = mode
        self.channel_count = channel_count
        # print("ImportCPostprocess.__init__ done")

    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        pass

    def apply(self, sch: Schedule) -> bool:
        # print("apply", sch)
        # return False
        # has = has_block(sch, "block")
        has = has_block(sch, "root")
        # print("has", has)
        if has:
            has_tensorize = False
            is_legal = False
            try:
                # block = sch.get_block("block")
                block = sch.get_block("root")
                # print("block", block)
                sch.annotate(block, "foo", "bar")
                # code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
                # sch.annotate(block, "pragma_import_c", code)
                # print("dir(sch)", dir(sch))
                mod = sch.mod

                packed_weights_arr = None
                codebook_arr = None
                const_name = None
                pack_factor = None
                tensorize_func = None
                tensorize_block = None


                def _visit(stmt):
                    nonlocal has_tensorize, is_legal, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_block
                    if isinstance(stmt, tvm.tir.Block):  # finding blocks to be tensorized?
                        tensorize_attr = stmt.annotations.get("meta_schedule.auto_tensorize")
                        if tensorize_attr is None:
                            return
                        if not tensorize_attr.startswith("cfu_"):
                            return
                        # print("BLOCK", dir(stmt))
                        # print("stmt.name_hint", stmt.name_hint, dir(stmt.name_hint))
                        block_name = stmt.name_hint
                        # print("block_name", block_name)
                        tensorize_count = int(tensorize_attr.split("_", 1)[1][:-1])
                        # print("tensorize_count", tensorize_count)
                        assert not has_tensorize, "Can only tensorize once per block!"
                        has_tensorize = True
                        tensorize_func = tensorize_attr
                        tensorize_block = block_name
                        # print("stmt.annotations", stmt.annotations, dir(stmt.annotations))
                        # print("stmt.annotations.items()", stmt.annotations.items())
                        # print("stmt.annotations.keys()", stmt.annotations.keys())
                        # print("A", stmt.annotations.get("meta_schedule.auto_tensorize"))
                        # print("B", stmt.annotations.get("meta_schedule.auto_tensorize", None))
                        # input("!!!")
                    elif isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
                        pass
                        # print("CALL", dir(stmt))
                        # print("stmt.op", stmt.op, dir(stmt.op))
                        # print("stmt.op.name", stmt.op.name)
                        # input("!!!")
                    elif isinstance(stmt, tvm.tir.AllocateConst):  # Finding constants for weight clustering
                        # print("alloc_const")
                        # print("sch.mod", sch.mod)
                        # print("mod.attrs", mod.attrs)
                        # print("mod.functions", mod.functions)
                        # print("stmt", stmt)
                        # print("dir(stmt)", dir(stmt))
                        # print("stmt.annotations", stmt.annotations)
                        # print("stmt.body", stmt.body)
                        # print("stmt.buffer_var", stmt.buffer_var)
                        # print("dir(stmt.buffer_var)", dir(stmt.buffer_var))
                        buffer_var = stmt.buffer_var
                        name = buffer_var.name
                        # print("buffer_var.name", name)
                        # print("buffer_var.dtype", buffer_var.dtype)
                        # print("stmt.data", stmt.data)
                        #print("stmt.data.numpy()", stmt.data.numpy())
                        # print("dir(stmt.data)", dir(stmt.data))
                        data = stmt.data.numpy()
                        # print("data", data.dtype)
                        values, counts = np.unique(data, return_counts=True)
                        num_clusters = len(values)
                        if num_clusters in [2, 4, 16]:  # TODO: 3, 5-15 also fine?
                            if data.dtype == "int8":
                                dtype_bits = 8
                                # print("values", values)
                                # print("counts", counts)
                                from math import log2
                                cluster_bits = int(log2(num_clusters))
                                # print("cluster_bits", cluster_bits)
                                pack_factor = dtype_bits / cluster_bits
                                # print("pack_factor", pack_factor)
                                shape = data.shape
                                # print("shape", shape)
                                extent = shape[-1]
                                # print("extent", extent)
                                ok = extent % pack_factor == 0
                                # print("ok?", ok)
                                packed_weights = [values.tolist().index(x) for x in data.flatten()]
                                # print("packed_weights", packed_weights)
                                packed_weights = np.array(packed_weights, dtype="uint8")
                                # print("packed_weights2", packed_weights)
                                packed_weights = packed_weights.reshape(shape)
                                # print("packed_weights3", packed_weights)
                                # packed_weights = packed_weights.astype("uint8")
                                # print("packed_weights4", packed_weights, packed_weights.shape)
                                def pack_bits(arr, n_bits: int):
                                    assert arr.dtype == np.uint8, "Input array must be of dtype uint8"
                                    max_val = 2**n_bits
                                    assert np.all(arr < max_val), f"All elements must be less than {max_val}"
                                    factor = 8 // n_bits
                                    assert arr.shape[-1] % factor == 0, f"Innermost axis length must be divisible by {factor}"

                                    # Reshape to group every 4 elements along the innermost axis
                                    shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
                                    grouped = arr.reshape(shape)

                                    # TODO: little or big endian?
                                    if n_bits == 1:
                                        # Pack each group of 8 uint1s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 7) |
                                            (grouped[..., 1] << 6) |
                                            (grouped[..., 2] << 5) |
                                            (grouped[..., 3] << 4)
                                            (grouped[..., 4] << 3) |
                                            (grouped[..., 5] << 2) |
                                            (grouped[..., 6] << 1) |
                                            (grouped[..., 7])
                                        )
                                    elif n_bits == 2:
                                        # Pack each group of 4 uint2s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 6) |
                                            (grouped[..., 1] << 4) |
                                            (grouped[..., 2] << 2) |
                                            (grouped[..., 3])
                                        )
                                    elif n_bits == 4:
                                        # Pack each group of 2 uint4s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 4) |
                                            (grouped[..., 1])
                                        )
                                    packed = packed.astype(np.uint8)

                                    return packed, factor
                                packed_weights, factor = pack_bits(packed_weights, cluster_bits)
                                # print("packed_weights5", packed_weights, packed_weights.shape)
                                packed_weights_arr = packed_weights
                                codebook_arr = values
                                const_name = name
                                pack_factor = factor
                                is_legal = True
                                # print("packed_weights5.shape", packed_weights.shape)
                                # print("stmt.dtype", stmt.dtype)
                                # print("stmt.extents", stmt.extents)
                                # print("stmt.span", stmt.span)
                                # annotations', 'body', 'buffer_var', 'data', 'dtype', 'extents', 'handle', 'irmod_storage_idx', 'legacy_repr', 'same_as', 'script', 'show', 'span
                                # input("€")
                        # assert np.array_equal(stmt.data.numpy(), constants[int(stmt.irmod_storage_idx)].numpy())

                # for n, f in mod.functions.items():
                #     tvm.tir.stmt_functor.post_order_visit(f.body, _visit)
                # def _mutate(stmt):
                #     nonlocal has_tensorize, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func
                #     if not has_tensorize:
                #         return stmt
                #     if isinstance(stmt, tvm.tir.Block):  # finding blocks to be tensorized?
                #         block_name = stmt.name_hint
                #         # print("block_name", block_name)
                #         # if block_name == "root":
                #         #     # ann = stmt.annotations
                #         #     # print("stmt", dir(stmt))
                #         #     ann = {k: v for k, v in stmt.annotations.items()}
                #         #     # print("ann", ann, dir(ann))
                #         #     code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
                #         #     # sch.annotate(block, "pragma_import_c", code)
                #         #     ann["pragma_import_c"] = code
                #         #     # print("ann2", ann, dir(ann))
                #         #     # stmt.annotations = ann
                #         #     new_block = tvm.tir.Block()
                #         #     # print("stmt2", dir(stmt))
                #         #     # input("***")
                #         #     return stmt
                #     elif isinstance(stmt, tvm.tir.AllocateConst):  # Replace constant for weight clustering
                #         # print("alloc_const")
                #         buffer_var = stmt.buffer_var
                #         name = buffer_var.name
                #         if name == const_name:
                #             # TODO: change dtype?
                #             # buffer_var = tir.Var("v", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
                #             new_extents = list(stmt.extents)
                #             new_extents[-1] = new_extents[-1] // pack_factor
                #             new_data = ndarray.array(packed_weights_arr)
                #             codebook_var = tvm.tir.Var("codebook", tvm.ir.PointerType(tvm.ir.PrimType("int8")))
                #             # print("codebook_var", codebook_var, dir(codebook_var))
                #             # codebook_buf = tvm.tir.decl_buffer((len(codebook_arr),), "int8")
                #             codebook_buf = tvm.tir.decl_buffer(
                #                 shape=[len(codebook_arr)],
                #                 dtype="int8",
                #                 data=codebook_var  # Bind it to the actual var
                #             )
                #             # print("codebook_buf", codebook_buf)
                #             set_codebook_stmt = tvm.tir.Evaluate(tvm.tir.call_extern(
                #                 "void",
                #                 f"set_codebook_{self.num_clusters}",
                #                 codebook_buf.access_ptr("r", offset=0),
                #                 # codebook_var.access_ptr("r", offset=0),
                #             ))
                #             new_body = tvm.tir.SeqStmt([set_codebook_stmt, stmt.body])
                #             # print("new_body", new_body)
                #             newer_body = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype="int8", extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=new_body)
                #             # print("newer_body", newer_body)
                #             # new_body = ret = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype=tvm.tir.int8, extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=stmt.body)
                #                 # T.call_pure_extern(
                #             ret = tvm.tir.AllocateConst(buffer_var=stmt.buffer_var, dtype=stmt.dtype, extents=new_extents, data_or_idx=new_data, body=newer_body, annotations=stmt.annotations, span=stmt.span)
                #             # print("ret", ret)
                #             # input("€2")
                #             return ret
                #     return stmt

                # print("functions", mod.functions)
                # f_old = mod.functions["main"]
                f_old = mod["main"]
                # print("f_old", dir(f_old))
                # new_body = stmt_functor.ir_transform(f_old.body, _visit, _mutate, ["tir.Block", "tir.AllocateConst"])
                new_body = stmt_functor.ir_transform(f_old.body, _visit, None, ["tir.Block", "tir.AllocateConst"])
                # print("has_tensorize", has_tensorize)
                if has_tensorize:
                    if is_legal:
                        # f_new = f_old.with_body(new_body)
                        # mod["main"] = f_new
                        # print("mod", mod)
                        # block = sch.get_block("root")
                        # print("block_new", block)
                        # input("&2")
                        code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
                        # code = "dummy code"
                        # print("code", code)
                        sch.annotate(block, "pragma_import_c", code)
                    else:
                        print("illegal!")
                        sch.unannotate(tensorize_block, "meta_schedule.auto_tensorize")
                        input("#")
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                input("&&&")
                raise ex
        # print("sch", sch)
        # input(">")
        return True

    def clone(self) -> "ImportCPostprocess":
        return ImportCPostprocess(self.num_clusters, self.mode, self.channel_count)

    def __str__(self) -> str:
        return "ImportCPostprocess"


@derived_object
class ImportC2Postprocess(ms.postproc.PyPostproc):
    """A postproc that always fails."""

    def __init__(
        self,
        num_clusters: int,
        mode: str,
        channel_count: int,
        # f_initialize_with_tune_context: Callable = None,
        # f_apply: Callable = None,
        # f_clone: Callable = None,
        # f_as_string: Callable = None,
    ):
        # print("ImportCPostprocess.__init__")
        super().__init__(
            # self,
            # f_initialize_with_tune_context,
            # f_apply,
            # f_clone,
            # f_as_string,
        )
        self.num_clusters = num_clusters
        self.mode = mode
        self.channel_count = channel_count
        # print("ImportCPostprocess.__init__ done")

    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        pass

    def apply(self, sch: Schedule) -> bool:
        # print("apply", sch)
        # return False
        # has = has_block(sch, "block")
        has = has_block(sch, "root")
        # print("has", has)
        if has:
            has_call = False
            try:
                # block = sch.get_block("block")
                block = sch.get_block("root")
                # print("block", block)
                sch.annotate(block, "foo", "bar")
                code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
                sch.annotate(block, "pragma_import_c", code)
                # print("dir(sch)", dir(sch))
                mod = sch.mod

                packed_weights_arr = None
                codebook_arr = None
                const_name = None
                pack_factor = None
                tensorize_func = None


                def _visit(stmt):
                    nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func
                    if isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
                        if stmt.op.name == "tir.call_pure_extern":
                            func_name = stmt.args[0]
                            # print("func_name", func_name, dir(func_name))
                            if func_name.value.startswith("cfu_kernel"):
                                has_call = True
                                # print("mod", mod)
                                # print("CALL", dir(stmt))
                                # print("stmt.op", stmt.op, dir(stmt.op))
                                # print("stmt.op.name", stmt.op.name)
                                # print("args", stmt.args)
                                # input("!!!")
                    elif isinstance(stmt, tvm.tir.AllocateConst):  # Finding constants for weight clustering
                        # print("alloc_const")
                        # print("sch.mod", sch.mod)
                        # print("mod.attrs", mod.attrs)
                        # print("mod.functions", mod.functions)
                        # print("stmt", stmt)
                        # print("dir(stmt)", dir(stmt))
                        # print("stmt.annotations", stmt.annotations)
                        # print("stmt.body", stmt.body)
                        # print("stmt.buffer_var", stmt.buffer_var)
                        # print("dir(stmt.buffer_var)", dir(stmt.buffer_var))
                        buffer_var = stmt.buffer_var
                        name = buffer_var.name
                        # print("buffer_var.name", name)
                        # print("buffer_var.dtype", buffer_var.dtype)
                        # print("stmt.data", stmt.data)
                        #print("stmt.data.numpy()", stmt.data.numpy())
                        # print("dir(stmt.data)", dir(stmt.data))
                        data = stmt.data.numpy()
                        # print("data", data.dtype)
                        values, counts = np.unique(data, return_counts=True)
                        num_clusters = len(values)
                        if num_clusters in [2, 4, 16]:  # TODO: 3, 5-15 also fine?
                            if data.dtype == "int8":
                                dtype_bits = 8
                                # print("values", values)
                                # print("counts", counts)
                                from math import log2
                                cluster_bits = int(log2(num_clusters))
                                # print("cluster_bits", cluster_bits)
                                pack_factor = dtype_bits / cluster_bits
                                # print("pack_factor", pack_factor)
                                shape = data.shape
                                # print("shape", shape)
                                extent = shape[-1]
                                # print("extent", extent)
                                ok = extent % pack_factor == 0
                                # print("ok?", ok)
                                packed_weights = [values.tolist().index(x) for x in data.flatten()]
                                # print("packed_weights", packed_weights)
                                packed_weights = np.array(packed_weights, dtype="uint8")
                                # print("packed_weights2", packed_weights)
                                packed_weights = packed_weights.reshape(shape)
                                # print("packed_weights3", packed_weights)
                                # packed_weights = packed_weights.astype("uint8")
                                # print("packed_weights4", packed_weights, packed_weights.shape)
                                def pack_bits(arr, n_bits: int):
                                    assert arr.dtype == np.uint8, "Input array must be of dtype uint8"
                                    max_val = 2**n_bits
                                    assert np.all(arr < max_val), f"All elements must be less than {max_val}"
                                    factor = 8 // n_bits
                                    assert arr.shape[-1] % factor == 0, f"Innermost axis length must be divisible by {factor}"

                                    # Reshape to group every 4 elements along the innermost axis
                                    shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
                                    grouped = arr.reshape(shape)

                                    # TODO: little or big endian?
                                    if n_bits == 1:
                                        # Pack each group of 8 uint1s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 7) |
                                            (grouped[..., 1] << 6) |
                                            (grouped[..., 2] << 5) |
                                            (grouped[..., 3] << 4)
                                            (grouped[..., 4] << 3) |
                                            (grouped[..., 5] << 2) |
                                            (grouped[..., 6] << 1) |
                                            (grouped[..., 7])
                                        )
                                    elif n_bits == 2:
                                        # Pack each group of 4 uint2s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 6) |
                                            (grouped[..., 1] << 4) |
                                            (grouped[..., 2] << 2) |
                                            (grouped[..., 3])
                                        )
                                    elif n_bits == 4:
                                        # Pack each group of 2 uint4s into a uint8
                                        packed = (
                                            (grouped[..., 0] << 4) |
                                            (grouped[..., 1])
                                        )
                                    packed = packed.astype(np.uint8)

                                    return packed, factor
                                packed_weights, factor = pack_bits(packed_weights, cluster_bits)
                                # print("packed_weights5", packed_weights, packed_weights.shape)
                                packed_weights_arr = packed_weights
                                codebook_arr = values
                                const_name = name
                                pack_factor = factor
                                # print("packed_weights5.shape", packed_weights.shape)
                                # print("stmt.dtype", stmt.dtype)
                                # print("stmt.extents", stmt.extents)
                                # print("stmt.span", stmt.span)
                                # annotations', 'body', 'buffer_var', 'data', 'dtype', 'extents', 'handle', 'irmod_storage_idx', 'legacy_repr', 'same_as', 'script', 'show', 'span
                                # input("€")
                        # assert np.array_equal(stmt.data.numpy(), constants[int(stmt.irmod_storage_idx)].numpy())

                # for n, f in mod.functions.items():
                #     tvm.tir.stmt_functor.post_order_visit(f.body, _visit)
                def _mutate(stmt):
                    nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func
                    if not has_call:
                        return stmt
                    # if isinstance(stmt, tvm.tir.MatchBufferRegion):
                    #     print("MATCH2")
                    #     print("stmt", stmt, dir(stmt))
                    #     input("!!!5")
                    elif isinstance(stmt, tvm.tir.Evaluate):  # finding call_extern after RewriteTensorize
                        pass
                        # print("EVAL")
                        # print("stmt.op.name", stmt.op.name)
                    elif isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
                        # print("stmt.op.name", stmt.op.name)
                        # if stmt.op.name == "tir.reads":
                        #     print("READS")
                        #     print("stmt", stmt, dir(stmt))
                        #     input("!!!2")
                        # elif stmt.op.name == "tir.match_buffer":
                        #     print("MATCH")
                        #     print("stmt", stmt, dir(stmt))
                        #     input("!!!3")
                        # elif stmt.op.name == "tir.tvm_access_ptr":
                        # elif stmt.op.name == "tir.tvm_access_ptr":
                        #     print("PTR")
                        #     print("stmt", stmt, dir(stmt))
                        #     print("stmt.args[1]", stmt.args[1], dir(stmt.args[1]), type(stmt.args[1]))
                        #     print("stmt.args[2]", stmt.args[2], dir(stmt.args[2]), type(stmt.args[2]))
                        #     input("!!!4")
                        # elif stmt.op.name == "tir.call_pure_extern":
                        if stmt.op.name == "tir.call_pure_extern":
                            func_name = stmt.args[0]
                            if func_name.value.startswith("cfu_kernel"):
                                pass
                                # print("mod", mod)
                                # print("args", stmt.args)
                                # for arg in stmt.args:
                                #     print("arg", arg, dir(arg), type(arg))
                                # new_args = stmt.args
                                # print("new_args", new_args)
                                # stmt = tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)
                                # print("stmt", stmt)
                                # input("!!!1")
                    elif isinstance(stmt, tvm.tir.AllocateConst):  # Replace constant for weight clustering
                        # print("alloc_const")
                        buffer_var = stmt.buffer_var
                        name = buffer_var.name
                        if name == const_name:
                            # TODO: change dtype?
                            # buffer_var = tir.Var("v", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
                            new_extents = list(stmt.extents)
                            new_extents[-1] = new_extents[-1] // pack_factor
                            new_data = ndarray.array(packed_weights_arr)
                            codebook_var = tvm.tir.Var("codebook", tvm.ir.PointerType(tvm.ir.PrimType("int8")))
                            # print("codebook_var", codebook_var, dir(codebook_var))
                            # codebook_buf = tvm.tir.decl_buffer((len(codebook_arr),), "int8")
                            codebook_buf = tvm.tir.decl_buffer(
                                shape=[len(codebook_arr)],
                                dtype="int8",
                                data=codebook_var  # Bind it to the actual var
                            )
                            # print("codebook_buf", codebook_buf)
                            set_codebook_stmt = tvm.tir.Evaluate(tvm.tir.call_extern(
                                "void",
                                f"set_codebook_{self.num_clusters}",
                                codebook_buf.access_ptr("r", offset=0),
                                # codebook_var.access_ptr("r", offset=0),
                            ))
                            new_body = tvm.tir.SeqStmt([set_codebook_stmt, stmt.body])
                            # print("new_body", new_body)
                            newer_body = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype="int8", extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=new_body)
                            # print("newer_body", newer_body)
                            # new_body = ret = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype=tvm.tir.int8, extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=stmt.body)
                                # T.call_pure_extern(
                            ret = tvm.tir.AllocateConst(buffer_var=stmt.buffer_var, dtype=stmt.dtype, extents=new_extents, data_or_idx=new_data, body=newer_body, annotations=stmt.annotations, span=stmt.span)
                            # print("ret", ret)
                            # input("€2")
                            return ret
                    return stmt

                # print("functions", mod.functions)
                # f_old = mod.functions["main"]
                f_old = mod["main"]
                # print("f_old", dir(f_old))
                new_body = stmt_functor.ir_transform(f_old.body, _visit, _mutate, ["tir.Block", "tir.AllocateConst", "tir.Call", "tir.MatchBufferRegion", "tir.Evaluate"])
                # print("has_call", has_call)
                if has_call:
                    # mod.functions["main"] = f_new
                    f_new = f_old.with_body(new_body)
                    mod["main"] = f_new
                    # print("mod_new", mod)
                    # sch.mod = mod
                    # print("sch", sch, dir(sch))
                    # block = sch.get_block("root")
                    # print("block_new", block)
                    # input("&2")
                    # code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
                    # code = "dummy code"
                    # print("code", code)
                    # sch.annotate(block, "pragma_import_c2", code)
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                input("&&&")
                raise ex
        # print("sch", sch)
        # input(">")
        return True

    def clone(self) -> "ImportC2Postprocess":
        return ImportC2Postprocess(self.num_clusters, self.mode, self.channel_count)

    def __str__(self) -> str:
        return "ImportC2Postprocess"



# CODE = _gen_cfu_kernel_code(4, "MODE_EMUL", 32)
# print("CODE", CODE)


def get_tuning_config(enable_intrin: bool = False, num_clusters: Optional[int] = None, cfu_mode: Optional[str] = None, channel_count: Optional[int] = None):
    print("get_tuning_config", enable_intrin, num_clusters, cfu_mode, channel_count)
    if num_clusters is not None:
        assert channel_count is not None
        from math import log2
        max_channels = 64 // int(log2(num_clusters))
        channel_count = min(max_channels, channel_count)
    print("channel_count", channel_count)

    def _get_sch_rules(intrin: Optional[str] = None, num_clusters: Optional[int] = None, channel_count: Optional[int] = None):
        print("_get_sch_rules", intrin, num_clusters, channel_count)
        # init_intrin = DP4A_S8S8S32_INIT_INTRIN
        # structure_lookup = {
        #     AMDGPU_SDOT4_INTRIN: "SSSRRSRS",
        #     VRMPY_i8i8i32_INTRIN: "SRSRS",
        #     DP4A_S8S8S32_INTRIN: "SR",
        #     # DP4A_S8S8S32_INIT_INTRIN: "SR",
        #     # ARM_DOT_4x4_i8_NEON_INTRIN: "SR",
        #     ARM_DOT_4x4_i8_NEON_INTRIN: "RS",
        # }
        if intrin == "auto":

            intrin_lookup = {
                # 32: DP4A_S8S8S32_INTRIN,
                64: CFU_64X_INTRIN,
                56: CFU_56X_INTRIN,
                48: CFU_48X_INTRIN,
                40: CFU_40X_INTRIN,
                32: CFU_32X_INTRIN,
                24: CFU_24X_INTRIN,
                16: CFU_16X_INTRIN,
                8: CFU_16X_INTRIN,
            }
            intrin = intrin_lookup.get(channel_count)
            assert intrin is not None, f"Could not determine intrin for channel_count: {channel_count}"


        structure = "SR"
        print("intrin", intrin)
        return [
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.InlineConstantScalars(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            # ms.schedule_rule.AddRFactor(max_jobs_per_core=1, max_innermost_factor=64),
            *([ms.schedule_rule.MultiLevelTilingWithIntrin(
                    intrin,
                    # structure=structure_lookup[intrin],
                    structure=structure,
                    # tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                    # max_innermost_factor=32,
                    # vector_load_lens=[1, 2, 3, 4],
                    # reuse_read=ms.schedule_rule.ReuseType(
                    #     req="must",
                    #     levels=[4],
                    #     scope="shared",
                    # ),
                    # reuse_write=ms.schedule_rule.ReuseType(
                    #     req="must",
                    #     levels=[3],
                    #     scope="local",
                    # ),
                )] if intrin is not None and num_clusters is not None else []),
            # *([ms.schedule_rule.MultiLevelTilingWithIntrin(
            #         init_intrin,
            #         structure=structure_lookup[init_intrin],
            #         # tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            #         # max_innermost_factor=32,
            #         # vector_load_lens=[1, 2, 3, 4],
            #         # reuse_read=ms.schedule_rule.ReuseType(
            #         #     req="must",
            #         #     levels=[4],
            #         #     scope="shared",
            #         # ),
            #         # reuse_write=ms.schedule_rule.ReuseType(
            #         #     req="must",
            #         #     levels=[3],
            #         #     scope="local",
            #         # ),
            #     )] if init_intrin is not None else []),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
                # structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 2, 4, 8, 16, 32, 64],
                unroll_explicit=True,
                # unroll_explicit=False,
            ),
            ms.schedule_rule.RandomComputeLocation(),
        ]

    def _get_postprocs(num_clusters: Optional[int] = None, cfu_mode: Optional[str] = None, channel_count: Optional[int] = None):
        print("_get_postprocs", num_clusters, cfu_mode, channel_count)
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            *([ImportCPostprocess(num_clusters, cfu_mode, channel_count)] if enable_intrin and num_clusters is not None else []),
            ms.postproc.RewriteTensorize(),
            # *([ImportC2Postprocess(num_clusters, cfu_mode, channel_count)] if enable_intrin and num_clusters is not None else []),
            # ms.postproc.RewriteTensorize(vectorize_init_loop=True),
        ]

    def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
            # ms.mutator.Parallel(): 0.02,
        }

    # default_intrin = DP4A_S8S8S32_INTRIN
    default_intrin = "auto"
    intrin = default_intrin if enable_intrin else None
    sch_rules = _get_sch_rules(intrin, num_clusters, channel_count)
    postprocs = _get_postprocs(num_clusters, cfu_mode, channel_count)
    mutator_probs = _get_mutator_probs()
    # input(">>>")
    return sch_rules, postprocs, mutator_probs


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn


def create_relay_module():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    np.random.seed(seed=1234)
    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info
###

# def CompressWeights():
#     def _transform(func, mod, ctx):
#         print("CompressWeights")
#         print("func", func)
#         print("mod", mod)
#         print("ctx", ctx)
#         input("@A")
#         def stmt_post(stmt):
#             print("stmt_post", stmt)
#             return stmt
# 
#         new_body = tvm.tir.stmt_functor.ir_transform(
#             func.body,
#             None,
#             stmt_post,
#             ["tir.Evaluate", "tir.Call"],
#         )
#         print("new_body", new_body)
#         input("@B")
#         return func.with_body(new_body)
#     return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="CompressWeights")


@pytest.mark.parametrize("alter_op", [
    # False,
    True
])
@pytest.mark.parametrize("toolchain", [
    "gcc",
    # "llvm",
])
@pytest.mark.parametrize("target", [
    # "c -num-cores 1",
    "c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc -global-isel=1 -global-isel-abort=2 -basic-block-sections=1",
])
@pytest.mark.parametrize("num_trials_per_iter,max_trials_per_task,max_trials_global", [
    # (0, 0, 1000000),
    # (1, 1, 1000000),
    # (5, 10, 1000000),
    # (5, 20, 1000000),
    # (5, 50, 1000000),
    # (5, 100, 1000000),
    # (5, 200, 1000000),
    # (1, 200, 1000000),
    (1, 1, 1000000),
    # (5, 400, 1000000),
    # (1, 400, 1000000),
    # (5, 800, 1000000),
    # (5, 1600, 1000000),
])
@pytest.mark.parametrize("enable_custom,enable_intrin,cfu_mode", [
    # (False, False, None),
    # (True, False, None),
    # (True, True, "MODE_EMUL"),
    (True, True, "MODE_CFU"),
])
@pytest.mark.parametrize("module_equality", ["ignore-ndarray"])
@pytest.mark.parametrize("model,num_clusters,channel_count", [  # in or out?
    # ("default", None),
    # ("resnet_clustered", None),
    # ("resnet_clustered_layer0", 16, 3),  # conv2d, 3 in channels, no accel?
    # ("resnet_clustered_layer0", None, None),  # conv2d, 3 in channels, no accel?
    # ("resnet_clustered_layer1", 16, 16),  # conv2d
    # ("resnet_clustered_layer2", 16, 16),  # conv2d
    # ("resnet_clustered_layer3", None, None),  # add
    # ("resnet_clustered_layer4", 16, 16),  # conv2d
    # ("resnet_clustered_layer5", 16, 32),  # conv2d
    # ("resnet_clustered_layer6", 16, 16),  # conv2d
    # ("resnet_clustered_layer7", None, None),  # add
    ("resnet_clustered_layer8", 4, 32),  # conv2d
    # ("resnet_clustered_layer9", 4, 64),  # conv2d
    # ("resnet_clustered_layer10", 4, 32),  # conv2d
    # ("resnet_clustered_layer11", None, None),  # add
    # ("resnet_clustered_layer12", None, None),  # avg_pool
    # ("resnet_clustered_layer13", None, None),  # reshape
    # ("resnet_clustered_layer14", None, None),  # dense
    # ("resnet_clustered_layer15", None, None),  # softmax
])
@pytest.mark.parametrize("transform_layout", [
    # False,
    True,
])
@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule(alter_op, toolchain, target, num_trials_per_iter, max_trials_per_task, max_trials_global, enable_custom, enable_intrin, cfu_mode, module_equality, model, num_clusters, channel_count, transform_layout):
    print()
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.local_builder_micro import CompressWeights
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    import pathlib
    platform = DIR / "../../../../microtvm-etiss-template/template_project"
    print("platform", platform)
    options = {
        "verbose": True,
        "quiet": True,
        "gcc_prefix": str(BASE_DIR / "install/rv32gc_ilp32d"),
        "gcc_name": "riscv32-unknown-elf",
        "llvm_dir": str(BASE_DIR / "install/seal5_llvm"),
        "toolchain": toolchain,
        "etiss_script": str(BASE_DIR / "etiss/build/install/bin/run_helper.sh"),
        "etiss_args": "",
        "arch": "rv32gc_zicsr_zifencei",
        "abi": "ilp32d",
        # "cpu_arch": "RV32IMACFD",
        "cpu_arch": "RV32IMACFDXCFU0",
        "cpu_freq": 100000000,
    }
    opt_level = 3
    pass_config = {
        "tir.disable_vectorize": True,
        "tir.add_lower_pass": [(3, CompressWeights())],
    }
    disabled_pass = []
    if not alter_op:
        disabled_pass += ["AlterOpLayout"]

    KEEP = True
    if KEEP:
        base_dir = Path("/tmp/base")
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")
        def sanitize(x):
            if not isinstance(x, str):
                x = str(x)
            x = x.replace(" ", "").replace(",", "").replace("/", "").replace(";", "").replace("=", "-").replace("+", "")
            return x
        # fields = [target, toolchain, alter_op, num_trials_per_iter, max_trials_per_task, max_trials_global, ts, opt_level, enable_custom, enable_intrin, cfu_mode, module_equality, model, num_clusters, channel_count, transform_layout, list(map(lambda x: str(x)[:10], *sum(map(list, pass_config.items()), []))), *[f"no{x}" for x in disabled_pass]]
        fields = [target, toolchain, alter_op, num_trials_per_iter, max_trials_per_task, max_trials_global, ts, opt_level, enable_custom, enable_intrin, cfu_mode, module_equality, model, num_clusters, channel_count, transform_layout, *[f"no{x}" for x in disabled_pass]]
        label = "-".join([sanitize(x) for x in fields])
        work_dir_path = base_dir / label
    else:
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    print("work_dir_path", work_dir_path)
    # MODEL = "default"
    # MODEL = "resnet_clustered"
    # MODEL = "resnet_clustered_layer14"
    # MODEL = "resnet_clustered_layer10"

    # TODO: move to helper and share code!
    if model == "default":
        # input("1")
        mod, params, model_info = create_relay_module()
        input_name = model_info["in_tensor"]
        input_shape = model_info["in_shape"]
        input_dtype = model_info["in_dtype"]
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered":
        model = tvmc.load(
            str(BASE_DIR / "models/pretrainedResnet_clustered_quant_remap.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer0":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer0.tflite")
            # str(BASE_DIR / "models/layers_unpacked/pretrainedResnet_clustered_quant_remap_layer0.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer1":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer1.tflite")
            # str(BASE_DIR / "models/layers_unpacked/pretrainedResnet_clustered_quant_remap_layer1.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer2":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer2.tflite")
            # str(BASE_DIR / "models/layers_unpacked/pretrainedResnet_clustered_quant_remap_layer2.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer3":  # add
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer3.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer3.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer4":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer4.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer4.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer5":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer5.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer5.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer6":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer6.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer6.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer7":  # add
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer7.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer7.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer8":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer8.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer8.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer9":  # conv2d(?)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer9.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer9.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer10":  # conv2d(1x16x16x32, 64x1x1x32, 1x8x8x64)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer10.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer10.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer11":  # add
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer11.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer11.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer12":  # avg pool
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer12.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer12.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer13":  # rehape
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer13.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer13.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer14":  # fully connected
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer14.tflite")
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer14.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif model == "resnet_clustered_layer15":  # softmax
        model = tvmc.load(
            # str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer15.tflite")
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer15.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    else:
        assert False, f"Unsupported Model: {model}"

    if transform_layout:
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config=pass_config,
            disabled_pass=disabled_pass,
        ):
            desired_layouts = {"qnn.conv2d": ["NHWC", "HWOI"]}
            # desired_layouts = {"qnn.conv2d": ["NHWC", "OHWI"]}

            # Convert the layout of the graph where possible.
            seq = transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

    print("model.mod", model.mod)
    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    # SKIP_TUNING = True
    SKIP_TUNING = False
    builder = get_local_builder_micro()


    with ms.Profiler() as profiler:
        if not SKIP_TUNING:
            # print("a1")
            if enable_custom:
                sch_rules, postprocs, mutator_probs = get_tuning_config(enable_intrin, num_clusters, cfu_mode, channel_count)
                space = ms.space_generator.PostOrderApply(
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs=mutator_probs,
                )
                strategy = "evolutionary"
            else:
                space = "post-order-apply"
                strategy = "evolutionary"
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            with get_rpc_runner_micro(
                platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
            ) as runner:
                # print("runner", runner)
                # if True:
                if max_trials_global > 0:
                    db: ms.Database = ms.relay_integration.tune_relay(
                        mod=mod,
                        params=params,
                        target=target,
                        builder=builder,
                        runner=runner,
                        strategy=strategy,
                        space=space,
                        # num_trials_per_iter=2,
                        num_trials_per_iter=num_trials_per_iter,
                        # max_trials_per_task=10,
                        max_trials_per_task=max_trials_per_task,
                        # max_trials_global=100,
                        max_trials_global=max_trials_global,
                        work_dir=str(work_dir_path),
                        module_equality=module_equality,
                        pass_config=MappingProxyType(
                            pass_config,
                            # {
                            #     "tir.disable_vectorize": True,
                            #     # "tir.enable_debug": True,
                            # }
                        ),
                        disabled_pass=disabled_pass,
                    )
                else:
                    # db = ms.database.MemoryDatabase()
                    db = ms.database.ScheduleFnDatabase(
                        _schedule_dummy()
                    )

            #  Build model using meta_schedule logs
            ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
                database=db,
                mod=mod,
                target=target,
                params=params,
                pass_config=MappingProxyType(
                    {
                        **pass_config,
                        "relay.backend.use_meta_schedule": True,
                        "relay.backend.tir_converter": "default",
                        "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
                        # "tir.enable_debug": True,
                    }
                ),
                disabled_pass=disabled_pass,
                executor=executor,
                runtime=runtime,
            )
    print(profiler.table())
    import time
    # print("sleeping")
    # time.sleep(10)
    non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        None,
        mod=mod,
        target=target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )

    if not SKIP_TUNING:
        # TUNED
        # TODO: wrap in helper
        project = tvm.micro.generate_project(
            # str(tvm.micro.get_microtvm_template_projects(platform)),
            str(platform),
            ms_mod,
            str(work_dir_path / "project"),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            # aot_executor.get_input(0).copyfrom(data_sample)
            # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            result = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result", result)
            print("mean: ", result.mean)
            # output = aot_executor.get_output(0).numpy()

    # UNTUNED
    # TODO: wrap in helper
    project = tvm.micro.generate_project(
        # str(tvm.micro.get_microtvm_template_projects(platform)),
        str(platform),
        non_ms_mod,
        str(work_dir_path / "project2"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        # aot_executor.get_input(0).copyfrom(data_sample)
        # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
        result2 = aot_executor.module.time_evaluator("run", session.device, number=1)()
        print("result2", result2)
        print("mean2:", result2.mean)
        # import time
        # time.sleep(100)
        # output = aot_executor.get_output(0).numpy()
    if not SKIP_TUNING:
        rel = result.mean / result2.mean
        print("rel:  ", rel)

    # Build reference model (without tuning)
    # dev = tvm.cpu()
    # target = tvm.target.target.micro(model="host")
    # with tvm.transform.PassContext(
    #     opt_level=opt_level,
    #     config=pass_config,
    #     disabled_pass=disabled_pass,
    # ):
    #     ref_mod = relay.build(
    #         mod,
    #         target=target,
    #         params=params,
    #         runtime=runtime,
    #     )
    # ref_mod.export_library(work_dir / "compiled_lib2.so")
    # mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    # graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    # graph_mod.set_input(input_name, data_sample)
    # graph_mod.run()
    # ref_output = graph_mod.get_output(0).numpy()

    # assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    # if not KEEP:
    #     work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
