/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.tc.tornado;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import uk.ac.manchester.tornado.tc.ir.AddressSpace;
import uk.ac.manchester.tornado.tc.ir.Diagnostic;
import uk.ac.manchester.tornado.tc.ir.Function;
import uk.ac.manchester.tornado.tc.ir.Inst;
import uk.ac.manchester.tornado.tc.ir.Loc;
import uk.ac.manchester.tornado.tc.ir.Opcode;
import uk.ac.manchester.tornado.tc.ir.Type;
import uk.ac.manchester.tornado.tc.ir.Value;
import uk.ac.manchester.tornado.tc.ir.analysis.LoopForest;
import uk.ac.manchester.tornado.tc.ir.target.TargetCaps;

/**
 * What the target cannot run, refused before code is generated, from {@link TargetCaps} (the
 * driver's device queries). Each check names the construct and the limit:
 * <ul>
 * <li>TC-E4012: double precision on a device without it;</li>
 * <li>TC-E4013: half precision on a device without it (not on OpenCL, whose compilers take half
 * with the pragma);</li>
 * <li>TC-E4015: warp operations, written for 32-lane warps, on another SIMD width;</li>
 * <li>TC-E4016: a device launch where kernels cannot launch kernels;</li>
 * <li>TC-E4017: tensor-core MMA or {@code cp.async} below CUDA sm_80, or MMA on an AMD GPU without matrix cores;</li>
 * <li>TC-E4018: FP8 tensor-core MMA below sm_89, or with an NVRTC older than 12.4 (PTX ISA 8.4);</li>
 * <li>TC-E4019: on a 64-lane HIP wavefront, a loop that waits for another thread of the work-group to
 * write local memory ({@link #localWait});</li>
 * <li>TC-E4010, TC-E4011: CUDA Tile kernels ({@link TileTyping#checkTarget}).</li>
 * </ul>
 */
public final class Legality {

    private Legality() {
    }

    public static List<Diagnostic> run(Function function, TargetCaps target) {
        var problems = new ArrayList<Diagnostic>();
        if (!target.fp64()) {
            first(function, t -> uses(t, Type.F64)).ifPresent(i -> problems.add(at(function, i, "TC-E4012", "this kernel uses double precision, which the device does not support")));
        }
        // OpenCL compilers accept half, scalars and vectors, with the cl_khr_fp16 pragma even where the
        // extension is not advertised (the RTX 4090's driver), which the Graal path relies on too.
        if (!target.fp16() && target.dialect() != TargetCaps.Dialect.OPENCL) {
            first(function, t -> uses(t, Type.F16)).ifPresent(i -> problems.add(at(function, i, "TC-E4013", "this kernel uses half precision, which the device does not support")));
        }
        // Local memory is not checked here: the driver is the authority, and its limit is not the one the
        // device reports (CUDA takes up to the opt-in size, 99 KB on sm_89, against 48 KB reported;
        // NVIDIA's OpenCL adds its own bytes; Metal's pipeline aligns). It refuses with the reason.
        if (!target.warps32()) {
            insts(function).filter(i -> intrinsic(i, "tornado.simd.")).findFirst()
                    .ifPresent(i -> problems.add(at(function, i, "TC-E4015", "warp operations are written for 32-lane warps; the device's SIMD width is " + target.warpSize())));
        }
        if (!target.deviceLaunch()) {
            insts(function).filter(i -> intrinsic(i, "tornado.device_launch")).findFirst()
                    .ifPresent(i -> problems.add(at(function, i, "TC-E4016", "KernelContext.launch needs kernels that launch kernels (CUDA dynamic parallelism), which this target lacks")));
        }
        if (target.dialect() == TargetCaps.Dialect.CUDA) {
            var mma = insts(function).filter(i -> intrinsic(i, "tornado.mma.") || intrinsic(i, "tornado.async_copy")).findFirst();
            var fp8 = insts(function).filter(i -> intrinsic(i, "tornado.mma.m16n8k32_e4m3") || intrinsic(i, "tornado.mma.m16n8k32_e5m2")).findFirst();
            if (!target.mma() && mma.isPresent()) {
                problems.add(at(function, mma.get(), "TC-E4017", "tensor-core MMA and cp.async need compute capability 8.0 or higher; the device is sm_" + target.computeCapability()));
            } else if (!target.fp8Mma() && fp8.isPresent()) {
                problems.add(at(function, fp8.get(), "TC-E4018", target.computeCapability() < 89
                        ? "FP8 tensor-core MMA needs compute capability 8.9 or higher; the device is sm_" + target.computeCapability()
                        : "FP8 tensor-core MMA needs PTX ISA 8.4 (NVRTC 12.4 or newer); NVRTC is " + target.nvrtcVersion() / 1000 + "." + target.nvrtcVersion() % 1000));
            }
        }
        // HIP copies asynchronously-staged tiles synchronously, which any AMD GPU can do; MMA needs matrix cores.
        if (target.dialect() == TargetCaps.Dialect.HIP && !target.mma()) {
            insts(function).filter(i -> intrinsic(i, "tornado.mma.")).findFirst()
                    .ifPresent(i -> problems.add(at(function, i, "TC-E4017", hipNoMatrixCores(target))));
        }
        if (target.dialect() == TargetCaps.Dialect.HIP && target.warpSize() == 64) {
            localWait(function).ifPresent(i -> problems.add(at(function, i, "TC-E4019",
                    "this loop waits for a local-memory value that another thread of the work-group writes; a 64-lane wavefront runs two 32-lane warps in lockstep, so the wait can hang when the writer is in the same wavefront")));
        }
        insts(function).filter(i -> intrinsic(i, "tornado.vector.lane_at") || intrinsic(i, "tornado.vector.insert_at")).findFirst()
                .ifPresent(i -> problems.add(at(function, i, "TC-E1010", "vector lanes can only be accessed at a constant index (a loop over lanes must have a constant trip count to unroll)")));
                if (TileTyping.usesTiles(function)) {
            if (target.dialect() != TargetCaps.Dialect.CUDA) {
                problems.add(new Diagnostic("TC-E4010", "TileContext kernels are CUDA Tile C++: they run only on the CUDA backend", "@" + function.name(), Loc.UNKNOWN));
            } else {
                problems.addAll(TileTyping.checkTarget(function, target.computeCapability()));
            }
        }
        return problems;
    }

    private static String hipNoMatrixCores(TargetCaps target) {
        return "tensor-core MMA needs AMD matrix cores (MFMA on CDNA, WMMA on RDNA3, RDNA4 or gfx1250); the device is " + (target.arch().isEmpty() ? "an AMD GPU without them" : target.arch());
    }

    /**
     * The first read of a loop that only waits for local memory to change: no value is carried from one
     * iteration to the next, nothing in the loop writes memory or synchronises, every exit is decided by a
     * local-memory read in the loop, and the kernel writes that local array elsewhere. Lanes of one
     * wavefront do not make independent progress, so if the writer shares the waiting thread's wavefront the
     * loop can spin before the write runs. A wait on global memory is not reported: its writer may be
     * another work-group, which does progress.
     */
    static java.util.Optional<Inst> localWait(Function function) {
        var loops = new LoopForest(function);
        for (var header : function.blocks()) {
            var loop = loops.loopWithHeader(header).orElse(null);
            if (loop == null || !header.args().isEmpty()) {
                continue;
            }
            var exits = new ArrayList<Inst>();
            var waits = true;
            for (var block : function.blocks()) {
                if (!loop.contains(block)) {
                    continue;
                }
                for (var inst : block.insts()) {
                    if (inst.isTerminator()) {
                        if (inst.successors().stream().anyMatch(s -> !loop.contains(s.target()))) {
                            exits.add(inst);
                        }
                    } else if (!inst.opcode().isPure() && !(inst.opcode() instanceof Opcode.Load) && !(inst.opcode() instanceof Opcode.ArrayLoad) && !readOnlyCas(inst)) {
                        waits = false;
                    }
                }
            }
            if (!waits || exits.isEmpty()) {
                continue;
            }
            Inst read = null;
            for (var exit : exits) {
                var r = exit.opcode() instanceof Opcode.CondBranch ? localRead(exit.operand(0), loop) : null;
                if (r == null) {
                    read = null;
                    break;
                }
                read = read == null ? r : read;
            }
            if (read != null && writtenOutside(root(read.operand(0)), loop, function)) {
                return java.util.Optional.of(read);
            }
        }
        return java.util.Optional.empty();
    }

    /** A compare-and-swap that stores the value it expects: a read of the location. */
    private static boolean readOnlyCas(Inst inst) {
        return intrinsic(inst, "tornado.atomic.cas") && inst.operandCount() == 3 && sameValue(inst.operand(1), inst.operand(2));
    }

    private static boolean sameValue(Value a, Value b) {
        if (a == b) {
            return true;
        }
        return a instanceof Inst x && b instanceof Inst y && x.opcode() instanceof Opcode.ConstInt(var u) && y.opcode() instanceof Opcode.ConstInt(var v) && u == v;
    }

    /** A local-memory read in the loop that {@code value} depends on through pure instructions, or null. */
    private static Inst localRead(Value value, LoopForest.Loop loop) {
        if (!(value instanceof Inst inst) || inst.block().map(b -> !loop.contains(b)).orElse(true)) {
            return null;
        }
        if ((inst.opcode() instanceof Opcode.Load || readOnlyCas(inst)) && inst.operand(0).type() instanceof Type.PointerType(var space) && space == AddressSpace.SHARED) {
            return inst;
        }
        if (!inst.opcode().isPure()) {
            return null;
        }
        for (var operand : inst.operands()) {
            var read = localRead(operand, loop);
            if (read != null) {
                return read;
            }
        }
        return null;
    }

    private static Value root(Value pointer) {
        while (pointer instanceof Inst inst && inst.opcode() instanceof Opcode.PointerOffset) {
            pointer = inst.operand(0);
        }
        return pointer;
    }

    /** Whether an instruction outside the loop stores to, or atomically updates, the local array {@code root}. */
    private static boolean writtenOutside(Value root, LoopForest.Loop loop, Function function) {
        for (var block : function.blocks()) {
            if (loop.contains(block)) {
                continue;
            }
            for (var inst : block.insts()) {
                if ((inst.opcode() instanceof Opcode.Store || intrinsic(inst, "tornado.atomic.")) && inst.operandCount() > 0 && root(inst.operand(0)) == root) {
                    return true;
                }
            }
        }
        return false;
    }

    private static Stream<Inst> insts(Function function) {
        return function.blocks().stream().flatMap(b -> b.insts().stream());
    }

    private static java.util.Optional<Inst> first(Function function, java.util.function.Predicate<Type> type) {
        // The first instruction producing or using such a value (parameters count where they are used).
        return insts(function).filter(i -> type.test(i.type()) || i.operands().stream().anyMatch(o -> type.test(o.type()))).findFirst();
    }

    private static boolean uses(Type type, Type scalar) {
        return switch (type) {
            case Type.VectorType(var element, var length) -> element.equals(scalar);
            case Type.ArrayType(var element) -> uses(element, scalar);
            default -> type.equals(scalar);
        };
    }

    private static boolean intrinsic(Inst inst, String prefix) {
        return inst.opcode() instanceof Opcode.Intrinsic(var name) && name.startsWith(prefix);
    }

    private static Diagnostic at(Function function, Inst inst, String code, String message) {
        return new Diagnostic(code, message, "@" + function.name(), inst.loc());
    }
}
