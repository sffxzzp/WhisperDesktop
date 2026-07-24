#pragma once
#include <stdint.h>
#include <stddef.h>

// GGML block-quantization support for loading quantized Whisper models
// (e.g. large-v3-turbo-q5_0 / q8_0 from https://huggingface.co/ggerganov/whisper.cpp).
//
// Reference: ref/whisper.cpp/ggml/src/ggml-common.h (block layouts) and
//            ref/whisper.cpp/ggml/src/ggml-quants.c (dequantization).

namespace Whisper
{
	// A subset of ggml_type, the per-tensor "ttype" value stored in the model file.
	enum struct eGgmlType : int
	{
		f32 = 0,
		f16 = 1,
		q4_0 = 2,
		q4_1 = 3,
		// 4 (q4_2) and 5 (q4_3) were removed from ggml long ago, never produced for whisper.
		q5_0 = 6,
		q5_1 = 7,
		q8_0 = 8,
		// K-quants: 256-element super-blocks with sub-block scales. Loaded by
		// dequantizing to fp16 on the CPU (no native GPU shader).
		q2_K = 10,
		q3_K = 11,
		q4_K = 12,
		q5_K = 13,
		q6_K = 14,
		q8_K = 15,	// intermediate type, not normally stored as a weight
	};

	// Elements per quantization block (QK). 1 for the non-blocked f16/f32.
	uint32_t ggmlBlockSize( eGgmlType t );
	// Bytes per block (ggml_type_size). f16=2, f32=4, q5_0=22, q8_0=34, ...
	uint32_t ggmlTypeSize( eGgmlType t );
	// True for the block-quantized types (everything except f16/f32).
	bool ggmlIsQuantized( eGgmlType t );
	// True for the types we compute natively on the GPU (dequantize inline in the matmul shaders).
	// Other quantized types are dequantized to f16 at load time instead.
	bool ggmlHasNativeGpuQuant( eGgmlType t );

	// Byte size of a whole tensor of `nElements` elements stored as type `t`.
	// = (nElements / blockSize) * typeSize. Requires nElements % blockSize == 0.
	size_t ggmlTensorBytes( eGgmlType t, size_t nElements );

	// Dequantize `nElements` values from the ggml on-disk representation `src`
	// (blocks laid out exactly as in the .bin file) into `dst` (fp32).
	// Handles f16/f32 (straight copy/convert) and all quantized types.
	void dequantizeToFp32( eGgmlType t, const void* src, float* dst, size_t nElements );

	// Same, but writes fp16 (uint16) output. Used to dequantize tensors we don't
	// compute natively (e.g. token_embedding, or q4_* types) into ordinary f16 GPU tensors.
	void dequantizeToFp16( eGgmlType t, const void* src, uint16_t* dst, size_t nElements );

	// ---- GPU native-quant storage (Structure-of-Arrays inside one R32_UINT buffer) ----
	//
	// A quantized weight is uploaded to VRAM as a single Buffer<uint> whose bytes are
	// split into 4-byte-aligned planar sections, so the matmul shaders can read them
	// with plain typed loads and dequantize inline. Layout per type:
	//   q8_0:  [ scales : fp16 x nBlocks ][ qs : int8  x nElements ]
	//   q5_0:  [ scales : fp16 x nBlocks ][ qh : u32 x nBlocks ][ nibbles : u8 x 16*nBlocks ]
	//   q4_0:  [ scales : fp16 x nBlocks ][ nibbles : u8 x 16*nBlocks ]
	//   q4_1:  [ scales : fp16 x nBlocks ][ mins : fp16 x nBlocks ][ nibbles : u8 x 16*nBlocks ]
	//   q5_1:  [ scales : fp16 x nBlocks ][ mins : fp16 x nBlocks ][ qh : u32 x nBlocks ][ nibbles : u8 x 16*nBlocks ]
	// Section offsets below are expressed in 32-bit words (the shader reads Buffer<uint>).
	struct QuantGpuLayout
	{
		uint32_t nBlocks = 0;
		uint32_t scalesWord = 0;   // word offset of the fp16 scales section (always 0)
		uint32_t minsWord = 0;     // word offset of the fp16 mins section (q4_1 / q5_1; else 0)
		uint32_t highWord = 0;     // word offset of the qh section (q5_* only; else 0)
		uint32_t quantsWord = 0;   // word offset of the qs / nibbles section
		uint32_t totalBytes = 0;   // total buffer size, multiple of 4
	};

	// Compute the SoA layout for a native-GPU-quant tensor. Only valid when
	// ggmlHasNativeGpuQuant( t ) is true (q5_0 / q8_0).
	QuantGpuLayout computeQuantGpuLayout( eGgmlType t, size_t nElements );

	// Repack ggml on-disk blocks `src` into the SoA byte layout described by `layout`,
	// writing `layout.totalBytes` bytes into `dst`.
	void repackQuantToGpu( eGgmlType t, const void* src, size_t nElements, void* dst );
}
