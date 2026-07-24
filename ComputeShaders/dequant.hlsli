// Inline dequantization of ggml block-quantized weights for the matmul compute shaders.
//
// A quantized weight is bound as `Buffer<uint> arg0` — a raw R32_UINT view over a
// Structure-of-Arrays byte layout produced on the CPU by Whisper::repackQuantToGpu().
// Section word-offsets are derived from the total block count (nBlocks = ne0*ne1/32),
// read from arg0Size, so no extra constants are needed. Per-type section order:
//
//   q8_0 (QUANT_TYPE==8):  [ scales fp16 ][ qs int8 ]
//   q5_0 (QUANT_TYPE==5):  [ scales fp16 ][ qh u32 ][ nibbles u8 ]
//   q4_0 (QUANT_TYPE==40): [ scales fp16 ][ nibbles u8 ]
//   q4_1 (QUANT_TYPE==41): [ scales fp16 ][ mins fp16 ][ nibbles u8 ]
//   q5_1 (QUANT_TYPE==51): [ scales fp16 ][ mins fp16 ][ qh u32 ][ nibbles u8 ]
//
// dequantElement( arg0, arg0Size, e ) returns the fp32 value of logical element `e`,
// matching Whisper::dequantizeToFp32 / ggml's dequantize_row_*.

#ifndef DEQUANT_HLSLI
#define DEQUANT_HLSLI

static const uint DQ_QK = 32u;

inline uint dq_numBlocks( uint4 arg0Size )
{
	const uint total = arg0Size.x * arg0Size.y * arg0Size.z * arg0Size.w;
	return total / DQ_QK;
}

// words occupied by an fp16 plane of `nBlocks` values ( = ceil(nBlocks*2 / 4) )
inline uint dq_scaleWords( uint nBlocks )
{
	return ( nBlocks + 1u ) / 2u;
}

// fp16 value #b from a plane starting at 32-bit word `wordBase` (two halves per word).
inline float dq_halfAt( Buffer<uint> w, uint wordBase, uint b )
{
	const uint word = w[ wordBase + ( b >> 1 ) ];
	const uint half16 = ( 0u != ( b & 1u ) ) ? ( word >> 16 ) : ( word & 0xFFFFu );
	return f16tof32( half16 );
}

inline float dq_scale( Buffer<uint> w, uint b ) { return dq_halfAt( w, 0u, b ); }

// 4-bit quant of element w32 (0..31) from a block's 16 nibble bytes at word `nibWord`.
// Elements 0..15 use the low nibble, 16..31 the high nibble, of nibble-byte (w32 & 15).
inline uint dq_nibble( Buffer<uint> w, uint nibWord, uint block, uint w32 )
{
	const uint byteIndex = nibWord * 4u + block * 16u + ( w32 & 15u );
	const uint word = w[ byteIndex >> 2 ];
	const uint nibByte = ( word >> ( ( byteIndex & 3u ) * 8u ) ) & 0xFFu;
	return ( w32 < 16u ) ? ( nibByte & 0x0Fu ) : ( nibByte >> 4 );
}

#if QUANT_TYPE == 8

// q8_0: value = int8(qs[e]) * scale[e/32]
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint quantsWord = dq_scaleWords( nBlocks );
	const uint block = e / DQ_QK;
	const float d = dq_scale( w, block );

	const uint byteIndex = quantsWord * 4u + e;			// int8 per element
	const uint word = w[ byteIndex >> 2 ];
	const uint shift = ( byteIndex & 3u ) * 8u;
	const int q = ( (int)( word << ( 24u - shift ) ) ) >> 24;	// sign-extend the byte
	return (float)q * d;
}

#elif QUANT_TYPE == 5

// q5_0: value = ( (nibble | highBit) - 16 ) * scale
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint highWord = dq_scaleWords( nBlocks );
	const uint quantsWord = highWord + nBlocks;
	const uint block = e / DQ_QK;
	const uint w32 = e & ( DQ_QK - 1u );
	const float d = dq_scale( w, block );

	const uint qh = w[ highWord + block ];
	const uint highBit = ( ( qh >> w32 ) & 1u ) << 4;
	const uint nib = dq_nibble( w, quantsWord, block, w32 );
	const int x = (int)( nib | highBit ) - 16;
	return (float)x * d;
}

#elif QUANT_TYPE == 40

// q4_0: value = (nibble - 8) * scale
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint quantsWord = dq_scaleWords( nBlocks );
	const uint block = e / DQ_QK;
	const uint w32 = e & ( DQ_QK - 1u );
	const float d = dq_scale( w, block );
	const uint nib = dq_nibble( w, quantsWord, block, w32 );
	return (float)( (int)nib - 8 ) * d;
}

#elif QUANT_TYPE == 41

// q4_1: value = nibble * scale + min
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint sw = dq_scaleWords( nBlocks );
	const uint minsWord = sw;
	const uint quantsWord = sw + sw;
	const uint block = e / DQ_QK;
	const uint w32 = e & ( DQ_QK - 1u );
	const float d = dq_scale( w, block );
	const float m = dq_halfAt( w, minsWord, block );
	const uint nib = dq_nibble( w, quantsWord, block, w32 );
	return (float)nib * d + m;
}

#elif QUANT_TYPE == 51

// q5_1: value = (nibble | highBit) * scale + min
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint sw = dq_scaleWords( nBlocks );
	const uint minsWord = sw;
	const uint highWord = sw + sw;
	const uint quantsWord = highWord + nBlocks;
	const uint block = e / DQ_QK;
	const uint w32 = e & ( DQ_QK - 1u );
	const float d = dq_scale( w, block );
	const float m = dq_halfAt( w, minsWord, block );
	const uint qh = w[ highWord + block ];
	const uint highBit = ( ( qh >> w32 ) & 1u ) << 4;
	const uint nib = dq_nibble( w, quantsWord, block, w32 );
	return (float)( nib | highBit ) * d + m;
}

#endif

#endif // DEQUANT_HLSLI
