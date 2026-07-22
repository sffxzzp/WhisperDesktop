// Inline dequantization of ggml block-quantized weights for the matmul compute shaders.
//
// A quantized weight is bound as `Buffer<uint> arg0` — a raw R32_UINT view over a
// Structure-of-Arrays byte layout produced on the CPU by Whisper::repackQuantToGpu().
// Section word-offsets are derived from the total block count (nBlocks = ne0*ne1/32),
// which the shader reads from arg0Size, so no extra constants are needed.
//
//   q8_0 (QUANT_TYPE==8):  [ scales: fp16 x nBlocks ][ qs: int8 x nElements ]
//   q5_0 (QUANT_TYPE==5):  [ scales: fp16 x nBlocks ][ qh: u32 x nBlocks ][ nibbles: u8 x 16*nBlocks ]
//
// dequantElement( arg0, arg0Size, e ) returns the fp32 value of logical element `e`,
// exactly matching Whisper::dequantizeToFp32 / ggml's dequantize_row_q{5,8}_0.

#ifndef DEQUANT_HLSLI
#define DEQUANT_HLSLI

static const uint DQ_QK = 32u;

inline uint dq_numBlocks( uint4 arg0Size )
{
	const uint total = arg0Size.x * arg0Size.y * arg0Size.z * arg0Size.w;
	return total / DQ_QK;
}

// ceil( nBlocks * 2 bytes / 4 bytes ) — words occupied by the fp16 scales section
inline uint dq_scaleWords( uint nBlocks )
{
	return ( nBlocks + 1u ) / 2u;
}

// fp16 scale of block `b`, packed two per 32-bit word starting at word 0.
inline float dq_scale( Buffer<uint> w, uint b )
{
	const uint word = w[ b >> 1 ];
	const uint half16 = ( 0u != ( b & 1u ) ) ? ( word >> 16 ) : ( word & 0xFFFFu );
	return f16tof32( half16 );
}

#if QUANT_TYPE == 8

// q8_0: value = int8(qs[e]) * scale[e/32]
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint quantsWord = dq_scaleWords( nBlocks );	// qs section immediately after scales

	const uint block = e / DQ_QK;
	const float d = dq_scale( w, block );

	const uint byteIndex = quantsWord * 4u + e;			// int8 per element
	const uint word = w[ byteIndex >> 2 ];
	const uint shift = ( byteIndex & 3u ) * 8u;
	const int q = ( (int)( word << ( 24u - shift ) ) ) >> 24;	// sign-extend the byte
	return (float)q * d;
}

#elif QUANT_TYPE == 5

// q5_0: value = ( (nibble | highBit) - 16 ) * scale[e/32]
inline float dequantElement( Buffer<uint> w, uint4 arg0Size, uint e )
{
	const uint nBlocks = dq_numBlocks( arg0Size );
	const uint scaleWords = dq_scaleWords( nBlocks );
	const uint highWord = scaleWords;					// qh section (one u32 per block)
	const uint quantsWord = highWord + nBlocks;			// nibble section (16 bytes per block)

	const uint block = e / DQ_QK;
	const uint w32 = e & ( DQ_QK - 1u );				// 0..31 within block
	const float d = dq_scale( w, block );

	const uint qh = w[ highWord + block ];
	const uint highBit = ( ( qh >> w32 ) & 1u ) << 4;	// 0 or 0x10

	const uint j = w32 & 15u;							// nibble byte index
	const uint byteIndex = quantsWord * 4u + block * 16u + j;
	const uint word = w[ byteIndex >> 2 ];
	const uint nibByte = ( word >> ( ( byteIndex & 3u ) * 8u ) ) & 0xFFu;
	const uint nib = ( w32 < 16u ) ? ( nibByte & 0x0Fu ) : ( nibByte >> 4 );

	const int x = (int)( nib | highBit ) - 16;
	return (float)x * d;
}

#endif

#endif // DEQUANT_HLSLI
