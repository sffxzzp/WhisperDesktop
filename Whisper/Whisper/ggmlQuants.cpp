#include "stdafx.h"
#include "ggmlQuants.h"
#include <immintrin.h>
#include <string.h>
#include <assert.h>
#include <algorithm>

namespace
{
	constexpr uint32_t QK = 32;	// block size for all q4_*/q5_*/q8_* types

	// F16C hardware conversion of a single half to float. The project requires F16C.
	inline float halfToFloat( uint16_t h )
	{
		return _mm_cvtss_f32( _mm_cvtph_ps( _mm_cvtsi32_si128( (int)h ) ) );
	}
	inline uint16_t floatToHalf( float f )
	{
		return (uint16_t)_mm_extract_epi16( _mm_cvtps_ph( _mm_set_ss( f ), _MM_FROUND_TO_NEAREST_INT ), 0 );
	}

	// ---------------- K-quant super-block dequant (256 elements) ----------------
	// Ported from ggml-quants.c (llama.cpp). Each function dequantizes ONE super-block
	// of 256 fp32 values. `d`/`dmin`/`m` are fp16.

	inline float f16( const uint8_t* p )
	{
		uint16_t h; memcpy( &h, p, 2 );
		return _mm_cvtss_f32( _mm_cvtph_ps( _mm_cvtsi32_si128( (int)h ) ) );
	}

	// Unpack one of 8 (6-bit scale, 6-bit min) pairs from a 12-byte block (q4_K / q5_K).
	inline void get_scale_min_k4( int j, const uint8_t* q, uint8_t& d, uint8_t& m )
	{
		if( j < 4 )
		{
			d = q[ j ] & 63;
			m = q[ j + 4 ] & 63;
		}
		else
		{
			d = ( q[ j + 4 ] & 0xF ) | ( ( q[ j - 4 ] >> 6 ) << 4 );
			m = ( q[ j + 4 ] >> 4 ) | ( ( q[ j ] >> 6 ) << 4 );
		}
	}

	void dequant_q2_K( const uint8_t* src, float* y )	// 84 bytes
	{
		const float d = f16( src + 80 );
		const float dmin = f16( src + 82 );
		const uint8_t* sc = src;			// scales[16]
		const uint8_t* q = src + 16;		// qs[64]
		int is = 0;
		for( int n = 0; n < 256; n += 128, q += 32 )
		{
			int shift = 0;
			for( int j = 0; j < 4; j++, shift += 2 )
			{
				uint8_t s = sc[ is++ ];
				float dl = d * ( s & 0xF ), ml = dmin * ( s >> 4 );
				for( int l = 0; l < 16; l++ )
					*y++ = dl * (float)( ( q[ l ] >> shift ) & 3 ) - ml;
				s = sc[ is++ ];
				dl = d * ( s & 0xF ); ml = dmin * ( s >> 4 );
				for( int l = 0; l < 16; l++ )
					*y++ = dl * (float)( ( q[ l + 16 ] >> shift ) & 3 ) - ml;
			}
		}
	}

	void dequant_q3_K( const uint8_t* src, float* y )	// 110 bytes
	{
		const float d = f16( src + 108 );
		const uint8_t* hm = src;			// hmask[32]
		const uint8_t* q = src + 32;		// qs[64]

		uint32_t aux[ 4 ];
		memcpy( aux, src + 96, 12 );		// scales[12]
		const uint32_t tmp = aux[ 2 ];
		aux[ 2 ] = ( ( aux[ 0 ] >> 4 ) & 0x0f0f0f0f ) | ( ( ( tmp >> 4 ) & 0x03030303 ) << 4 );
		aux[ 3 ] = ( ( aux[ 1 ] >> 4 ) & 0x0f0f0f0f ) | ( ( ( tmp >> 6 ) & 0x03030303 ) << 4 );
		aux[ 0 ] = ( aux[ 0 ] & 0x0f0f0f0f ) | ( ( ( tmp >> 0 ) & 0x03030303 ) << 4 );
		aux[ 1 ] = ( aux[ 1 ] & 0x0f0f0f0f ) | ( ( ( tmp >> 2 ) & 0x03030303 ) << 4 );
		const int8_t* scales = (const int8_t*)aux;	// 16 signed 6-bit (bias 32)

		int is = 0; uint8_t m = 1;
		for( int n = 0; n < 256; n += 128, q += 32 )
		{
			int shift = 0;
			for( int j = 0; j < 4; j++, shift += 2, m <<= 1 )
			{
				float dl = d * ( scales[ is++ ] - 32 );
				for( int l = 0; l < 16; l++ )
					*y++ = dl * (float)( (int)( ( q[ l ] >> shift ) & 3 ) - ( ( hm[ l ] & m ) ? 0 : 4 ) );
				dl = d * ( scales[ is++ ] - 32 );
				for( int l = 0; l < 16; l++ )
					*y++ = dl * (float)( (int)( ( q[ l + 16 ] >> shift ) & 3 ) - ( ( hm[ l + 16 ] & m ) ? 0 : 4 ) );
			}
		}
	}

	void dequant_q4_K( const uint8_t* src, float* y )	// 144 bytes
	{
		const float d = f16( src + 0 );
		const float dmin = f16( src + 2 );
		const uint8_t* scales = src + 4;	// [12]
		const uint8_t* q = src + 16;		// qs[128]
		int is = 0;
		for( int j = 0; j < 256; j += 64, q += 32, is += 2 )
		{
			uint8_t sc, m;
			get_scale_min_k4( is + 0, scales, sc, m );
			const float d1 = d * sc, m1 = dmin * m;
			get_scale_min_k4( is + 1, scales, sc, m );
			const float d2 = d * sc, m2 = dmin * m;
			for( int l = 0; l < 32; l++ )
				*y++ = d1 * (float)( q[ l ] & 0xF ) - m1;
			for( int l = 0; l < 32; l++ )
				*y++ = d2 * (float)( q[ l ] >> 4 ) - m2;
		}
	}

	void dequant_q5_K( const uint8_t* src, float* y )	// 176 bytes
	{
		const float d = f16( src + 0 );
		const float dmin = f16( src + 2 );
		const uint8_t* scales = src + 4;	// [12]
		const uint8_t* qh = src + 16;		// [32]
		const uint8_t* ql = src + 48;		// [128]
		int is = 0; uint8_t u1 = 1, u2 = 2;
		for( int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2 )
		{
			uint8_t sc, m;
			get_scale_min_k4( is + 0, scales, sc, m );
			const float d1 = d * sc, m1 = dmin * m;
			get_scale_min_k4( is + 1, scales, sc, m );
			const float d2 = d * sc, m2 = dmin * m;
			for( int l = 0; l < 32; l++ )
				*y++ = d1 * (float)( ( ql[ l ] & 0xF ) + ( ( qh[ l ] & u1 ) ? 16 : 0 ) ) - m1;
			for( int l = 0; l < 32; l++ )
				*y++ = d2 * (float)( ( ql[ l ] >> 4 ) + ( ( qh[ l ] & u2 ) ? 16 : 0 ) ) - m2;
		}
	}

	void dequant_q6_K( const uint8_t* src, float* y )	// 210 bytes
	{
		const float d = f16( src + 208 );
		const uint8_t* ql = src;			// [128]
		const uint8_t* qh = src + 128;		// [64]
		const int8_t* sc = (const int8_t*)( src + 192 );	// scales[16], signed 8-bit
		for( int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8, y += 128 )
		{
			for( int l = 0; l < 32; l++ )
			{
				const int is = l / 16;
				const int q1 = (int)( (int8_t)( ( ( ql[ l + 0 ] & 0xF ) | ( ( ( qh[ l ] >> 0 ) & 3 ) << 4 ) ) ) ) - 32;
				const int q2 = (int)( (int8_t)( ( ( ql[ l + 32 ] & 0xF ) | ( ( ( qh[ l ] >> 2 ) & 3 ) << 4 ) ) ) ) - 32;
				const int q3 = (int)( (int8_t)( ( ( ql[ l + 0 ] >> 4 ) | ( ( ( qh[ l ] >> 4 ) & 3 ) << 4 ) ) ) ) - 32;
				const int q4 = (int)( (int8_t)( ( ( ql[ l + 32 ] >> 4 ) | ( ( ( qh[ l ] >> 6 ) & 3 ) << 4 ) ) ) ) - 32;
				y[ l + 0 ] = d * sc[ is + 0 ] * q1;
				y[ l + 32 ] = d * sc[ is + 2 ] * q2;
				y[ l + 64 ] = d * sc[ is + 4 ] * q3;
				y[ l + 96 ] = d * sc[ is + 6 ] * q4;
			}
		}
	}
}

namespace Whisper
{
	uint32_t ggmlBlockSize( eGgmlType t )
	{
		switch( t )
		{
		case eGgmlType::f32:
		case eGgmlType::f16:
			return 1;
		case eGgmlType::q2_K:
		case eGgmlType::q3_K:
		case eGgmlType::q4_K:
		case eGgmlType::q5_K:
		case eGgmlType::q6_K:
		case eGgmlType::q8_K:
			return 256;	// QK_K
		default:
			return QK;	// 32
		}
	}

	uint32_t ggmlTypeSize( eGgmlType t )
	{
		switch( t )
		{
		case eGgmlType::f32: return 4;
		case eGgmlType::f16: return 2;
		case eGgmlType::q4_0: return 2 + QK / 2;			// 18
		case eGgmlType::q4_1: return 2 + 2 + QK / 2;			// 20
		case eGgmlType::q5_0: return 2 + 4 + QK / 2;			// 22
		case eGgmlType::q5_1: return 2 + 2 + 4 + QK / 2;		// 24
		case eGgmlType::q8_0: return 2 + QK;				// 34
		case eGgmlType::q2_K: return 84;
		case eGgmlType::q3_K: return 110;
		case eGgmlType::q4_K: return 144;
		case eGgmlType::q5_K: return 176;
		case eGgmlType::q6_K: return 210;
		case eGgmlType::q8_K: return 292;
		default: return 0;
		}
	}

	bool ggmlIsQuantized( eGgmlType t )
	{
		return t != eGgmlType::f32 && t != eGgmlType::f16;
	}

	bool ggmlHasNativeGpuQuant( eGgmlType t )
	{
		// The simple per-32-block legacy types are computed natively on the GPU.
		// K-quants (256-element super-blocks) are dequantized to fp16 at load time.
		switch( t )
		{
		case eGgmlType::q4_0:
		case eGgmlType::q4_1:
		case eGgmlType::q5_0:
		case eGgmlType::q5_1:
		case eGgmlType::q8_0:
			return true;
		default:
			return false;
		}
	}

	size_t ggmlTensorBytes( eGgmlType t, size_t nElements )
	{
		const uint32_t bs = ggmlBlockSize( t );
		assert( 0 == ( nElements % bs ) );
		return ( nElements / bs ) * ggmlTypeSize( t );
	}

	// ---------------- CPU dequantization ----------------

	void dequantizeToFp32( eGgmlType t, const void* src, float* y, size_t k )
	{
		if( t == eGgmlType::f32 )
		{
			memcpy( y, src, k * 4 );
			return;
		}
		if( t == eGgmlType::f16 )
		{
			const uint16_t* h = (const uint16_t*)src;
			size_t i = 0;
			for( ; i + 8 <= k; i += 8 )
			{
				__m128i hv = _mm_loadu_si128( (const __m128i*)( h + i ) );
				_mm256_storeu_ps( y + i, _mm256_cvtph_ps( hv ) );
			}
			for( ; i < k; i++ )
				y[ i ] = halfToFloat( h[ i ] );
			return;
		}

		// K-quants: 256-element super-blocks, dispatched to the per-type routines above.
		if( ggmlBlockSize( t ) == 256 )
		{
			assert( 0 == ( k % 256 ) );
			const size_t nsb = k / 256;
			const uint8_t* rsi = (const uint8_t*)src;
			const uint32_t sbBytes = ggmlTypeSize( t );
			for( size_t i = 0; i < nsb; i++, rsi += sbBytes, y += 256 )
			{
				switch( t )
				{
				case eGgmlType::q2_K: dequant_q2_K( rsi, y ); break;
				case eGgmlType::q3_K: dequant_q3_K( rsi, y ); break;
				case eGgmlType::q4_K: dequant_q4_K( rsi, y ); break;
				case eGgmlType::q5_K: dequant_q5_K( rsi, y ); break;
				case eGgmlType::q6_K: dequant_q6_K( rsi, y ); break;
				default: assert( false ); break;	// q8_K (intermediate) not supported
				}
			}
			return;
		}

		assert( 0 == ( k % QK ) );
		const size_t nb = k / QK;
		const uint8_t* rsi = (const uint8_t*)src;
		const uint32_t bytesPerBlock = ggmlTypeSize( t );

		for( size_t i = 0; i < nb; i++, rsi += bytesPerBlock )
		{
			float* const yy = y + i * QK;
			uint16_t dh;
			memcpy( &dh, rsi, 2 );
			const float d = halfToFloat( dh );

			switch( t )
			{
			case eGgmlType::q8_0:
			{
				const int8_t* qs = (const int8_t*)( rsi + 2 );
				for( int j = 0; j < (int)QK; j++ )
					yy[ j ] = qs[ j ] * d;
				break;
			}
			case eGgmlType::q4_0:
			{
				const uint8_t* qs = rsi + 2;
				for( int j = 0; j < (int)QK / 2; j++ )
				{
					const int x0 = ( qs[ j ] & 0x0F ) - 8;
					const int x1 = ( qs[ j ] >> 4 ) - 8;
					yy[ j ] = x0 * d;
					yy[ j + QK / 2 ] = x1 * d;
				}
				break;
			}
			case eGgmlType::q4_1:
			{
				uint16_t mh;
				memcpy( &mh, rsi + 2, 2 );
				const float m = halfToFloat( mh );
				const uint8_t* qs = rsi + 4;
				for( int j = 0; j < (int)QK / 2; j++ )
				{
					const int x0 = qs[ j ] & 0x0F;
					const int x1 = qs[ j ] >> 4;
					yy[ j ] = x0 * d + m;
					yy[ j + QK / 2 ] = x1 * d + m;
				}
				break;
			}
			case eGgmlType::q5_0:
			{
				uint32_t qh;
				memcpy( &qh, rsi + 2, 4 );
				const uint8_t* qs = rsi + 6;
				for( int j = 0; j < (int)QK / 2; j++ )
				{
					const uint8_t xh0 = (uint8_t)( ( ( qh >> ( j + 0 ) ) << 4 ) & 0x10 );
					const uint8_t xh1 = (uint8_t)( ( ( qh >> ( j + 12 ) ) ) & 0x10 );
					const int x0 = ( ( qs[ j ] & 0x0F ) | xh0 ) - 16;
					const int x1 = ( ( qs[ j ] >> 4 ) | xh1 ) - 16;
					yy[ j ] = x0 * d;
					yy[ j + QK / 2 ] = x1 * d;
				}
				break;
			}
			case eGgmlType::q5_1:
			{
				uint16_t mh;
				memcpy( &mh, rsi + 2, 2 );
				const float m = halfToFloat( mh );
				uint32_t qh;
				memcpy( &qh, rsi + 4, 4 );
				const uint8_t* qs = rsi + 8;
				for( int j = 0; j < (int)QK / 2; j++ )
				{
					const uint8_t xh0 = (uint8_t)( ( ( qh >> ( j + 0 ) ) << 4 ) & 0x10 );
					const uint8_t xh1 = (uint8_t)( ( ( qh >> ( j + 12 ) ) ) & 0x10 );
					const int x0 = ( qs[ j ] & 0x0F ) | xh0;
					const int x1 = ( qs[ j ] >> 4 ) | xh1;
					yy[ j ] = x0 * d + m;
					yy[ j + QK / 2 ] = x1 * d + m;
				}
				break;
			}
			default:
				assert( false );
				break;
			}
		}
	}

	void dequantizeToFp16( eGgmlType t, const void* src, uint16_t* dst, size_t nElements )
	{
		if( t == eGgmlType::f16 )
		{
			memcpy( dst, src, nElements * 2 );
			return;
		}
		// Dequantize a row at a time through a small fp32 scratch, then pack to fp16.
		constexpr size_t chunk = 4096;
		float tmp[ chunk ];
		size_t done = 0;
		while( done < nElements )
		{
			size_t n = nElements - done;
			if( n > chunk )
				n = chunk;
			// For quantized types the chunk must be a whole number of blocks.
			const uint32_t bs = ggmlBlockSize( t );
			if( bs > 1 )
				n = ( n / bs ) * bs;
			if( 0 == n )
				n = std::min( nElements - done, chunk );	// safety, shouldn't happen (ne0 % 32 == 0)

			const uint8_t* s = (const uint8_t*)src + ggmlTensorBytes( t, done );
			dequantizeToFp32( t, s, tmp, n );

			size_t i = 0;
			for( ; i + 8 <= n; i += 8 )
			{
				__m256 f = _mm256_loadu_ps( tmp + i );
				__m128i h = _mm256_cvtps_ph( f, _MM_FROUND_TO_NEAREST_INT );
				_mm_storeu_si128( (__m128i*)( dst + done + i ), h );
			}
			for( ; i < n; i++ )
				dst[ done + i ] = floatToHalf( tmp[ i ] );
			done += n;
		}
	}

	// ---------------- GPU SoA repack ----------------

	static inline uint32_t alignWords( uint32_t bytes )
	{
		return ( bytes + 3u ) / 4u;
	}

	QuantGpuLayout computeQuantGpuLayout( eGgmlType t, size_t nElements )
	{
		assert( ggmlHasNativeGpuQuant( t ) );
		assert( 0 == ( nElements % QK ) );

		QuantGpuLayout r;
		r.nBlocks = (uint32_t)( nElements / QK );
		const uint32_t sw = alignWords( r.nBlocks * 2 );	// fp16 scales, in 32-bit words
		r.scalesWord = 0;
		uint32_t cursor = sw;

		// mins plane (q4_1 / q5_1): one fp16 per block, same count as scales.
		if( t == eGgmlType::q4_1 || t == eGgmlType::q5_1 )
		{
			r.minsWord = cursor;
			cursor += sw;
		}
		// qh plane (q5_0 / q5_1): one u32 per block.
		if( t == eGgmlType::q5_0 || t == eGgmlType::q5_1 )
		{
			r.highWord = cursor;
			cursor += r.nBlocks;
		}
		// quants: q8_0 = int8 x nElements; the 4-bit types = 16 nibble bytes/block.
		r.quantsWord = cursor;
		const uint32_t qWords = ( t == eGgmlType::q8_0 )
			? alignWords( r.nBlocks * QK )
			: alignWords( r.nBlocks * ( QK / 2 ) );
		r.totalBytes = ( r.quantsWord + qWords ) * 4;
		return r;
	}

	void repackQuantToGpu( eGgmlType t, const void* src, size_t nElements, void* dstVoid )
	{
		const QuantGpuLayout layout = computeQuantGpuLayout( t, nElements );
		uint8_t* const dst = (uint8_t*)dstVoid;
		memset( dst, 0, layout.totalBytes );

		uint16_t* const scales = (uint16_t*)( dst + layout.scalesWord * 4 );
		uint16_t* const mins = (uint16_t*)( dst + layout.minsWord * 4 );
		uint32_t* const high = (uint32_t*)( dst + layout.highWord * 4 );
		uint8_t* const quants = dst + layout.quantsWord * 4;

		const uint8_t* rsi = (const uint8_t*)src;
		const uint32_t bpb = ggmlTypeSize( t );

		for( uint32_t b = 0; b < layout.nBlocks; b++, rsi += bpb )
		{
			// Every legacy block begins with the fp16 scale d.
			memcpy( &scales[ b ], rsi, 2 );
			const uint8_t* p = rsi + 2;

			if( t == eGgmlType::q4_1 || t == eGgmlType::q5_1 )
			{
				memcpy( &mins[ b ], p, 2 );	// fp16 min m
				p += 2;
			}
			if( t == eGgmlType::q5_0 || t == eGgmlType::q5_1 )
			{
				memcpy( &high[ b ], p, 4 );	// qh (u32)
				p += 4;
			}
			if( t == eGgmlType::q8_0 )
				memcpy( quants + (size_t)b * QK, p, QK );			// 32 int8
			else
				memcpy( quants + (size_t)b * ( QK / 2 ), p, QK / 2 );	// 16 nibble bytes
		}
	}
}
