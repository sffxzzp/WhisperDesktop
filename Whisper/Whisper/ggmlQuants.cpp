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
		default:
			return QK;
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
		default: return 0;
		}
	}

	bool ggmlIsQuantized( eGgmlType t )
	{
		return t != eGgmlType::f32 && t != eGgmlType::f16;
	}

	bool ggmlHasNativeGpuQuant( eGgmlType t )
	{
		return t == eGgmlType::q5_0 || t == eGgmlType::q8_0;
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
		r.scalesWord = 0;
		const uint32_t scalesWords = alignWords( r.nBlocks * 2 );	// fp16 scales

		if( t == eGgmlType::q8_0 )
		{
			r.highWord = 0;										// unused
			r.quantsWord = scalesWords;
			const uint32_t qsWords = alignWords( r.nBlocks * QK );	// int8 x nElements
			r.totalBytes = ( r.quantsWord + qsWords ) * 4;
		}
		else // q5_0
		{
			r.highWord = scalesWords;
			const uint32_t qhWords = r.nBlocks;						// one u32 per block
			r.quantsWord = r.highWord + qhWords;
			const uint32_t nibWords = alignWords( r.nBlocks * ( QK / 2 ) );	// 16 bytes/block
			r.totalBytes = ( r.quantsWord + nibWords ) * 4;
		}
		return r;
	}

	void repackQuantToGpu( eGgmlType t, const void* src, size_t nElements, void* dstVoid )
	{
		const QuantGpuLayout layout = computeQuantGpuLayout( t, nElements );
		uint8_t* const dst = (uint8_t*)dstVoid;
		memset( dst, 0, layout.totalBytes );

		uint16_t* const scales = (uint16_t*)( dst + layout.scalesWord * 4 );
		uint8_t* const quants = dst + layout.quantsWord * 4;
		uint32_t* const high = (uint32_t*)( dst + layout.highWord * 4 );

		const uint8_t* rsi = (const uint8_t*)src;
		const uint32_t bytesPerBlock = ggmlTypeSize( t );

		for( uint32_t b = 0; b < layout.nBlocks; b++, rsi += bytesPerBlock )
		{
			memcpy( &scales[ b ], rsi, 2 );	// fp16 d

			if( t == eGgmlType::q8_0 )
			{
				// int8 qs[32] -> quants[b*32 .. ]
				memcpy( quants + (size_t)b * QK, rsi + 2, QK );
			}
			else // q5_0
			{
				memcpy( &high[ b ], rsi + 2, 4 );			// qh (u32)
				memcpy( quants + (size_t)b * ( QK / 2 ), rsi + 6, QK / 2 );	// 16 nibble bytes
			}
		}
	}
}
