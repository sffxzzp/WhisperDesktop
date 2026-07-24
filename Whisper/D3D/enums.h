#pragma once
#include <stdint.h>
#include <assert.h>

namespace DirectCompute
{
	enum struct eDataType : uint8_t
	{
		FP16,
		FP32,
		U32,
		// Block-quantized weight storage (native GPU dequant in the matmul shaders).
		// MUST be appended after the originals: some code assumes FP16==0, FP32==1,
		// and viewFormat() indexes a table by this value.
		Q5_0,
		Q8_0,
		Q4_0,
		Q4_1,
		Q5_1,
	};

	inline size_t elementSize( eDataType dt )
	{
		assert( dt == eDataType::FP16 || dt == eDataType::FP32 || dt == eDataType::U32 );

		return ( dt == eDataType::FP16 ) ? 2 : 4;
	}

	// True for the block-quantized weight types, which have no single element size.
	inline bool isQuantized( eDataType dt )
	{
		switch( dt )
		{
		case eDataType::Q5_0:
		case eDataType::Q8_0:
		case eDataType::Q4_0:
		case eDataType::Q4_1:
		case eDataType::Q5_1:
			return true;
		default:
			return false;
		}
	}

	DXGI_FORMAT viewFormat( eDataType dt );

	enum struct eBufferUse : uint8_t
	{
		// Immutable tensor, readable from GPU
		Immutable,
		// Read+write tensor, readable and writable on GPU
		ReadWrite,
		// Read+write tensor, readable and writable on GPU, which supports downloads from GPU
		ReadWriteDownload,
		// The tensor is accessible by both GPU (read only) and CPU (write only). Optimized for resources frequently updated from CPU.
		Dynamic,
	};
}