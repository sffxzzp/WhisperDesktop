#include "stdafx.h"
#include "enums.h"

// Indexed by (uint8_t)eDataType: FP16, FP32, U32, Q5_0, Q8_0.
// Quantized weights are stored as a raw R32_UINT buffer and dequantized in-shader.
static const alignas( 16 ) std::array<DXGI_FORMAT, 5> s_tensorViewFormats =
{
	DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT,
	DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT
};

DXGI_FORMAT DirectCompute::viewFormat( eDataType dt )
{
	return s_tensorViewFormats[ (uint8_t)dt ];
}