#include "stdafx.h"
#include "ggufFile.h"
#include "loaderUtils.h"

using ComLight::iReadStream;

namespace Whisper
{
	bool GgufValue::isInt() const
	{
		switch( type )
		{
		case eGgufType::u8: case eGgufType::i8:
		case eGgufType::u16: case eGgufType::i16:
		case eGgufType::u32: case eGgufType::i32:
		case eGgufType::u64: case eGgufType::i64:
		case eGgufType::boolean:
			return true;
		default:
			return false;
		}
	}
	bool GgufValue::isFloat() const
	{
		return type == eGgufType::f32 || type == eGgufType::f64;
	}
}

namespace
{
	using namespace Whisper;

	// Upper bounds to reject corrupt / hostile files before allocating.
	constexpr uint64_t maxCount = 1u << 28;		// array / kv / tensor counts
	constexpr uint64_t maxStringLen = 1u << 26;	// 64 MiB

	HRESULT readGgufString( iReadStream* stm, std::string& s )
	{
		uint64_t len = 0;
		CHECK( readStruct( stm, len ) );
		if( len > maxStringLen )
			return E_INVALIDARG;
		s.resize( (size_t)len );
		if( len != 0 )
			CHECK( readBytes( stm, &s[ 0 ], (size_t)len ) );
		return S_OK;
	}

	// Reads one scalar of the given gguf type; sets exactly one of iOut/fOut/sOut.
	HRESULT readScalarInto( iReadStream* stm, eGgufType t, int64_t& iOut, double& fOut, std::string& sOut )
	{
		switch( t )
		{
		case eGgufType::u8: { uint8_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::i8: { int8_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::u16: { uint16_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::i16: { int16_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::u32: { uint32_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::i32: { int32_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::u64: { uint64_t x; CHECK( readStruct( stm, x ) ); iOut = (int64_t)x; return S_OK; }
		case eGgufType::i64: { int64_t x; CHECK( readStruct( stm, x ) ); iOut = x; return S_OK; }
		case eGgufType::boolean: { uint8_t x; CHECK( readStruct( stm, x ) ); iOut = x ? 1 : 0; return S_OK; }
		case eGgufType::f32: { float x; CHECK( readStruct( stm, x ) ); fOut = x; return S_OK; }
		case eGgufType::f64: { double x; CHECK( readStruct( stm, x ) ); fOut = x; return S_OK; }
		case eGgufType::str: return readGgufString( stm, sOut );
		default:
			return E_INVALIDARG;	// nested arrays are not valid GGUF
		}
	}
}

namespace Whisper
{
	HRESULT GgufFile::parse( iReadStream* stm, bool magicAlreadyRead )
	{
		if( !magicAlreadyRead )
		{
			uint32_t m = 0;
			CHECK( readStruct( stm, m ) );
			if( m != magic )
			{
				logError( u8"Not a GGUF file (bad magic 0x%08X)", m );
				return E_INVALIDARG;
			}
		}

		CHECK( readStruct( stm, version ) );
		if( version != 2 && version != 3 )
		{
			logError( u8"Unsupported GGUF version %u (need 2 or 3)", version );
			return E_INVALIDARG;
		}

		int64_t nTensors = 0, nKv = 0;
		CHECK( readStruct( stm, nTensors ) );
		CHECK( readStruct( stm, nKv ) );
		if( nTensors < 0 || nKv < 0 || (uint64_t)nTensors > maxCount || (uint64_t)nKv > maxCount )
			return E_INVALIDARG;

		// ---- metadata KV ----
		for( int64_t k = 0; k < nKv; k++ )
		{
			std::string key;
			CHECK( readGgufString( stm, key ) );

			int32_t typeCode = 0;
			CHECK( readStruct( stm, typeCode ) );

			GgufValue v;
			v.type = (eGgufType)typeCode;
			if( v.type == eGgufType::array )
			{
				int32_t et = 0;
				CHECK( readStruct( stm, et ) );
				v.elemType = (eGgufType)et;
				uint64_t count = 0;
				CHECK( readStruct( stm, count ) );
				if( count > maxCount )
					return E_INVALIDARG;
				for( uint64_t j = 0; j < count; j++ )
				{
					int64_t iv = 0; double fv = 0; std::string sv;
					CHECK( readScalarInto( stm, v.elemType, iv, fv, sv ) );
					if( v.elemType == eGgufType::str )
						v.strArray.push_back( std::move( sv ) );
					else if( v.elemType == eGgufType::f32 || v.elemType == eGgufType::f64 )
						v.numArray.push_back( fv );
					else
						v.numArray.push_back( (double)iv );
				}
			}
			else
			{
				CHECK( readScalarInto( stm, v.type, v.i, v.f, v.str ) );
			}
			meta[ key ] = std::move( v );
		}

		// ---- tensor info ----
		tensors.resize( (size_t)nTensors );
		for( int64_t t = 0; t < nTensors; t++ )
		{
			GgufTensor& ti = tensors[ (size_t)t ];
			CHECK( readGgufString( stm, ti.name ) );
			CHECK( readStruct( stm, ti.nDims ) );
			if( ti.nDims > 4 )
				return E_INVALIDARG;
			for( uint32_t d = 0; d < ti.nDims; d++ )
				CHECK( readStruct( stm, ti.ne[ d ] ) );
			for( uint32_t d = ti.nDims; d < 4; d++ )
				ti.ne[ d ] = 1;
			int32_t tc = 0;
			CHECK( readStruct( stm, tc ) );
			ti.ggmlType = tc;
			CHECK( readStruct( stm, ti.offset ) );
		}

		// ---- alignment + data section start ----
		alignment = 32;
		const GgufValue* pa = find( "general.alignment" );
		if( nullptr != pa && pa->isInt() && pa->i > 0 )
			alignment = (uint32_t)pa->i;
		if( 0 != ( alignment & ( alignment - 1 ) ) )	// must be power of two
			return E_INVALIDARG;

		int64_t pos = 0;
		CHECK( stm->getPosition( pos ) );
		dataOffset = ( ( pos + alignment - 1 ) / alignment ) * alignment;
		return S_OK;
	}

	const GgufValue* GgufFile::find( const char* key ) const
	{
		auto it = meta.find( key );
		return ( it == meta.end() ) ? nullptr : &it->second;
	}

	bool GgufFile::getInt( const char* key, int64_t& out ) const
	{
		const GgufValue* v = find( key );
		if( nullptr == v || !v->isInt() )
			return false;
		out = v->i;
		return true;
	}

	bool GgufFile::getString( const char* key, std::string& out ) const
	{
		const GgufValue* v = find( key );
		if( nullptr == v || v->type != eGgufType::str )
			return false;
		out = v->str;
		return true;
	}

	const std::vector<std::string>* GgufFile::getStringArray( const char* key ) const
	{
		const GgufValue* v = find( key );
		if( nullptr == v || v->type != eGgufType::array || v->elemType != eGgufType::str )
			return nullptr;
		return &v->strArray;
	}

	const GgufTensor* GgufFile::findTensor( const char* name ) const
	{
		for( const GgufTensor& t : tensors )
			if( t.name == name )
				return &t;
		return nullptr;
	}
}
