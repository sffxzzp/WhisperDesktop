#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <array>
#include "../../ComLightLib/streams.h"

// Minimal parser for the GGUF container format (v2 / v3), enough to load a GGUF-packaged
// Whisper model. Reference: llama.cpp ggml/src/gguf.cpp. Little-endian only.
namespace Whisper
{
	// GGUF metadata value types (stored on disk as int32). Matches gguf_type in gguf.h.
	enum struct eGgufType : int
	{
		u8 = 0, i8 = 1, u16 = 2, i16 = 3, u32 = 4, i32 = 5, f32 = 6,
		boolean = 7, str = 8, array = 9, u64 = 10, i64 = 11, f64 = 12,
	};

	struct GgufValue
	{
		eGgufType type = eGgufType::u32;
		eGgufType elemType = eGgufType::u32;	// meaningful when type == array

		int64_t i = 0;			// integer / bool scalar
		double f = 0;			// float scalar
		std::string str;		// string scalar
		std::vector<std::string> strArray;	// array of strings
		std::vector<double> numArray;		// array of numeric values (as double)

		bool isInt() const;
		bool isFloat() const;
	};

	struct GgufTensor
	{
		std::string name;
		std::array<int64_t, 4> ne = { 1, 1, 1, 1 };
		uint32_t nDims = 0;
		int ggmlType = 0;		// ggml_type code
		uint64_t offset = 0;	// byte offset relative to the data section start
	};

	class GgufFile
	{
	public:
		static constexpr uint32_t magic = 0x46554747;	// "GGUF"

		uint32_t version = 0;
		uint32_t alignment = 32;
		int64_t dataOffset = 0;		// absolute file offset where tensor data begins
		std::vector<GgufTensor> tensors;
		std::map<std::string, GgufValue> meta;

		// Reads header + metadata + tensor-info table and resolves dataOffset.
		// The stream's magic (uint32) is assumed already consumed by the caller
		// (so the same first 4 bytes can dispatch legacy-vs-GGUF); pass alreadyReadMagic.
		HRESULT parse( ComLight::iReadStream* stm, bool magicAlreadyRead );

		// Typed metadata accessors. Return false / nullptr when the key is absent.
		const GgufValue* find( const char* key ) const;
		bool getInt( const char* key, int64_t& out ) const;
		bool getString( const char* key, std::string& out ) const;
		const std::vector<std::string>* getStringArray( const char* key ) const;

		// Find a tensor by exact name, or nullptr.
		const GgufTensor* findTensor( const char* name ) const;
	};
}
