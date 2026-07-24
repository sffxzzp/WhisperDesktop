#pragma once
#include "../../ComLightLib/streams.h"
#include "../API/SpecialTokens.h"
#include "../Utils/MurmurHash3.h"

namespace Whisper
{
	class Vocabulary
	{
		std::vector<const char*> tokens;
		std::vector<char> stringData;
		using THashMap = CAtlMap<const char*, int, StringPtrTraits>;
		THashMap idFromToken;

		void addExtra( int index, const char* format, int i );

		void completeBuild();

		// Shared tail of load(): derive special-token ids, label the extra tokens,
		// then completeBuild(). countWords = number of real vocab strings present.
		void finalizeSpecialTokens( int countWords, int lengthInHeader );
	public:
		Vocabulary();

		int n_vocab = 51864;

		HRESULT load( ComLight::iReadStream* stm, int lengthInHeader );

		// Populate from an explicit token list (GGUF `tokenizer.ggml.tokens`).
		// lengthInHeader = the model's vocab size (from token_embedding / metadata).
		HRESULT loadFromTokens( const std::vector<std::string>& toks, int lengthInHeader );

		using id = int;

		id token_eot = 50256;
		id token_sot = 50257;
		id token_prev = 50360;
		id token_solm = 50361; // ??
		id token_not = 50362; // no timestamps
		id token_beg = 50363;

		// available tasks
		// Not static/const: large-v3 (n_vocab 51866) inserts one extra language token,
		// which shifts these task tokens up by one. See Vocabulary::load().
		id token_translate = 50358;
		id token_transcribe = 50359;

		bool is_multilingual() const
		{
			// 51865 = large-v1/v2 (99 languages), 51866 = large-v3 (100 languages, adds Cantonese)
			return n_vocab == 51865 || n_vocab == 51866;
		}

		const char* string( int id ) const
		{
			if( id >= 0 && id < (int)tokens.size() )
				return tokens[ id ];
			return nullptr;
		}

		int findId( const char* token ) const;
		int findId( const std::string& token ) const
		{
			return findId( token.c_str() );
		}

		size_t size() const
		{
			return tokens.size();
		}

		void getSpecialTokens( SpecialTokens& rdi ) const;

		size_t getMemoryUse() const
		{
			return vectorMemoryUse( tokens ) + vectorMemoryUse( stringData );
		}

		HRESULT tokenize( const std::string& text, std::vector<id>& tokens ) const;
	};
}