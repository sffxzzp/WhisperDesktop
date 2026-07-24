#include "stdafx.h"
#include "WhisperModel.h"
#include "loaderUtils.h"
#include "audioConstants.h"
#include "ggmlQuants.h"
#include "ggufFile.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include "../D3D/createBuffer.h"
#include <atlcoll.h>
#include <atlstr.h>
#include "../Utils/GpuProfilerSimple.h"
#include "../Utils/CpuProfiler.h"
#include "../CPU/HybridLoader.h"
#include "../ML/Reshaper.h"
using namespace Whisper;
using namespace DirectCompute;

namespace
{
	struct ParamsAndMelHeader
	{
		sModelParams mp;
		uint32_t n_mel = 0, n_fft = 0;
	};

	enum struct ePostProcessing : uint8_t
	{
		None = 0,
		MakePanels = 1
	};

	struct PendingTensor
	{
		DirectCompute::Tensor* dest = nullptr;
		ePostProcessing postProcessing = ePostProcessing::None;

		PendingTensor() = default;
		PendingTensor( const PendingTensor& ) = default;
		PendingTensor( DirectCompute::Tensor& tensor, ePostProcessing pp = ePostProcessing::None ) :
			dest( &tensor ), postProcessing( pp ) { }

		// If you wonder why not reshape them after all tensors are loaded, doing that on the fly is faster because CPU and GPU work in parallel
		// In the current version, CPU reads data for a next tensor, while in the meantime GPU reshapes a previously loaded tensor.
		HRESULT postProcess( Reshaper& rs, eDataType dt )
		{
			// Block-quantized weights are computed natively via the plain (non-panel) matmul
			// shaders; the panel reshape shuffles individual floats and would destroy the
			// quant blocks, so it never applies to them.
			if( isQuantized( dt ) )
				return S_OK;
			switch( postProcessing )
			{
			case ePostProcessing::None:
				return S_OK;
			case ePostProcessing::MakePanels:
				if( gpuInfo().useReshapedMatMul() )
				{
					// GpuInfo structure says we should use that new method
					return rs.makePanels( *dest, dt );
				}
				else
				{
					// The feature ain't enabled on the current user's GPU
					return S_OK;
				}
			default:
				return E_UNEXPECTED;
			}
		}
	};

	void populateEncodeTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersEnc, DirectCompute::ModelBuffers& tensors )
	{
		tensors.enc.layers.resize( layersEnc );

		CStringA tempString;
		// Encoder tensors
		auto& enc = tensors.enc;

		map[ "encoder.positional_embedding" ] = enc.positionalEmbedding;
		map[ "encoder.conv1.weight" ] = enc.conv1.w;
		map[ "encoder.conv1.bias" ] = enc.conv1.b;

		map[ "encoder.conv2.weight" ] = enc.conv2.w;
		map[ "encoder.conv2.bias" ] = enc.conv2.b;

		map[ "encoder.ln_post.weight" ] = enc.lnPost.w;
		map[ "encoder.ln_post.bias" ] = enc.lnPost.b;

		auto add = [ & ]( const char* name, int i, DirectCompute::Tensor& t, ePostProcessing pp = ePostProcessing::None )
		{
			tempString.Format( "encoder.blocks.%i.%s", i, name );
			map[ tempString ] = PendingTensor{ t, pp };
		};
		auto add2 = [ & ]( const char* name, int i, DirectCompute::TensorPair& t, ePostProcessing ppWeight = ePostProcessing::None, ePostProcessing ppBias = ePostProcessing::None )
		{
			tempString.Format( "encoder.blocks.%i.%s.weight", i, name );
			map[ tempString ] = PendingTensor{ t.w, ppWeight };
			tempString.Format( "encoder.blocks.%i.%s.bias", i, name );
			map[ tempString ] = PendingTensor{ t.b, ppBias };
		};

		for( int i = 0; i < layersEnc; i++ )
		{
			auto& gpu = enc.layers[ i ];
			add2( "mlp_ln", i, gpu.mlpLn );
			add2( "mlp.0", i, gpu.mlp0, ePostProcessing::MakePanels );
			add2( "mlp.2", i, gpu.mlp1, ePostProcessing::MakePanels );
			add2( "attn_ln", i, gpu.attnLn0 );
			add2( "attn.query", i, gpu.attnQuery, ePostProcessing::MakePanels );
			add( "attn.key.weight", i, gpu.attnKey, ePostProcessing::MakePanels );
			add2( "attn.value", i, gpu.attnValue, ePostProcessing::MakePanels );
			add2( "attn.out", i, gpu.attnLn1, ePostProcessing::MakePanels );
		}
	}

	void populateDecodeTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersDec, DirectCompute::ModelBuffers& tensors, bool hybrid )
	{
		tensors.dec.layers.resize( layersDec );
		CStringA tempString;
		// Decoder tensors

		auto& dec = tensors.dec;
		if( !hybrid )
		{
			map[ "decoder.positional_embedding" ] = dec.positionalEmbedding;
			map[ "decoder.token_embedding.weight" ] = dec.tokenEmbedding;
			map[ "decoder.ln.weight" ] = dec.ln.w;
			map[ "decoder.ln.bias" ] = dec.ln.b;
		}

		auto add = [ & ]( const char* name, int i, DirectCompute::Tensor& t, ePostProcessing pp = ePostProcessing::None )
		{
			tempString.Format( "decoder.blocks.%i.%s", i, name );
			map[ tempString ] = PendingTensor{ t, pp };
		};
		auto add2 = [ & ]( const char* name, int i, DirectCompute::TensorPair& t, ePostProcessing ppWeight = ePostProcessing::None, ePostProcessing ppBias = ePostProcessing::None )
		{
			tempString.Format( "decoder.blocks.%i.%s.weight", i, name );
			map[ tempString ] = PendingTensor{ t.w, ppWeight };
			tempString.Format( "decoder.blocks.%i.%s.bias", i, name );
			map[ tempString ] = PendingTensor{ t.b, ppBias };
		};

		for( int i = 0; i < layersDec; i++ )
		{
			auto& gpu = dec.layers[ i ];
			add( "cross_attn.key.weight", i, gpu.crossAttnKey, ePostProcessing::MakePanels );
			add2( "cross_attn.value", i, gpu.crossAttnValue, ePostProcessing::MakePanels );
			if( hybrid )
				continue;

			add2( "mlp_ln", i, gpu.mlpLn );
			add2( "mlp.0", i, gpu.mlp0, ePostProcessing::MakePanels );
			add2( "mlp.2", i, gpu.mlp1, ePostProcessing::MakePanels );
			add2( "attn_ln", i, gpu.attnLn0 );
			add2( "attn.query", i, gpu.attnQuery );
			add( "attn.key.weight", i, gpu.attnKey );
			add2( "attn.value", i, gpu.attnValue );
			add2( "attn.out", i, gpu.attnLn1 );
			add2( "cross_attn_ln", i, gpu.crossAttnLn0 );
			add2( "cross_attn.query", i, gpu.crossAttnQuery );
			add2( "cross_attn.out", i, gpu.crossAttnLn1 );
		}
	}

	void populateTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersEnc, int layersDec, DirectCompute::ModelBuffers& tensors, bool hybrid )
	{
		populateEncodeTensorsMap( map, layersEnc, tensors );
		populateDecodeTensorsMap( map, layersDec, tensors, hybrid );
	}

	struct sTensorHeader
	{
		int n_dims, length, ftype;
	};

	// compare signed int32 lanes for a <= b
	inline __m128i cmple( __m128i a, __m128i b )
	{
		__m128i i = _mm_min_epi32( a, b );
		return _mm_cmpeq_epi32( a, i );
	}

	inline bool allPositive( const std::array<int, 4>& ne )
	{
		const __m128i v = _mm_loadu_si128( ( const __m128i* )ne.data() );
		const __m128i le = cmple( v, _mm_setzero_si128() );
		return (bool)_mm_testz_si128( le, le );
	}

	inline const char* cstr( const CStringA& s ) { return s; }

	// token_embedding is used both as the decoder input-embedding gather (addRows) and as
	// the output logits projection. To avoid needing a quantized addRows shader, we keep it
	// as fp16 in VRAM even for quantized models (dequantized at load).
	constexpr char c_tokenEmbeddingName[] = "decoder.token_embedding.weight";

	// Reusable scratch buffers for streaming tensor payloads.
	struct TensorLoadBuffers
	{
		std::vector<uint8_t> raw;		// raw on-disk bytes (f16/f32/quant blocks)
		std::vector<uint8_t> repacked;	// SoA bytes for native-quant GPU upload
		std::vector<uint16_t> fp16;		// dequantized fp16 (fallback / token_embedding)
	};

	// Reads one tensor's payload from the stream and creates the GPU tensor `dest`.
	// dtOut receives the effective GPU data type (for the caller's postProcess step).
	// preferNativeQuant=false forces a quantized tensor to be dequantized to fp16.
	HRESULT loadTensorPayload( ComLight::iReadStream* stm, int ftype, const std::array<int, 4>& ne,
		bool preferNativeQuant, DirectCompute::Tensor& dest, DirectCompute::eDataType& dtOut,
		TensorLoadBuffers& buf, int64_t& cbVram )
	{
		using namespace DirectCompute;
		const eGgmlType gt = (eGgmlType)ftype;
		switch( gt )
		{
		case eGgmlType::f32:
		case eGgmlType::f16:
		case eGgmlType::q4_0:
		case eGgmlType::q4_1:
		case eGgmlType::q5_0:
		case eGgmlType::q5_1:
		case eGgmlType::q8_0:
		case eGgmlType::q2_K:	// K-quants: loaded via dequant-to-fp16 (no native GPU shader)
		case eGgmlType::q3_K:
		case eGgmlType::q4_K:
		case eGgmlType::q5_K:
		case eGgmlType::q6_K:
			break;
		default:
			logError( u8"Unsupported tensor ggml type %i in model file", ftype );
			return E_INVALIDARG;
		}

		const size_t nElements = (size_t)(uint32_t)ne[ 0 ] * (uint32_t)ne[ 1 ] * (uint32_t)ne[ 2 ] * (uint32_t)ne[ 3 ];
		const size_t rawBytes = ggmlTensorBytes( gt, nElements );
		if( rawBytes > UINT_MAX )
			return DISP_E_OVERFLOW;
		try { buf.raw.resize( rawBytes ); }
		catch( const std::bad_alloc& ) { return E_OUTOFMEMORY; }
		CHECK( readBytes( stm, buf.raw.data(), rawBytes ) );

		if( gt == eGgmlType::f32 )
		{
			dtOut = eDataType::FP32;
			CHECK( dest.createImmutable( dtOut, ne, buf.raw.data() ) );
			cbVram += (int64_t)rawBytes;
		}
		else if( gt == eGgmlType::f16 )
		{
			dtOut = eDataType::FP16;
			CHECK( dest.createImmutable( dtOut, ne, buf.raw.data() ) );
			cbVram += (int64_t)rawBytes;
		}
		else if( preferNativeQuant && ggmlHasNativeGpuQuant( gt ) )
		{
			switch( gt )
			{
			case eGgmlType::q5_0: dtOut = eDataType::Q5_0; break;
			case eGgmlType::q8_0: dtOut = eDataType::Q8_0; break;
			case eGgmlType::q4_0: dtOut = eDataType::Q4_0; break;
			case eGgmlType::q4_1: dtOut = eDataType::Q4_1; break;
			case eGgmlType::q5_1: dtOut = eDataType::Q5_1; break;
			default: return E_UNEXPECTED;
			}
			const QuantGpuLayout layout = computeQuantGpuLayout( gt, nElements );
			try { buf.repacked.resize( layout.totalBytes ); }
			catch( const std::bad_alloc& ) { return E_OUTOFMEMORY; }
			repackQuantToGpu( gt, buf.raw.data(), nElements, buf.repacked.data() );
			CHECK( dest.createImmutableQuantized( dtOut, ne, buf.repacked.data(), layout.totalBytes ) );
			cbVram += (int64_t)layout.totalBytes;
		}
		else
		{
			// Dequantize to fp16: token_embedding, or a quant type without a native GPU
			// shader (the K-quants).
			dtOut = eDataType::FP16;
			try { buf.fp16.resize( nElements ); }
			catch( const std::bad_alloc& ) { return E_OUTOFMEMORY; }
			dequantizeToFp16( gt, buf.raw.data(), buf.fp16.data(), nElements );
			CHECK( dest.createImmutable( dtOut, ne, buf.fp16.data() ) );
			cbVram += (int64_t)nElements * 2;
		}
		return S_OK;
	}

	// ---- GGUF helpers ----

	// Slaney mel scale (matches librosa / whisper's precomputed mel_filters).
	inline double hzToMelSlaney( double hz )
	{
		constexpr double f_sp = 200.0 / 3.0;
		constexpr double min_log_hz = 1000.0;
		const double min_log_mel = min_log_hz / f_sp;			// 15
		const double logstep = std::log( 6.4 ) / 27.0;
		if( hz < min_log_hz )
			return hz / f_sp;
		return min_log_mel + std::log( hz / min_log_hz ) / logstep;
	}
	inline double melToHzSlaney( double mel )
	{
		constexpr double f_sp = 200.0 / 3.0;
		constexpr double min_log_hz = 1000.0;
		const double min_log_mel = min_log_hz / f_sp;
		const double logstep = std::log( 6.4 ) / 27.0;
		if( mel < min_log_mel )
			return f_sp * mel;
		return min_log_hz * std::exp( logstep * ( mel - min_log_mel ) );
	}

	// Reconstruct whisper's mel filter bank [n_mels x (1+FFT_SIZE/2)], row-major (mel-major),
	// matching librosa.filters.mel(sr=16000, n_fft=400, n_mels). GGUF has no standard slot
	// for the filters, so we compute the deterministic standard bank.
	void computeWhisperMelFilters( int n_mels, std::vector<float>& out )
	{
		const int nBins = 1 + (int)FFT_SIZE / 2;			// 201
		const double sr = (double)SAMPLE_RATE;
		out.assign( (size_t)n_mels * nBins, 0.0f );

		const double melMin = hzToMelSlaney( 0.0 );
		const double melMax = hzToMelSlaney( sr / 2.0 );
		std::vector<double> hz( n_mels + 2 );
		for( int i = 0; i < n_mels + 2; i++ )
		{
			const double mel = melMin + ( melMax - melMin ) * i / ( n_mels + 1 );
			hz[ i ] = melToHzSlaney( mel );
		}
		for( int m = 0; m < n_mels; m++ )
		{
			const double lo = hz[ m ], ctr = hz[ m + 1 ], hi = hz[ m + 2 ];
			const double enorm = 2.0 / ( hi - lo );			// slaney normalization
			for( int k = 0; k < nBins; k++ )
			{
				const double f = (double)k * sr / (double)FFT_SIZE;	// fft bin frequency
				const double lower = ( f - lo ) / ( ctr - lo );
				const double upper = ( hi - f ) / ( hi - ctr );
				double w = std::min( lower, upper );
				if( w < 0 ) w = 0;
				out[ (size_t)m * nBins + k ] = (float)( w * enorm );
			}
		}
	}

	// Derive whisper hyperparameters from the GGUF tensor shapes (robust to any metadata
	// convention), with optional metadata overrides. whisper always uses head_dim 64.
	HRESULT inferGgufHparams( const GgufFile& gguf, Whisper::sModelParams& p )
	{
		const GgufTensor* tokEmb = gguf.findTensor( "decoder.token_embedding.weight" );
		const GgufTensor* conv1 = gguf.findTensor( "encoder.conv1.weight" );
		const GgufTensor* encPos = gguf.findTensor( "encoder.positional_embedding" );
		const GgufTensor* decPos = gguf.findTensor( "decoder.positional_embedding" );
		if( nullptr == tokEmb || nullptr == conv1 || nullptr == encPos || nullptr == decPos )
		{
			logError( u8"GGUF model is missing required whisper tensors (token_embedding / conv1 / positional_embedding)" );
			return E_INVALIDARG;
		}

		p.n_text_state = (int)tokEmb->ne[ 0 ];
		p.n_vocab = (int)tokEmb->ne[ 1 ];
		p.n_mels = (int)conv1->ne[ 1 ];
		p.n_audio_state = (int)encPos->ne[ 0 ];
		p.n_audio_ctx = (int)encPos->ne[ 1 ];
		p.n_text_ctx = (int)decPos->ne[ 1 ];

		int encLayers = 0, decLayers = 0, idx = 0;
		for( const GgufTensor& t : gguf.tensors )
		{
			if( 1 == sscanf_s( t.name.c_str(), "encoder.blocks.%d.", &idx ) )
				encLayers = std::max( encLayers, idx + 1 );
			else if( 1 == sscanf_s( t.name.c_str(), "decoder.blocks.%d.", &idx ) )
				decLayers = std::max( decLayers, idx + 1 );
		}
		p.n_audio_layer = encLayers;
		p.n_text_layer = decLayers;
		p.n_audio_head = p.n_audio_state / 64;
		p.n_text_head = p.n_text_state / 64;
		p.f16 = 1;

		// Optional metadata overrides (documented whisper.* keys), if present.
		int64_t v;
		if( gguf.getInt( "whisper.mel_bins", v ) ) p.n_mels = (int)v;
		if( gguf.getInt( "whisper.vocab_size", v ) ) p.n_vocab = (int)v;

		if( p.n_audio_layer <= 0 || p.n_text_layer <= 0 || p.n_audio_head <= 0 || p.n_text_head <= 0 ||
			p.n_vocab <= 0 || p.n_text_state <= 0 || p.n_audio_state <= 0 )
		{
			logError( u8"GGUF model has implausible whisper dimensions" );
			return E_INVALIDARG;
		}
		if( p.n_mels > N_MEL_MAX )
		{
			logError( u8"Unsupported mel-filterbank size %i (max %u)", p.n_mels, (uint32_t)N_MEL_MAX );
			return E_INVALIDARG;
		}
		return S_OK;
	}
}

class WhisperModel::CallbacksImpl : public CpuCompute::iLoaderProgressSink
{
	sLoadModelCallbacks lmcb;
	int64_t fileSize;

	HRESULT gotBytes( int64_t cb ) override final
	{
		if( nullptr != lmcb.cancel )
		{
			HRESULT hr = lmcb.cancel( lmcb.pv );
			CHECK( hr );
			if( S_OK != hr )
				return HRESULT_FROM_WIN32( ERROR_CANCELLED );
		}

		if( nullptr != lmcb.progress )
		{
			postponedBytes -= cb;
			assert( postponedBytes >= 0 );
			int64_t pos = fileSize - postponedBytes;
			const double progressVal = (double)pos / (double)fileSize;
			HRESULT hr = lmcb.progress( progressVal, lmcb.pv );
			CHECK( hr );
		}
		return S_OK;
	}
public:
	int64_t postponedBytes;

	CallbacksImpl()
	{
		lmcb.progress = nullptr;
		lmcb.cancel = nullptr;
		lmcb.pv = nullptr;
		fileSize = 0;
		postponedBytes = 0;
	}

	HRESULT initialize( ComLight::iReadStream* stm, const sLoadModelCallbacks* rsi )
	{
		if( nullptr == rsi )
			return S_OK;
		lmcb = *rsi;
		if( nullptr != lmcb.progress )
			CHECK( stm->getLength( fileSize ) );
		return S_OK;
	}

	HRESULT call( ComLight::iReadStream* stm )
	{
		if( nullptr != lmcb.cancel )
		{
			HRESULT hr = lmcb.cancel( lmcb.pv );
			CHECK( hr );
			if( S_OK != hr )
				return HRESULT_FROM_WIN32( ERROR_CANCELLED );
		}

		if( nullptr != lmcb.progress )
		{
			int64_t pos;
			CHECK( stm->getPosition( pos ) );
			pos -= postponedBytes;
			const double progressVal = (double)pos / (double)fileSize;
			HRESULT hr = lmcb.progress( progressVal, lmcb.pv );
			CHECK( hr );
		}
		return S_OK;
	}
};

HRESULT WhisperModel::loadGpu( ComLight::iReadStream* stm, CallbacksImpl& callbacks )
{
	CAtlMap<CStringA, PendingTensor> map;
	populateTensorsMap( map, parameters.n_audio_layer, parameters.n_text_layer, tensors, false );

	DirectCompute::Reshaper reshape;

	TensorLoadBuffers loadBuffers;
	size_t countLoaded = 0;
	CStringA name;
	int64_t cb = 0;
	while( true )
	{
		CHECK( callbacks.call( stm ) );

		sTensorHeader header;
		HRESULT hr = readStruct( stm, header );
		if( hr == E_EOF )
			break;
		if( FAILED( hr ) )
			return hr;
		if( header.n_dims < 1 || header.n_dims>3 )
			return E_INVALIDARG;

		std::array<int, 4> ne = { 1, 1, 1, 1 };
		CHECK( readBytes( stm, ne.data(), header.n_dims * 4 ) );
		if( !allPositive( ne ) )
			return E_INVALIDARG;

		char* nameBuffer = name.GetBufferSetLength( header.length );
		hr = readBytes( stm, nameBuffer, header.length );
		name.ReleaseBuffer();
		if( FAILED( hr ) )
			return hr;

		auto p = map.Lookup( name );
		if( nullptr == p )
		{
			logError( u8"%s: unknown tensor '%s' in model file", __func__, cstr( name ) );
			return E_INVALIDARG;
		}

		DirectCompute::eDataType dt;
		const bool preferNativeQuant = ( 0 != name.Compare( c_tokenEmbeddingName ) );
		CHECK( loadTensorPayload( stm, header.ftype, ne, preferNativeQuant, *p->m_value.dest, dt, loadBuffers, cb ) );
		CHECK( p->m_value.postProcess( reshape, dt ) );
		countLoaded++;
	}

	if( countLoaded != map.GetCount() )
	{
		logError( u8"Not all tensors loaded from model file - expected %zu, got %zu", map.GetCount(), countLoaded );
		return E_INVALIDARG;
	}

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded %zu GPU tensors, %g MB VRAM", countLoaded, mulMb * cb );
	return S_OK;
}

#if BUILD_HYBRID_VERSION
HRESULT WhisperModel::loadHybrid( ComLight::iReadStream* stm, CallbacksImpl& callbacks )
{
	CAtlMap<CStringA, PendingTensor> map;
	populateTensorsMap( map, parameters.n_audio_layer, parameters.n_text_layer, tensors, true );
	DirectCompute::Reshaper reshape;
	CpuCompute::HybridLoader loader( shared->hybridTensors, parameters.n_text_layer );

	TensorLoadBuffers loadBuffers;
	size_t countLoaded = 0;
	CStringA name;
	int64_t cb = 0;
	while( true )
	{
		CHECK( callbacks.call( stm ) );

		sTensorHeader header;
		HRESULT hr = readStruct( stm, header );
		if( hr == E_EOF )
			break;
		if( FAILED( hr ) )
			return hr;
		if( header.n_dims < 1 || header.n_dims > 3 )
			return E_INVALIDARG;

		std::array<int, 4> ne = { 1, 1, 1, 1 };
		CHECK( readBytes( stm, ne.data(), header.n_dims * 4 ) );
		if( !allPositive( ne ) )
			return E_INVALIDARG;

		char* nameBuffer = name.GetBufferSetLength( header.length );
		hr = readBytes( stm, nameBuffer, header.length );
		name.ReleaseBuffer();
		if( FAILED( hr ) )
			return hr;

		auto p = map.Lookup( name );
		if( nullptr == p )
		{
			HRESULT hr = loader.setupTensor( name, header.n_dims, header.ftype, ne, stm, callbacks.postponedBytes );
			if( hr == S_OK )
				continue;
			logError( u8"%s: unknown tensor '%s' in model file", __func__, cstr( name ) );
			return E_INVALIDARG;
		}

		DirectCompute::eDataType dt;
		const bool preferNativeQuant = ( 0 != name.Compare( c_tokenEmbeddingName ) );
		CHECK( loadTensorPayload( stm, header.ftype, ne, preferNativeQuant, *p->m_value.dest, dt, loadBuffers, cb ) );
		CHECK( p->m_value.postProcess( reshape, dt ) );
		countLoaded++;
	}

	if( countLoaded != map.GetCount() )
	{
		logError( u8"Not all tensors loaded from model file - expected %zu, got %zu", map.GetCount(), countLoaded );
		return E_INVALIDARG;
	}

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded %zu GPU tensors, %g MB VRAM", countLoaded, mulMb * cb );

	CHECK( loader.completeLoad( stm, callbacks ) );
	return S_OK;
}
#endif

HRESULT WhisperModel::loadGguf( ComLight::iReadStream* stm, CallbacksImpl& callbacks )
{
	GgufFile gguf;
	CHECK( gguf.parse( stm, true ) );	// the 4-byte magic was already consumed by load()

	// Hyperparameters — inferred from the tensor shapes (+ optional metadata overrides).
	CHECK( inferGgufHparams( gguf, parameters ) );

	// Vocabulary from the GGUF tokenizer metadata.
	const std::vector<std::string>* toks = gguf.getStringArray( "tokenizer.ggml.tokens" );
	if( nullptr == toks )
	{
		logError( u8"GGUF model has no 'tokenizer.ggml.tokens' metadata; cannot build vocabulary" );
		return E_INVALIDARG;
	}
	CHECK( shared->vocab.loadFromTokens( *toks, parameters.n_vocab ) );

	// Mel filter bank — GGUF has no standard slot, so compute the deterministic whisper bank.
	shared->filters.n_mel = (uint32_t)parameters.n_mels;
	shared->filters.n_fft = 1 + FFT_SIZE / 2;
	computeWhisperMelFilters( parameters.n_mels, shared->filters.data );

	// GPU tensors — reuse the legacy name map + per-tensor loader; read each from the
	// GGUF data section by its offset.
	CAtlMap<CStringA, PendingTensor> map;
	populateTensorsMap( map, parameters.n_audio_layer, parameters.n_text_layer, tensors, false );

	DirectCompute::Reshaper reshape;
	TensorLoadBuffers loadBuffers;
	size_t countLoaded = 0;
	int64_t cb = 0;

	for( const GgufTensor& gt : gguf.tensors )
	{
		CStringA name( gt.name.c_str() );
		auto p = map.Lookup( name );
		if( nullptr == p )
			continue;	// not part of the whisper model (e.g. an embedded mel bank)

		std::array<int, 4> ne = { (int)gt.ne[ 0 ], (int)gt.ne[ 1 ], (int)gt.ne[ 2 ], (int)gt.ne[ 3 ] };
		if( !allPositive( ne ) )
			return E_INVALIDARG;

		CHECK( stm->seek( gguf.dataOffset + (int64_t)gt.offset, ComLight::eSeekOrigin::Begin ) );

		DirectCompute::eDataType dt;
		const bool preferNativeQuant = ( 0 != name.Compare( c_tokenEmbeddingName ) );
		CHECK( loadTensorPayload( stm, gt.ggmlType, ne, preferNativeQuant, *p->m_value.dest, dt, loadBuffers, cb ) );
		CHECK( p->m_value.postProcess( reshape, dt ) );
		countLoaded++;
		CHECK( callbacks.call( stm ) );
	}

	if( countLoaded != map.GetCount() )
	{
		logError( u8"GGUF model is missing tensors - expected %zu, got %zu", map.GetCount(), countLoaded );
		return E_INVALIDARG;
	}

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded GGUF model: %zu GPU tensors, %g MB VRAM", countLoaded, mulMb * cb );
	return S_OK;
}

HRESULT WhisperModel::load( ComLight::iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks )
{
	CpuProfiler cpuPerf;
	CallbacksImpl cb;
	CHECK( cb.initialize( stm, callbacks ) );
	// verify magic — dispatch to the GGUF path if it's a GGUF container.
	{
		uint32_t magic;
		CHECK( readStruct( stm, magic ) );
		if( magic == GgufFile::magic )
		{
			shared = std::make_shared<ModelShared>();
			DirectCompute::GpuProfilerSimple gpuProf;
			CHECK( gpuProf.create() );
			CHECK( loadGguf( stm, cb ) );
			CHECK( gpuProf.time( loadTimeGpu ) );
			loadTimeCpu = cpuPerf.elapsed();
			return S_OK;
		}
		if( magic != 0x67676d6c )
		{
			logError( u8"Invalid model file, bad magic" );
			return E_INVALIDARG;
		}
	}

	shared = std::make_shared<ModelShared>();

	// hparams and MEL filters
	{
		ParamsAndMelHeader pmh;
		CHECK( readStruct( stm, pmh ) );
		parameters = pmh.mp;
		assert( parameters.n_text_state == parameters.n_audio_state );

		shared->filters.n_mel = pmh.n_mel;
		shared->filters.n_fft = pmh.n_fft;

		// The CPU mel-spectrogram path sizes std::array buffers to N_MEL_MAX and copies
		// exactly parameters.n_mels rows into the encoder input tensor. Guard both invariants.
		if( pmh.n_mel > N_MEL_MAX )
		{
			logError( u8"Unsupported mel-filterbank size %u (max %u). Rebuild with a larger N_MEL_MAX.", pmh.n_mel, (uint32_t)N_MEL_MAX );
			return E_INVALIDARG;
		}
		if( (int)pmh.n_mel != parameters.n_mels )
		{
			logError( u8"Model is inconsistent: MEL filter rows %u != hparams n_mels %i", pmh.n_mel, parameters.n_mels );
			return E_INVALIDARG;
		}

		const size_t len = (size_t)pmh.n_mel * pmh.n_fft;
		shared->filters.data.resize( len );
		CHECK( readBytes( stm, shared->filters.data.data(), len * 4 ) );

		const int64_t cb = len * 4;
		constexpr double mulKb = 1.0 / ( 1 << 10 );
		logDebug( u8"Loaded MEL filters, %.1f kb RAM", mulKb * cb );
	}
	CHECK( cb.call( stm ) );

	// Vocabulary
	CHECK( shared->vocab.load( stm, parameters.n_vocab ) );
	CHECK( cb.call( stm ) );

	DirectCompute::GpuProfilerSimple gpuProfiler;
	CHECK( gpuProfiler.create() );

	if( hybrid )
	{
#if BUILD_HYBRID_VERSION
		CHECK( loadHybrid( stm, cb ) )
#else
		return E_NOTIMPL;
#endif
	}
	else
		CHECK( loadGpu( stm, cb ) );

	CHECK( gpuProfiler.time( loadTimeGpu ) );
	loadTimeCpu = cpuPerf.elapsed();
	return S_OK;
}

HRESULT Whisper::WhisperModel::createClone( const WhisperModel& rsi )
{
	parameters = rsi.parameters;
	shared = rsi.shared;
	CHECK( tensors.createClone( rsi.tensors ) );
	return S_OK;
}

__m128i Whisper::WhisperModel::getMemoryUse() const
{
	size_t cb = shared->vocab.getMemoryUse();
	cb += vectorMemoryUse( shared->filters.data );
	__m128i v = _mm_cvtsi64_si128( (int64_t)cb );
	v = _mm_add_epi64( v, tensors.getMemoryUse() );
	return v;
}