#pragma once
#include <stdint.h>

namespace Whisper
{
	// WHISPER_SAMPLE_RATE, 16 kHz
	constexpr uint32_t SAMPLE_RATE = 16000;
	// WHISPER_N_FFT, 25 milliseconds
	constexpr uint32_t FFT_SIZE = 400;
	// WHISPER_HOP_LENGTH, 10 milliseconds
	constexpr uint32_t FFT_STEP = 160;
	// WHISPER_N_MEL — maximum supported mel-filterbank size.
	// large-v3 / large-v3-turbo use 128 mel bins; all older models use 80.
	// This is a COMPILE-TIME CAPACITY only (sizes std::array buffers). The actual
	// number of mel bins for a given model is a runtime value, read from the model
	// file header (Filters::n_mel / sModelParams::n_mels) and threaded through the code.
	constexpr uint32_t N_MEL_MAX = 128;
}