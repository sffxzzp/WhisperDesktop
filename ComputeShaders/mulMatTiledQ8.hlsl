// Native q8_0 variant of mulMatTiled (the main matrix*matrix product).
// The body lives in mulMatTiled.hlsl, guarded by QUANT_TYPE.
#define QUANT_TYPE 8
#include "mulMatTiled.hlsl"
