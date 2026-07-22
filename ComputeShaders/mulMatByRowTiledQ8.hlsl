// Native q8_0 variant of mulMatByRowTiled (matrix * single-column, the decode hot path).
// The body lives in mulMatByRowTiled.hlsl, guarded by QUANT_TYPE.
#define QUANT_TYPE 8
#include "mulMatByRowTiled.hlsl"
