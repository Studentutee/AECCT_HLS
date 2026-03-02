#pragma once
// Design-safe shim for TB weight streamer helpers.
// TB implementation moved to tb/weights_streamer.h to keep design roots free of weights.h includes.

#ifndef __SYNTHESIS__
#include "tb/weights_streamer.h"
#endif