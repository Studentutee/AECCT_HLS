#pragma once
// Local wrapper for repo-tracked exported weight/header payloads.
//
// Intent:
// - keep design/test code on a stable include spelling: "weights.h"
// - avoid scattering data/weights path knowledge across many files
// - allow later replacement or indirection without changing every include site

#include "data/weights/weights.h"
