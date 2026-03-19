# REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW
Date: 2026-03-19

用途：提供可直接貼用的 reviewer verdict 範本，降低「口語化但不一致」與「過度宣稱 closure」風險。

## 1. Verdict levels（建議用語）
- 可接受 / Acceptable
- 可接受但有保留 / Acceptable with reservations
- 部分完成 / Partial
- 需補件 / Needs follow-up

<a id="ultra-short-note-template"></a>
## 2. Ultra-short note template
```md
Verdict: <Acceptable | Acceptable with reservations | Partial | Needs follow-up>
Scope checked: <checked files/sections>
Key findings: <1~3 items>
Evidence posture: local-only / deferred closure
Limitations: <none or 1 line>
Deferred items: <none or 1~2 items>
```

## 3. Ultra-short examples
### 3.1 可接受 / Acceptable
```md
Verdict: Acceptable
Scope checked: REVIEWER_GUIDE + CHECKLIST + ATTN cross-check + ternary role map
Key findings: reviewer-facing ownership/boundary and quick-entry pointers are coherent
Evidence posture: local-only
Limitations: Catapult/SCVerify not in this pass
Deferred items: none (reviewer-facing blocking debt)
```

### 3.2 可接受但有保留 / Acceptable with reservations
```md
Verdict: Acceptable with reservations
Scope checked: guide + checklist
Key findings: main review flow usable, but one cross-link or section wording still ambiguous
Evidence posture: local-only
Limitations: minor doc ambiguity remains
Deferred items: tighten one specific section in next docs-only pass
```

### 3.3 部分完成 / Partial
```md
Verdict: Partial
Scope checked: checklist and one companion doc
Key findings: core checklist added, but quick-entry map or verdict consistency not fully merged
Evidence posture: local-only
Limitations: artifact coverage incomplete
Deferred items: finish missing companion docs and guide sync
```

### 3.4 需補件 / Needs follow-up
```md
Verdict: Needs follow-up
Scope checked: partial review only
Key findings: blocking inconsistency remains in ownership/fallback/write-back wording
Evidence posture: local-only
Limitations: cannot issue stable reviewer-ready acceptance
Deferred items: resolve blocking wording contradictions first
```

<a id="full-review-note-template"></a>
## 4. Full review note template
```md
Verdict: <Acceptable | Acceptable with reservations | Partial | Needs follow-up>

Scope checked:
- Files:
- Sections:
- Out-of-scope:

Key findings:
1.
2.
3.

Evidence posture:
- What PASS means in this review:
- What PASS does NOT mean:

Limitations:
- 

Deferred items:
- 

Recommended next step:
- optional enhancement docs
- or resume functional/mainline work
```

## 5. Wording guardrails（避免 overclaim）
- 若未執行 Catapult/SCVerify，不可寫成 formal closure。
- docs-only pass 應明示：不含 source behavior re-validation。
- 若是可接受但有保留，必須列出保留項與建議下一步。
