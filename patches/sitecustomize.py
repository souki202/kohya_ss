"""Runtime patches for external training scripts.

This module is discovered automatically by Python on start-up
so we can tweak behaviour without touching vendored sources.
"""

from __future__ import annotations

import logging
from typing import Iterable


log = logging.getLogger(__name__)


def _patch_lumina_strategy() -> None:
    try:
        from library import strategy_lumina  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Unable to import strategy_lumina for patching: %s", exc)
        return

    original_encode_tokens = strategy_lumina.LuminaTextEncodingStrategy.encode_tokens

    def encode_tokens(self, tokenize_strategy, models, tokens_and_masks):  # type: ignore[override]
        if not isinstance(models, Iterable) or isinstance(models, (str, bytes)):
            models_list = [models]
        else:
            models_list = list(models)
        return original_encode_tokens(self, tokenize_strategy, models_list, tokens_and_masks)

    strategy_lumina.LuminaTextEncodingStrategy.encode_tokens = encode_tokens


_patch_lumina_strategy()
