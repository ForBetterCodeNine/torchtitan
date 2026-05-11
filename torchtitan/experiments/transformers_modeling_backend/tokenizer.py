# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


class HFBackendTokenizer(HuggingFaceTokenizer):
    """Tokenizer that passes special tokens to chat templates.

    Some HF models' chat templates reference bos_token, eos_token, etc.
    The base tokenizer only passes messages. This subclass auto-injects
    special tokens so all HF chat templates work.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("bos_token", self.bos_token or "")
        kwargs.setdefault("eos_token", self.eos_token or "")
        kwargs.setdefault("add_generation_prompt", True)
        return super().apply_chat_template(messages, **kwargs)
