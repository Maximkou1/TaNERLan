from transformers import AutoConfig

from .span_ner import ModernBertForSpanNer, ModernBertSpanNerConfig, SpanNerOutput

# чтобы AutoTokenizer/AutoConfig понимали config.json span-модели (model_type = modernbert_span_ner)
try:
    AutoConfig.register(ModernBertSpanNerConfig.model_type, ModernBertSpanNerConfig)
except ValueError:  # уже зарегистрирован
    pass

__all__ = ["ModernBertForSpanNer", "ModernBertSpanNerConfig", "SpanNerOutput"]
