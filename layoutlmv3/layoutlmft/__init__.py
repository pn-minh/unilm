import requests

# Fix for transformers==4.12.5 / datasets==1.15.1 with modern Hugging Face Hub:
# HF Hub now returns relative redirect URLs (e.g. '/api/resolve-cache/...') on 302 responses,
# which older transformers/datasets pass directly to requests.get() without prepending the host.
_orig_prepare_request = requests.Session.prepare_request


def _hf_relative_redirect_prepare_request(self, request, *args, **kwargs):
    if isinstance(request.url, str) and request.url.startswith("/"):
        request.url = "https://huggingface.co" + request.url
    return _orig_prepare_request(self, request, *args, **kwargs)


requests.Session.prepare_request = _hf_relative_redirect_prepare_request

from .models import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Tokenizer,
)
