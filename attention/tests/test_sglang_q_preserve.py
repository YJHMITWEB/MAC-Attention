import torch

from mac_attention.integrations.sglang.llama_hooks import _preserve_q_before_rope


class _Metadata:
    def __init__(self, num_heads=2, head_dim=4, decode_buf=None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decode_q_before_rope_buf = decode_buf
        self.cur_q_before_rope = None
        self.cur_q_before_rope_local_start = -1
        self.cur_q_before_rope_local_len = -1


class _ForwardBatch:
    def __init__(self, update_start, prefix_len=4, extend_len=5):
        if isinstance(update_start, list):
            self.mac_prefill_cache_update_starts = update_start
            self.extend_prefix_lens_cpu = prefix_len
            self.extend_seq_lens_cpu = extend_len
            self.batch_size = len(update_start)
        else:
            self.batch_size = 1
            self.mac_prefill_cache_update_starts = [update_start]
            self.extend_prefix_lens_cpu = [prefix_len]
            self.extend_seq_lens_cpu = [extend_len]


def test_decode_preserve_q_before_rope_does_not_alias_q():
    q = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    expected = q.reshape(1, 2, 4).clone()
    metadata = _Metadata(decode_buf=torch.empty((1, 2, 4), dtype=q.dtype))

    _preserve_q_before_rope(metadata, q, current_is_decode=True)
    q.add_(1000)

    torch.testing.assert_close(metadata.cur_q_before_rope, expected)
    assert metadata.cur_q_before_rope_local_start == 0
    assert metadata.cur_q_before_rope_local_len == 1


def test_single_request_prefill_preserves_only_scheduled_cache_rows():
    q = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    expected = q.reshape(5, 2, 4)[2:5].clone()
    metadata = _Metadata()
    forward_batch = _ForwardBatch(update_start=6, prefix_len=4, extend_len=5)

    _preserve_q_before_rope(
        metadata,
        q,
        current_is_decode=False,
        forward_batch=forward_batch,
    )
    q.add_(1000)

    torch.testing.assert_close(metadata.cur_q_before_rope, expected)
    assert metadata.cur_q_before_rope.shape[0] == 3
    assert metadata.cur_q_before_rope_local_start == 2
    assert metadata.cur_q_before_rope_local_len == 3


def test_single_request_prefill_without_cache_update_preserves_no_q():
    q = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    metadata = _Metadata()
    forward_batch = _ForwardBatch(update_start=-1)

    _preserve_q_before_rope(
        metadata,
        q,
        current_is_decode=False,
        forward_batch=forward_batch,
    )

    assert metadata.cur_q_before_rope is None
    assert metadata.cur_q_before_rope_local_start == 0
    assert metadata.cur_q_before_rope_local_len == 0


def test_multi_request_prefill_compacts_only_scheduled_cache_rows():
    q = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    q_view = q.reshape(12, 2, 4)
    expected = torch.cat([q_view[6:9], q_view[11:12]], dim=0).clone()
    metadata = _Metadata()
    forward_batch = _ForwardBatch(
        update_start=[-1, 22, 32],
        prefix_len=[10, 20, 30],
        extend_len=[4, 5, 3],
    )

    _preserve_q_before_rope(
        metadata,
        q,
        current_is_decode=False,
        forward_batch=forward_batch,
    )
    q.add_(1000)

    torch.testing.assert_close(metadata.cur_q_before_rope, expected)
    assert metadata.cur_q_before_rope.shape[0] == 4
    assert metadata.cur_q_before_rope_local_start == -1
    assert metadata.cur_q_before_rope_local_len == 4
    assert metadata.cur_q_before_rope_compacted
