import pytest
import torch

from turboquant import pack_bits, unpack_bits


def test_pack_roundtrip_all_supported_bit_widths() -> None:
    for bits in range(1, 9):
        values = (torch.arange(259) * 7 + 3).remainder(1 << bits).reshape(7, 37)
        packed = pack_bits(values, bits)
        recovered = unpack_bits(packed)
        torch.testing.assert_close(recovered, values)
        assert packed.nbytes == (values.numel() * bits + 7) // 8


def test_pack_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="fit"):
        pack_bits(torch.tensor([0, 4]), 2)
