# Unit tests

from src.data.loader import MIDICapsLoader

def test_small_batch():
    loader = MIDICapsLoader()
    train_loader, _ = loader.load_data(train_size=2)
    batch = next(iter(train_loader))
    assert batch['input_ids'].shape[0] == 2  # Batch size