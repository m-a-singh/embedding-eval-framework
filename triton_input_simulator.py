import numpy as np
from sentence_transformers import SentenceTransformer


def simulate_triton_encode(
    model: SentenceTransformer, texts: list[str], model_name: str
) -> tuple[np.ndarray, list[str]]:
    """Simulate Triton-style string tensor input formatting before local model encoding."""
    data_arr = [text.encode("UTF-8") for text in texts]
    string_data = np.array(data_arr, dtype=np.bytes_).reshape(len(data_arr), 1)

    input_text = string_data
    input_text = input_text.reshape(-1)
    input_text = input_text.tolist()
    input_text = np.char.decode(input_text, encoding="utf-8")
    triton_input = (
        input_text.tolist() if hasattr(input_text, "tolist") else list(input_text)
    )

    return model.encode(triton_input), triton_input
