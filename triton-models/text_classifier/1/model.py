import hashlib

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            values = tensor.as_numpy().reshape(-1)

            labels = []
            scores = []
            for value in values:
                text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                score = int.from_bytes(digest, "big") / (2**64 - 1)
                scores.append(score)
                labels.append("review" if score >= 0.62 else "pass")

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "LABEL",
                            np.asarray(labels, dtype=object).reshape(-1, 1),
                        ),
                        pb_utils.Tensor(
                            "SCORE",
                            np.asarray(scores, dtype=np.float32).reshape(-1, 1),
                        ),
                    ]
                )
            )
        return responses
