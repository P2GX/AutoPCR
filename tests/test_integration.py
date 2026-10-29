from pathlib import Path

from autopcr.autopcr import AutoPCR
from autopcr.config import AutoPCRConfig


def test_integration():
    config = AutoPCRConfig(
        working_dir=Path("/Users/rouvenreuter/Documents/Projects/AutoPCR/data/wd"),
        linking_model="qwen3:0.6b"

    )
    aprc = AutoPCR(config)
    aprc.initialize(Path("/Users/rouvenreuter/Downloads/hp (4).obo"), overwrite=False)
    res  = aprc.extract("During the interview the patient displayed marked autistic behaviors, including poor eye contact and rigid, repetitive speech, and also reported ongoing hallucinations that had worsened over the past month.")
    print(res)
