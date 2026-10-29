from pathlib import Path

from autopcr.autopcr import AutoPCR
from autopcr.config import AutoPCRConfig


def test_integration():
    config = AutoPCRConfig(
        working_dir=Path("/Users/rouvenreuter/Documents/Projects/AutoPCR/data/wd")
    )
    aprc = AutoPCR(config)
    aprc.initialize(Path("/Users/rouvenreuter/Downloads/hp (4).obo"), overwrite=True)
