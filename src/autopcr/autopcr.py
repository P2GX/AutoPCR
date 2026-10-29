from autopcr._build_index import IndexBuilder

from pathlib import Path

from nltk import WordNetLemmatizer

from autopcr._extractor import Extractor
from autopcr._macos import limit_openmp
from autopcr._ontology_builder import OntologyBuilder
from autopcr._working_files import ALL_FILES
from autopcr.config import AutoPCRConfig
from autopcr.hit import Hit


class UninitializedError(RuntimeError):
    pass


class AutoPCR:
    def __init__(
        self,
        config: AutoPCRConfig,
    ):
        limit_openmp()
        self._work_dir = config.working_dir
        self._extractor_config = config
        self._extractor: Extractor | None = None

    @property
    def is_initialized(self) -> bool:
        return self._extractor is not None

    def initialize(
        self,
        hpo_file: Path,
        overwrite: bool = False,
        root_node: str = "HP:0000118",
    ):
        if overwrite or not all(Path(f).exists() for f in ALL_FILES):
            ontology_builder = OntologyBuilder(hpo_file, root_node, WordNetLemmatizer())
            ontology_builder.build(self._work_dir)
            index_builder = IndexBuilder()
            index_builder.build_index(self._work_dir)

        self._extractor = Extractor(self._extractor_config)

    def extract(self, text: str) -> list[Hit]:
        if self._extractor is None:
            raise UninitializedError("Call initialize() first")
        return self._extractor.extract(text)
