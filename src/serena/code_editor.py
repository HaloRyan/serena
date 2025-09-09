import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from serena.symbol import JetBrainsSymbol, LanguageServerSymbol, LanguageServerSymbolRetriever, PositionInFile, Symbol
from solidlsp import SolidLanguageServer
from solidlsp.ls import LSPFileBuffer
from solidlsp.ls_utils import TextUtils

from .project import Project
from .tools.jetbrains_plugin_client import JetBrainsPluginClient

if TYPE_CHECKING:
    from .agent import SerenaAgent


log = logging.getLogger(__name__)
TSymbol = TypeVar("TSymbol", bound=Symbol)


class CodeEditor(Generic[TSymbol], ABC):
    def __init__(self, project_root: str, agent: Optional["SerenaAgent"] = None) -> None:
        self.project_root = project_root
        self.agent = agent

    class EditedFile(ABC):
        @abstractmethod
        def get_contents(self) -> str:
            """
            :return: the contents of the file.
            """

        @abstractmethod
        def delete_text_between_positions(self, start_pos: PositionInFile, end_pos: PositionInFile) -> None:
            pass

        @abstractmethod
        def insert_text_at_position(self, pos: PositionInFile, text: str) -> None:
            pass

    @contextmanager
    def _open_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        """
        Context manager for opening a file
        """
        raise NotImplementedError("This method must be overridden for each subclass")

    @contextmanager
    def _edited_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        """
        Context manager for editing a file.
        """
        with self._open_file_context(relative_path) as edited_file:
            yield edited_file
            # save the file
            abs_path = os.path.join(self.project_root, relative_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(edited_file.get_contents())
            # notify agent (if provided)
            if self.agent is not None:
                self.agent.mark_file_modified(relative_path)

    @abstractmethod
    def _find_unique_symbol(self, name_path: str, relative_file_path: str) -> TSymbol:
        """
        Finds the unique symbol with the given name in the given file.
        If no such symbol exists, raises a ValueError.

        :param name_path: the name path
        :param relative_file_path: the relative path of the file in which to search for the symbol.
        :return: the unique symbol
        """


class LanguageServerCodeEditor(CodeEditor[LanguageServerSymbol]):
    def __init__(self, symbol_retriever: LanguageServerSymbolRetriever, agent: Optional["SerenaAgent"] = None):
        super().__init__(project_root=symbol_retriever.get_language_server().repository_root_path, agent=agent)
        self._symbol_retriever = symbol_retriever

    @property
    def _lang_server(self) -> SolidLanguageServer:
        return self._symbol_retriever.get_language_server()

    class EditedFile(CodeEditor.EditedFile):
        def __init__(self, lang_server: SolidLanguageServer, relative_path: str, file_buffer: LSPFileBuffer):
            self._lang_server = lang_server
            self._relative_path = relative_path
            self._file_buffer = file_buffer

        def get_contents(self) -> str:
            return self._file_buffer.contents

        def delete_text_between_positions(self, start_pos: PositionInFile, end_pos: PositionInFile) -> None:
            self._lang_server.delete_text_between_positions(self._relative_path, start_pos.to_lsp_position(), end_pos.to_lsp_position())

        def insert_text_at_position(self, pos: PositionInFile, text: str) -> None:
            self._lang_server.insert_text_at_position(self._relative_path, pos.line, pos.col, text)

    @contextmanager
    def _open_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        with self._lang_server.open_file(relative_path) as file_buffer:
            yield self.EditedFile(self._lang_server, relative_path, file_buffer)

    def _get_code_file_content(self, relative_path: str) -> str:
        """Get the content of a file using the language server."""
        return self._lang_server.language_server.retrieve_full_file_content(relative_path)

    def _find_unique_symbol(self, name_path: str, relative_file_path: str) -> LanguageServerSymbol:
        symbol_candidates = self._symbol_retriever.find_by_name(name_path, within_relative_path=relative_file_path)
        if len(symbol_candidates) == 0:
            raise ValueError(f"No symbol with name {name_path} found in file {relative_file_path}")
        if len(symbol_candidates) > 1:
            raise ValueError(
                f"Found multiple {len(symbol_candidates)} symbols with name {name_path} in file {relative_file_path}. "
                "Their locations are: \n " + json.dumps([s.location.to_dict() for s in symbol_candidates], indent=2)
            )
        return symbol_candidates[0]


class JetBrainsCodeEditor(CodeEditor[JetBrainsSymbol]):
    def __init__(self, project: Project, agent: Optional["SerenaAgent"] = None) -> None:
        self._project = project
        super().__init__(project_root=project.project_root, agent=agent)

    class EditedFile(CodeEditor.EditedFile):
        def __init__(self, relative_path: str, project: Project):
            path = os.path.join(project.project_root, relative_path)
            log.info("Editing file: %s", path)
            with open(path, encoding=project.project_config.encoding) as f:
                self._content = f.read()

        def get_contents(self) -> str:
            return self._content

        def delete_text_between_positions(self, start_pos: PositionInFile, end_pos: PositionInFile) -> None:
            self._content, _ = TextUtils.delete_text_between_positions(
                self._content, start_pos.line, start_pos.col, end_pos.line, end_pos.col
            )

        def insert_text_at_position(self, pos: PositionInFile, text: str) -> None:
            self._content, _, _ = TextUtils.insert_text_at_position(self._content, pos.line, pos.col, text)

    @contextmanager
    def _open_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        yield self.EditedFile(relative_path, self._project)

    def _find_unique_symbol(self, name_path: str, relative_file_path: str) -> JetBrainsSymbol:
        with JetBrainsPluginClient.from_project(self._project) as client:
            result = client.find_symbol(name_path, relative_path=relative_file_path, include_body=False, depth=0, include_location=True)
            symbols = result["symbols"]
            if not symbols:
                raise ValueError(f"No symbol with name {name_path} found in file {relative_file_path}")
            if len(symbols) > 1:
                raise ValueError(
                    f"Found multiple {len(symbols)} symbols with name {name_path} in file {relative_file_path}. "
                    "Their locations are: \n " + json.dumps([s["location"] for s in symbols], indent=2)
                )
            return JetBrainsSymbol(symbols[0], self._project)
