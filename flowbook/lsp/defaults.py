from pathlib import Path


def default_initialize(root: Path):
    root_uri = root.as_uri()
    # print(root_uri)
    return {
        "processId": None,
        "rootUri": root_uri,
        "capabilities": {
            "workspace": {
                "workspaceFolders": True,
                "didChangeConfiguration": {"dynamicRegistration": False},
                "didChangeWatchedFiles": {"dynamicRegistration": False},
                "symbol": {"dynamicRegistration": False},
            },
            "textDocument": {
                "synchronization": {
                    "didSave": True,
                    "willSave": False,
                    "dynamicRegistration": False,
                },
                "completion": {"dynamicRegistration": False},
                "hover": {"dynamicRegistration": False},
                "definition": {"dynamicRegistration": False},
                "typeDefinition": {"dynamicRegistration": False},
                "implementation": {"dynamicRegistration": False},
                "references": {"dynamicRegistration": False},
                "publishDiagnostics": {"relatedInformation": True},
            },
            "window": {"workDoneProgress": False},
            "general": {"positionEncodings": ["utf-16"]},
        },
        "workspaceFolders": [{"uri": root_uri, "name": root.name}],
        "trace": "messages",
    }
