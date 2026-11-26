"""Collect the camera transport libraries used by pypylon on Windows."""

from pathlib import Path

from PyInstaller.utils.hooks import get_module_file_attribute


package = Path(get_module_file_attribute("pypylon")).parent
# The SDK loads transport layers from its package directory at runtime.
binaries = [
    (str(path), "pypylon")
    for path in sorted(package.glob("*.dll"))
    if not path.name.startswith(("PylonDataProcessing", "PylonUtilityPcl"))
]
binaries.extend((str(path), "pypylon") for path in package.glob("PylonGigEConnectionGuard.exe"))
