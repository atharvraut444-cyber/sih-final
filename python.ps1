# Local Python shim for PowerShell — delegates to py -3 (Python 3.13)
# Allows running .\python or python from PowerShell without PATH conflicts
& py -3 $args
