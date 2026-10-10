"""Allow ``python -m mcp_server``."""

import sys

from mcp_server.server import main

sys.exit(main())
