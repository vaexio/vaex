__author__ = 'maartenbreddels'
import os
import sys
content = """APPKEY={DB_APPKEY}
APPSECRET={DB_APPSECRET}
ACCESS_LEVEL=sandbox
OAUTH_ACCESS_TOKEN={DB_OAUTH_ACCESS_TOKEN}
OAUTH_ACCESS_TOKEN_SECRET={DB_OAUTH_ACCESS_TOKEN_SECRET}
""".format(**os.environ)
open(sys.argv[1], "w").write(content)
