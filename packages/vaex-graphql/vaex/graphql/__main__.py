import sys
import vaex
from tornado.ioloop import IOLoop

df = vaex.open(sys.argv[1])
df.graphql.serve()
IOLoop.instance().start()
