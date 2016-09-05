__author__ = 'breddels'
import javaobj
import sys
import io


if __name__ == "__main__":
	import logging
	javaobj._log.setLevel(logging.DEBUG)
	jobj = file(sys.argv[1]).read()[16+5:]
	print((repr(jobj[:100])))
	pobj, index = javaobj.loads(jobj)
	rest = jobj[index:]
	import zlib
	print((repr(rest[:100])))
	datastr = zlib.decompress(rest, -15)
	stream = io.StringIO(datastr)
	m = javaobj.JavaObjectUnmarshaller(stream)
	data = m.readObject()
	while len(data) == 2:
		print((data[0].classdesc.name))
		#datastr = datastr[data[1]:]
		x = stream.read(3)
		print(("data left", [hex(ord(k)) for k in x]))
		stream.seek(-3, 1)
		#x = stream.read(10)
		#print "data left", [hex(ord(k)) for k in x]
		#stream.seek(-10, 1)
		#data = javaobj.loads(datastr)
		data = m.readObject()
		print(data)
		if data[0] == "END":
			print("end...")
			x = stream.read(10)
			i
			print(("data left", [hex(ord(k)) for k in x]))

	print(data)
	#print pobj, index