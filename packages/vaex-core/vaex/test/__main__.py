__author__ = 'maartenbreddels'
import unittest
import vaex.test.dataset
def main(args=None):
	if args is None or len(args) == 1:
		args = ["vaex", "vaex.test"]
	unittest.main(argv=args)

if __name__ == "__main__":
	main()
