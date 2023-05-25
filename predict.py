import sys
import importlib


def main():
    a = (sys.argv[1].split('.')[0])
    c = importlib.import_module(a)
    c.main(sys.argv[2])


if __name__ == '__main__':
    main()
