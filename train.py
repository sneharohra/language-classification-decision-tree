import sys
import  decisiontree
import adaboost

if __name__ == '__main__':
    if sys.argv[3] == "dt":
        decisiontree.main(sys.argv[1], sys.argv[2],True)
    else:
        adaboost.main(sys.argv[1], sys.argv[2])