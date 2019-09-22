import argparse
import sys

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data/', help='Path for input images')
    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' , help='Path for Output images')
    
    values = parser.parse_args()
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    print(str(values))