import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
     description=('''\
         Trajectory picking algorithm for the AI for Maturation project
         --------------------------------------------------------------
         More text here? Maybe, if needed there will be. Maybe...'''),epilog='This is the epilog',add_help=True)

parser.add_argument('--algorithm',default='greedy',type=str,choices=['greedy','bipartite','max_ind_set'],help='Type of algorithm used (default: greedy)',)
parser.add_argument('inputdata_filename',metavar='Input_filename',type=str,help='Filename string of the input data set, JSON format')
parser.add_argument('outputdata_filename',metavar='Output_filename',type=str,help='Filename string of output data result, JSON format')

args = parser.parse_args()
# print(args)

print(args[0])