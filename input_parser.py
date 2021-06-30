import argparse

# TODO set defaults visable, clean up names, other?

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
     description=('''\
         Trajectory picking algorithm for the AI for Maturation project
         --------------------------------------------------------------
         More text here? Maybe, if needed there will be. Maybe...'''),epilog='This is the epilog',add_help=True)

parser.add_argument('-alg',default='greedy',type=str,choices=['greedy','random_choice','bipartite','max_ind_set'],help='Type of algorithm used (default: greedy)',)
parser.add_argument("inputfile",metavar='Inputfile',type=str,help='Filename string of the input data set, JSON format')
parser.add_argument('-outputfile',metavar='Outputfile',type=str,default='trajectories.txt' ,help='Filename string of output data result, JSON format')
args = parser.parse_args()
print(args)
print(args.inputfile)
print(args.outputfile)
print(args.alg)