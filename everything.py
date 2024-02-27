import subprocess
from sys import argv
from PIL import Image


#
# TRAINING
#

subprocess.run(['python', 'train.py', '--epochs', '10'])



#
# CHIPPING
#

# example chipper command:
# chipper.py /{path}/ard/ 10 test

#chipper arguments
imgpath = '/Users/peter/work/maxar-open-data/Indonesia-Earthquake22/ard/'
chips = 10
outputdir = 'test'

# do it
subprocess.run(['python', 'chipper.py', imgpath, str(chips), outputdir])



#
# RIPPLING
#

#example apply_ripple command:
# apply_ripple.py weights/gen-r5-322.pts chips/12.pt ok.tiff

# apply_ripple arguments
weightpath = 'weights/space_heater-gen-40.pt'
chippath = 'chips/12.pt'
outputfile = 'ok.tiff'

# do it
subprocess.run(['python', 'apply_ripple.py', weightpath, chippath, outputfile])

image = Image.open(outputfile)
image.show()