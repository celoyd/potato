import subprocess
from sys import argv
from PIL import Image

import sys
print("python version", sys.version)


#
# INITIAL CHIPPING
#

# example chipper command:
# python chipper.py make-chips -a /{path}/ard/ -n 10 -c test
# python chipper.py make-chips -a /Users/peter/work/maxar-open-data/Indonesia-Earthquake22/ard/ -n 10 -c test

#chipper arguments
imgpath = '/Users/peter/work/maxar-open-data/Indonesia-Earthquake22/ard/'
chips = 10
outputdir = 'chips'

### do it
# subprocess.run(['python', 'chipper.py', 'make-chips', '-a', imgpath, '-n', str(chips), '-c', outputdir])



#
# TRAINING
#

### do it
# subprocess.run(['python', 'train.py', '--epochs', '10'])



#
# SUBSEQUENT CHIPPING
#

# example chipper command:
# python chipper.py make-chips -a /{path}/ard/ -n 10 -c test
# python chipper.py make-chips -a /Users/peter/work/maxar-open-data/Indonesia-Earthquake22/ard/ -n 10 -c test

#chipper arguments
imgpath = '/Users/peter/work/maxar-open-data/Indonesia-Earthquake22/ard/'
chips = 10
outputdir = 'test'

### do it
# subprocess.run(['python', 'chipper.py', 'make-chips', '-a', imgpath, '-n', str(chips), '-c', outputdir])



#
# RIPPLING
#

#example apply_ripple command:
# apply_ripple.py weights/gen-r5-322.pts chips/2.pt ok.tiff

# apply_ripple arguments
# weightpath = 'weights/space_heater-gen-40.pt'
weightpath = 'weights/space_heater-gen-8.pt'
chippath = 'chips/8.pt'
outputfile = 'ok.tiff'

# do it
subprocess.run(['python', 'apply_ripple.py', weightpath, chippath, outputfile])

image = Image.open(outputfile)
image.show()