

from Modules.FileManagers.FileManager import FileManager as FM

import pdb, datetime, os, subprocess, argparse, random, cv2
import pandas as pd

parser = argparse.ArgumentParser(description='This command runs HMM analysis on a single row of data.')
parser.add_argument('ProjectID', type = str, help = 'ProjectID to analyze')
parser.add_argument('-n', '--Number', type = int, help = 'Limit annotation to x number of frames.')
parser.add_argument('-p', '--Practice', action = 'store_true', help = 'Use if you dont want to save annotations')
parser.add_argument('-i', '--Initials', type = str, help = 'Initials to save annotations')

args = parser.parse_args()

if args.Initials is None:
	initials = socket.gethostname()
else:
	initials = args.Initials

fm_obj = FM(projectID = args.ProjectID)
fm_obj.createDirectory(fm_obj.localAnalysisDir)
fm_obj.downloadData(fm_obj.localManualLabelClipsDir, tarred = True)
fm_obj.downloadData(fm_obj.localLabeledClipsFile)

temp_csv = fm_obj.localAnalysisDir + 'NewAnnotations.csv'

# Read in annotations and create csv file for all annotations with the same user and projectID
dt = pd.read_csv(fm_obj.localLabeledClipsFile)
new_dt = pd.DataFrame(columns = dt.columns)
clips = [x for x in os.listdir(fm_obj.localManualLabelClipsDir) if 'ManualLabel.mp4' in x]

categories = ['c','f','p','t','b','m','s','x','o','d','q', 'k']

print("Type 'c': build scoop; 'f': feed scoop; 'p': build spit; 't': feed spit; 'b': build multiple; 'm': feed multiple; 'd': drop sand; s': spawn; 'o': fish other; 'x': nofish other; 'q': quit; 'k': skip")
		
newClips = []
annotatedClips = 0
#shuffle(clips)

for f in clips:

	if not dt.loc[dt.ClipName == f]['ManualLabel'].empty:
		print('Skipping ' + f + ': Label=' + dt.loc[dt.ClipName == f]['ManualLabel'].values[0], file = sys.stderr)
		continue
	
	cap = cv2.VideoCapture(fm_obj.localManualLabelClipsDir + f)
	
	while(True):

		ret, frame = cap.read()
		if not ret:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue
		cv2.imshow("Type 'c': build scoop; 'f': feed scoop; 'p': build spit; 't': feed spit; 'b': build multiple; 'm': feed multiple; 'd': drop sand; s': spawn; 'o': fish other; 'x': nofish other; 'q': quit",cv2.resize(frame,(0,0),fx=4, fy=4))
		info = cv2.waitKey(25)
	
		if info in [ord(x) for x in categories]:
			for i in range(1,10):
				cv2.destroyAllWindows()
				cv2.waitKey(1)
			break

	if info == ord('q'):
		break
	
	if info == ord('k'):
		continue #skip

	new_dt.loc[len(new_dt)] = [f.replace('.mp4',''), chr(info), initials, str(datetime.datetime.now())]

	new_dt.to_csv(temp_csv, sep = ',')

	annotatedClips += 1

	if args.Number is not None and annotatedClips >= args.Number:
		break

pdb.set_trace()


if not args.Practice:
	dt = dt.append(new_dt)
	dt.to_csv(fm_obj.localLabeledClipsFile, sep = ',')
	#fm_obj.uploadData(fm_obj.localLabeledClipsFile)

	fm_obj.downloadData(fm_obj.localLabeledClipsProjectDir, tarred = True)

"""
dt = dt[(dt.ProjectID == args.ProjectID) & (dt.User == os.getenv('USER'))]
dt.to_csv(fm_obj.localLabeledFramesFile, sep = ',', columns = ['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'CorrectAnnotation', 'User', 'DateTime'])

obj = OL(fm_obj.localManualLabelFramesDir, fm_obj.localLabeledFramesFile, args.Number, args.ProjectID)




if not args.Practice:
	# Backup annotations. Redownload to avoid race conditions
	if not os.path.exists(fm_obj.localLabeledFramesFile):
		print(fm_obj.localLabeledFramesFile + 'does not exist. Did you annotate any new frames? Quitting...')
	else:
		# First read in the new annotations
		newAnn_DT = pd.read_csv(fm_obj.localLabeledFramesFile, index_col = 0)

		# Next download the annotations and frames already stored in the annotation database
		fm_obj.downloadData(fm_obj.localBoxedFishFile)
		try:
			fm_obj.downloadData(fm_obj.localBoxedFishDir + args.ProjectID, tarred = True)
		except FileNotFoundError:
			fm_obj.createDirectory(fm_obj.localBoxedFishDir + args.ProjectID)

		# Read in and merge new annotations into annotation csv file
		if os.path.exists(fm_obj.localBoxedFishFile):
			old_DT = pd.read_csv(fm_obj.localBoxedFishFile)
			old_DT = old_DT.append(newAnn_DT, sort=True).drop_duplicates(subset = ['ProjectID', 'Framefile', 'User', 'Sex', 'Box'])
		else:
			print('Annotation database file does not exist yet. Creating')
			old_DT = newAnn_DT

		old_DT.to_csv(fm_obj.localBoxedFishFile, sep = ',', columns = ['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'CorrectAnnotation','User', 'DateTime'])

		# Add newly annotated frames to annotation database
		for row in newAnn_DT.itertuples():
			if not os.path.exists(fm_obj.localBoxedFishDir + args.ProjectID + '/' + row.Framefile):
				output = subprocess.run(['cp', fm_obj.localManualLabelFramesDir + row.Framefile, fm_obj.localBoxedFishDir + args.ProjectID + '/'], stderr = subprocess.PIPE, encoding = 'utf-8')
				if output.stderr != '':
					print(output.stderr)
					raise Exception

		fm_obj.uploadData(fm_obj.localBoxedFishFile)
		fm_obj.uploadData(fm_obj.localBoxedFishDir + args.ProjectID, tarred = True)
else:
	print('Practice mode enabled. Will not store annotations.')

subprocess.run(['rm', '-rf', fm_obj.localProjectDir])
subprocess.run(['rm', '-rf', fm_obj.localAnnotationDir])
"""