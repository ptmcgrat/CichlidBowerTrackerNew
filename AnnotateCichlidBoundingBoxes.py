import argparse, os

from Modules.FileManagers.FileManager import FileManager as FM
from Modules.Annotations.ObjectLabeler import ObjectLabeler as OL

import pandas as pd

import argparse, os

import pdb, datetime, os, subprocess, argparse, random
import pandas as pd

parser = argparse.ArgumentParser(description='This command runs HMM analysis on a single row of data.')
parser.add_argument('ProjectID', type = str, help = 'ProjectID to analyze')
parser.add_argument('-n', '--Number', type = int, help = 'Limit annotation to x number of frames.')
parser.add_argument('-p', '--Practice', action = 'store_true', help = 'Use if you dont want to save annotations')

args = parser.parse_args()

fm_obj = FM(projectID = args.ProjectID)
fm_obj.createDirectory(fm_obj.localAnalysisDir)
fm_obj.downloadData(fm_obj.localManualLabelFramesDir, tarred = True)
fm_obj.downloadData(fm_obj.localBoxedFishFile)

# Read in annotations and create csv file for all annotations with the same user and projectID
dt = pd.read_csv(fm_obj.localBoxedFishFile)
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
		fm_obj.downloadData(fm_obj.localBoxedFishDir + args.ProjectID, tarred = True)

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

subprocess.run(['rm', '-rf', projFileManager.localMasterDir], stderr = subprocess.PIPE)
