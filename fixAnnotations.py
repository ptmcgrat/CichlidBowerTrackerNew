import argparse, subprocess
from Modules.FileManagers.FileManager import FileManager as FM
from Modules.Annotations.ObjectLabeler import AnnotationDisagreements as AD
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('User1', type = str, help = 'Which user annotations to compare')
parser.add_argument('User2', type = str, help = 'Which user annotations to compare')
parser.add_argument('ProjectID', type = str, help = 'Project to analyze')
parser.add_argument('-p', '--Practice', action = 'store_true', help = 'Use if you dont want to save annotations')


args = parser.parse_args()

fm_obj = FM(projectID = args.ProjectID)
fm_obj.createDirectory(fm_obj.localAnalysisDir)
fm_obj.downloadData(fm_obj.localBoxedFishDir + args.ProjectID, tarred = True)
fm_obj.downloadData(fm_obj.localBoxedFishFile)

temp_dt = fm_obj.localBoxedFishFile.replace('.csv', '.bu.csv')
subprocess.run(['cp', fm_obj.localBoxedFishFile, temp_dt])

ad_obj = AD(fm_obj.localBoxedFishDir + args.ProjectID + '/', temp_dt, args.ProjectID, args.User1, args.User2)

# Redownload csv in case new annotations have been added
fm_obj.downloadData(fm_obj.localBoxedFishFile)

old_dt = pd.read_csv(fm_obj.localBoxedFishFile, index_col = 0)
new_dt = pd.read_csv(temp_dt)

old_dt = old_dt.append(new_dt, sort = 'False').drop_duplicates(subset = ['ProjectID', 'Framefile', 'User', 'Sex', 'Box'], keep = 'last').sort_values(by = ['ProjectID', 'Framefile'])
old_dt.to_csv(fm_obj.localBoxedFishFile, sep = ',', columns = ['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'CorrectAnnotation','User', 'DateTime'])

if not args.Practice:
	fm_obj.uploadData(fm_obj.localBoxedFishFile)

subprocess.run(['rm', '-rf', self.localProjectDir])
subprocess.run(['rm', '-rf', self.localAnnotationDir])

