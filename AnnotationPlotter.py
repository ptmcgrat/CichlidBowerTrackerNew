import argparse, matplotlib, sys, pdb
import pandas as pd
import numpy as np
from Modules.FileManagers.FileManager import FileManager as FM
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.image as mpimg

def makePrediction(image):
	import torchvision
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model.eval()
	from PIL import Image
	import torchvision.transforms.functional as TF

	image = Image.open(image)
	x = TF.to_tensor(image)
	x.unsqueeze_(0)
	out = model(x)

	plt.imshow(image)
	ax = plt.gca()

	for box in out[0]['boxes']:
		ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2]-box[0], box[3] - box[1], linewidth=1,edgecolor='g',facecolor='none'))
	
	print(out)
	plt.show()

def addIOU(dt):
	ious = []
	for row in dt.itertuples():
		ann1 = row.Box_x
		ann2 = row.Box_y

		if ann1 != ann1 and ann2!=ann2: # no annotations for both. IOU should not be calculated
			ious.append(np.nan)
		elif ann1 != ann1 or ann2!=ann2: # no annotation for one. Set IOU to zero
			ious.append(0)
		else: # annotations for both
			ann1 = eval(ann1)
			ann2 = eval(ann2)

			overlap_x0, overlap_y0, overlap_x1, overlap_y1 = max(ann1[0],ann2[0]), max(ann1[1],ann2[1]), min(ann1[0] + ann1[2],ann2[0] + ann2[2]), min(ann1[1] + ann1[3],ann2[1] + ann2[3])
			if overlap_x1 < overlap_x0 or overlap_y1 < overlap_y0:
				ious.append(0)
			else:
				intersection = (overlap_x1 - overlap_x0)*(overlap_y1 - overlap_y0)
				union = ann1[2]*ann1[3] + ann2[2]*ann2[3] - intersection
				ious.append(intersection/union)

	dt['IOU'] = pd.Series(ious)

def plotPhoto(framefile, dt, fm_obj, user1, user2):
	projectID = framefile.split('_' + framefile.split('_')[-3] + '_vid')[0]
	img = mpimg.imread(fm_obj.localBoxedFishDir + projectID + '/' + framefile)
	plt.imshow(img)
	ax = plt.gca()
	
	annotations = dt[(dt.User == user1) & (dt.Framefile == framefile)]
	if len(annotations) > 0:
		for row in annotations.itertuples():
			if row.Box == row.Box:
				box = eval(row.Box)
				ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='g',facecolor='none'))
	annotations = dt[(dt.User == user2) & (dt.Framefile == framefile)]
	if len(annotations) > 0:
		for row in annotations.itertuples():
			if row.Box == row.Box:
				box = eval(row.Box)
				ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='r',facecolor='none'))

	plt.title(framefile)
	plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('User1', type = str, help = 'Which user annotations to compare')
parser.add_argument('User2', type = str, help = 'Which user annotations to compare')

args = parser.parse_args()

fm_obj = FM()
fm_obj.downloadAnnotationData('BoxedFish')
dt = pd.read_csv(fm_obj.localBoxedFishFile)

all_dt = pd.merge(dt[dt.User == args.User1], dt[dt.User == args.User2], how = 'inner', on = 'Framefile') 

print('Number of frames: ' + str(len(all_dt.groupby('Framefile'))))
print('Number of frames with agreements: ' + str(len(all_dt[all_dt.Nfish_x == all_dt.Nfish_y].groupby('Framefile'))))
print('Number of fish per frame for ' + args.User1)
print(all_dt.groupby('Framefile').max().groupby('Nfish_x').count()['User_x'])
print('Number of fish per frame for ' + args.User2)
print(all_dt.groupby('Framefile').max().groupby('Nfish_y').count()['User_y'])
addIOU(all_dt)

user1_dt = all_dt.groupby(['Framefile','Box_x']).max()[['IOU','Nfish_x']].reset_index()

print('Average IOU by number of fish:')
print(user1_dt.groupby('Nfish_x').mean())

framefiles = all_dt.groupby('Framefile').count().index


for frame in framefiles:
	t_dt = all_dt[all_dt.Framefile == frame]
	if t_dt.iloc[0,3] != t_dt.iloc[0,10]:
		projectID = frame.split('_' + frame.split('_')[-3] + '_vid')[0]
	#makePrediction(fm_obj.localBoxedFishDir + projectID + '/' + frame)
		#plotPhoto(frame, dt, fm_obj, args.User1, args.User2)
	else:
		t_dt = user1_dt[user1_dt.Framefile == frame]
		if t_dt.IOU.min() < 0.5:
			plotPhoto(frame, dt, fm_obj, args.User1, args.User2)
