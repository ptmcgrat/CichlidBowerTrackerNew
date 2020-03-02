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
				if row.Sex == 'm':
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='blue',facecolor='none'))
				elif row.Sex == 'f':
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='pink',facecolor='none'))
				else:
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='gray',facecolor='none'))


	annotations = dt[(dt.User == user2) & (dt.Framefile == framefile)]
	if len(annotations) > 0:
		for row in annotations.itertuples():
			if row.Box == row.Box:
				box = eval(row.Box)
				if row.Sex == 'm':
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='blue',facecolor='none', linestyle='--'))
				elif row.Sex == 'f':
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='pink',facecolor='none', linestyle='--'))
				else:
					ax.add_patch(matplotlib.patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1,edgecolor='gray',facecolor='none', linestyle='--'))

	plt.title(framefile)
	plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('User1', type = str, help = 'Which user annotations to compare')
parser.add_argument('User2', type = str, help = 'Which user annotations to compare')
parser.add_argument('-p', '--projects', nargs = '+', type = str, help = 'Filter analysis to the following projects')
parser.add_argument('-l', '--plot', action = 'store_true', help = 'Filter analysis to the following projects')

args = parser.parse_args()

print('Downloading data')
fm_obj = FM()
fm_obj.downloadAnnotationData('BoxedFish')
dt = pd.read_csv(fm_obj.localBoxedFishFile)
if args.projects is not None:
	dt = dt[dt.ProjectID.isin(args.projects)]

print('Done')
all_dt = pd.merge(dt[dt.User == args.User1], dt[dt.User == args.User2], how = 'inner', on = 'Framefile') 
grouped = all_dt.groupby('Framefile').max()
numofframes=pd.pivot_table(grouped, values = 'Nfish_y', columns = ['ProjectID_x'], aggfunc = 'count')
numofagreements=pd.pivot_table(grouped[grouped.Nfish_x == grouped.Nfish_y], values = 'Nfish_y', columns = ['ProjectID_x'], aggfunc = 'count')
numoffish_user1=pd.pivot_table(grouped, values = 'Nfish_y', index = 'Nfish_x', columns = ['ProjectID_x'], aggfunc = 'count')
numoffish_user2=pd.pivot_table(grouped, values = 'Nfish_x', index = 'Nfish_y', columns = ['ProjectID_x'], aggfunc = 'count')

addIOU(all_dt)

#user1_dt = all_dt.groupby(['Framefile','Box_x'])['IOU'].transform(max) == all_dt [['IOU','Nfish_x','Nfish_y','ProjectID_x', 'Sex_x', 'Sex_y']].reset_index()
idx = all_dt.groupby(['Framefile','Box_x'])['IOU'].transform(max) == all_dt['IOU']
user1_dt = all_dt[idx][['IOU','Nfish_x','Nfish_y','ProjectID_x', 'Sex_x', 'Sex_y', 'Framefile']].reset_index()
#pdb.set_trace()
IOU=pd.pivot_table(user1_dt, values = 'IOU', index = 'Nfish_x', columns = ['ProjectID_x'])
Sexes=pd.pivot_table(user1_dt, values = 'IOU', index = ['Sex_x','Sex_y'], columns = ['ProjectID_x'],aggfunc='count')

writer = pd.ExcelWriter("output_file.xlsx", engine='xlsxwriter')
numofframes.to_excel(writer, sheet_name='Num_frames')
numofagreements.to_excel(writer, sheet_name='Num_agree')
numoffish_user1.to_excel(writer, sheet_name='Num_fish_'+args.User1)
numoffish_user2.to_excel(writer, sheet_name='Num_fish_'+args.User2)
IOU.to_excel(writer, sheet_name='IOU_avg')
Sexes.to_excel(writer, sheet_name='Sexes_avg')

writer.save()


sns.boxplot(data = user1_dt, y = 'IOU', x = 'ProjectID_x')
sns.swarmplot(data = user1_dt, y = 'IOU', x = 'ProjectID_x', color = ".25")
plt.tight_layout()
plt.xticks(rotation=90,horizontalalignment="right",fontsize=4.75)

plt.show()

framefiles = all_dt.groupby('Framefile').count().index

if args.plot:
	for frame in framefiles:
		t_dt = all_dt[all_dt.Framefile == frame]
		if t_dt.iloc[0,3] != t_dt.iloc[0,10]:
			plotPhoto(frame, dt, fm_obj, args.User1, args.User2)
		else:
			t_dt = user1_dt[user1_dt.Framefile == frame]
			if t_dt.IOU.min() < 0.5:
				plotPhoto(frame, dt, fm_obj, args.User1, args.User2)
			elif len(t_dt[t_dt.Sex_x != t_dt.Sex_y]) > 0:
				plotPhoto(frame, dt, fm_obj, args.User1, args.User2)


