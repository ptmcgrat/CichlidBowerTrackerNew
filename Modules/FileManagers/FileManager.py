import os, subprocess, pdb, platform
import pandas as pd

class FileManager():
	def __init__(self, projectID = None):

		# Identify directory for temporary local files
		if platform.node() == 'raspberrypi' or 'Pi' in platform.node():
			self._identifyPiDirectory()
		else:
			self.localMasterDir = os.getenv('HOME') + '/' + 'Temp/CichlidAnalyzer/'

		# Identify cloud directory for rclone
		self.rcloneRemote = 'cichlidVideo:'
		# On some computers, the first directory is McGrath, on others it's BioSci-McGrath. Use rclone to figure out which
		output = subprocess.run(['rclone', 'lsf', self.rcloneRemote], capture_output = True, encoding = 'utf-8')
		if 'McGrath/' in output.stdout.split():
			self.cloudMasterDir = self.rcloneRemote + 'McGrath/Apps/CichlidPiData/'
		elif 'BioSci-McGrath/' in output.stdout.split():
			self.cloudMasterDir = self.rcloneRemote + 'BioSci-McGrath/Apps/CichlidPiData/'
		else:
			raise Exception('Cant find master McGrath directory in rclone remote')

		if projectID is not None:
			self.createProjectData(projectID)

		# Create file names 
		self.createPiData()
		self.createMLData()
		self.createAnnotationData()

	def createPiData(self):
		self.localCredentialSpreadsheet = self.localMasterDir + '__CredentialFiles/SAcredentials.json'
		self.localCredentialDrive = self.localMasterDir + '__CredentialFiles/DriveCredentials.txt'

	def createProjectData(self, projectID):
		self.projectID = projectID
		self.localProjectDir = self.localMasterDir + projectID + '/'

		# Create logfile
		self.localLogfile = self.localProjectDir + 'Logfile.txt'

		# Data directories created by tracker
		self.localPrepDir = self.localProjectDir + 'PrepFiles/'
		self.localFrameDir = self.localProjectDir + 'Frames/'
		self.localVideoDir = self.localProjectDir + 'Videos/'
		self.localBackupDir = self.localProjectDir + 'Backups/'

		# Directories created by analysis
		self.localAnalysisDir = self.localProjectDir + 'MasterAnalysisFiles/'
		self.localFiguresDir = self.localProjectDir + 'Figures/'
		self.localAllClipsDir = self.localProjectDir + 'AllClips/'
		self.localManualLabelClipsDir = self.localProjectDir + 'MLClips/'
		self.localManualLabelFramesDir = self.localProjectDir + 'MLFrames/'
		self.localTroubleshootingDir = self.localProjectDir + 'Troubleshooting/'
		self.localTempDir = self.localProjectDir + 'Temp/'

		# Files created by tracker
		self.localFirstFrame = self.localPrepDir + 'FirstDepth.npy'
		self.localLastFrame = self.localPrepDir + 'LastDepth.npy'
		self.localPiRGB = self.localPrepDir + 'PiCameraRGB.jpg'
		self.localDepthRGB = self.localPrepDir + 'DepthRGB.jpg'

		# Files created by prep preparer
		self.localTrayFile = self.localAnalysisDir + 'DepthCrop.txt'
		self.localTransMFile = self.localAnalysisDir + 'TransMFile.npy'
		self.localVideoCropFile = self.localAnalysisDir + 'VideoCrop.npy'
		self.localVideoPointsFile = self.localAnalysisDir + 'VideoPoints.npy'
		self.localPrepSummaryFigure = self.localFiguresDir + 'PrepSummary.pdf' 

		# Files created by depth preparer
		self.localSmoothDepthFile = self.localAnalysisDir + 'smoothedDepthData.npy'
		self.localRGBDepthVideo = self.localAnalysisDir + 'DepthRGBVideo.mp4'
		self.localRawDepthFile = self.localTroubleshootingDir + 'rawDepthData.npy'
		self.localInterpDepthFile = self.localTroubleshootingDir + 'interpDepthData.npy'

		# Files created by cluster preparer
		self.localAllLabeledClustersFile = self.localAnalysisDir + 'AllLabeledClusters.csv'

		# Files created by manual labeler preparer
		self.localLabeledFramesFile = self.localAnalysisDir + 'LabeledFrames.csv'

		# Files created by manual labeler preparer

	def createMLData(self):
		self.localMLDir = self.localMasterDir + '__MachineLearningModels/'
		self.localVideoModelFile = self.localMLDir + 'MasterModels.txt'

		self.downloadData(self.localVideoModelFile)
		with open(self.localVideoModelFile) as f:
			line = next(f)
			line = next(f)
			self.vModelID = line.rstrip().split(',')[1]

		self.local3DModelDir = self.localMLDir + 'VideoModels/' + self.vModelID + '/'

		self.videoMLGithub = 'https://www.github.com/ptmcgrat/3D-Resnets'

		self.localVideoModelFile = self.local3DModelDir + 'model.pth'

		self.localVideoClassesFile = self.local3DModelDir + 'classInd.txt'
		self.localVideoCommandsFile = self.local3DModelDir + 'commands.pkl'

	def createAnnotationData(self):
		self.localAnnotationDir = self.localMasterDir + '__AnnotatedData/'
		self.localObjectDetectionDir = self.localAnnotationDir + 'BoxedFish/'
		self.local3DVideosDir = self.localAnnotationDir + 'LabeledVideos/'

		self.localLabeledClipsFile = self.local3DVideosDir + 'ManualLabels.csv'
		self.localLabeledClipsDir = self.local3DVideosDir + 'Clips/'
		self.localOrganizedLabeledClipsDir = self.local3DVideosDir + 'OrganizedClips/'

		self.localBoxedFishFile = self.localObjectDetectionDir + 'BoxedFish.csv'
		self.localBoxedFishDir = self.localObjectDetectionDir + 'BoxedImages/'

	def downloadProjectData(self, dtype):

		if dtype == 'Prep':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localFiguresDir)
			self.downloadData(self.localPrepDir)
			self.downloadData(self.localLogfile)

		elif dtype == 'Depth':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localFrameDir, tarred = True)

		elif dtype == 'Cluster':
			self.createMLData()
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localTempDir)
			self.createDirectory(self.localAllClipsDir)
			self.createDirectory(self.localManualLabelClipsDir)
			self.createDirectory(self.localManualLabelFramesDir)
			self.createDirectory(self.localManualLabelFramesDir[:-1] + '_pngs')
			self.downloadData(self.localLogfile)
			self.downloadData(self.localVideoDir)

		elif dtype == 'MLClassification':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.local3DModelDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localAllClipsDir, tarred = True)
			self.downloadData(self.localAnalysisDir)
			self.downloadData(self.local3DModelDir)

		elif dtype == 'ObjectLabeler':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.downloadData(self.manualLabelFramesDir, tarred = True)
			try:
				self.downloadData(self.localLabeledFramesFile)
			except FileNotFoundError:
				pass

		elif dtype == 'Figures':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localFiguresDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localAnalysisDir)

		elif dtype == 'All':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.local3DModelDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localTempDir)
			self.createDirectory(self.localAllClipsDir)
			self.createDirectory(self.localManualLabelClipsDir)
			self.createDirectory(self.localManualLabelFramesDir)
			self.createDirectory(self.localManualLabelFramesDir[:-1] + '_pngs')
			self.downloadData(self.localLogfile)
			self.downloadData(self.localVideoDir)
			self.downloadData(self.localFrameDir, tarred = True)
			self.downloadData(self.local3DModelDir)

		else:
			raise KeyError('Unknown key: ' + dtype)

	def downloadAnnotationData(self, dtype):
		if dtype == 'LabeledVideos':
			good_count, bad_count = 0,0

			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.local3DVideosDir)
			self.downloadData(self.localLabeledClipsFile)

			# Identify directories that need to be untarred and download them
			labeledProjects = subprocess.run(['rclone', 'lsf', self.localLabeledClipsDir.replace(self.localMasterDir, self.cloudMasterDir)], capture_output = True, encoding = 'utf-8').stdout.split()
			for lp in labeledProjects:
				if '.tar' in lp:
					self.downloadData(self.localLabeledClipsDir + lp.replace('.tar',''), tarred = True)

			# Reorganize data for VideoLoader to automatically download
			dt = pd.read_csv(self.localLabeledClipsFile)
			for row in dt.itertuples():
				dest_dir = self.localOrganizedLabeledClipsDir + row.ManualLabel + '/'
				if not os.path.exists(dest_dir):
					self.createDirectory(dest_dir)
				output = subprocess.run(['mv', self.localLabeledClipsDir + row.ClipName.split('__')[0] + '/' + row.ClipName + '.mp4', dest_dir], capture_output = True, encoding = 'utf-8')
				if output.stderr == '':
					good_count += 1
				else:
					bad_count += 1
			print(str(good_count) + ' labeled videos moved. Missing videos for ' + str(bad_count) + ' total videos.')
			subprocess.call(['rm', '-rf', self.localLabeledClipsDir])

		elif dtype == 'BoxedFish':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localObjectDetectionDir)
			self.downloadData(self.localBoxedFishFile)
			
			boxedProjects = subprocess.run(['rclone', 'lsf', self.localBoxedFishDir.replace(self.localMasterDir, self.cloudMasterDir)], capture_output = True, encoding = 'utf-8').stdout.split()
			for bp in boxedProjects:
				if '.tar' in bp:
					self.downloadData(self.localBoxedFishDir + bp.replace('.tar',''), tarred = True)
		else:
			raise KeyError('Unknown key: ' + dtype)

	def returnVideoObject(self, index):
		from Modules.DataObjects.LogParser import LogParser as LP

		self.downloadData(self.localLogfile)
		self.lp = LP(self.localLogfile)
		videoObj = self.lp.movies[index]
		videoObj.localVideoFile = self.localMasterDir + videoObj.mp4_file
		videoObj.localHMMFile = self.localTroubleshootingDir + videoObj.baseName + '.hmm'
		videoObj.localRawCoordsFile = self.localTroubleshootingDir + videoObj.baseName + '_rawCoords.npy'
		videoObj.localLabeledCoordsFile = self.localTroubleshootingDir + videoObj.baseName + '_labeledCoords.npy'
		videoObj.localLabeledClustersFile = self.localTroubleshootingDir + videoObj.baseName + '_labeledClusters.csv'
		videoObj.localAllClipsPrefix = self.localAllClipsDir + self.lp.projectID + '_' + videoObj.baseName
		videoObj.localManualLabelClipsPrefix = self.localManualLabelClipsDir + self.lp.projectID + '_' + videoObj.baseName
		videoObj.localIntensityFile = self.localFiguresDir + videoObj.baseName + '_intensity.pdf'
		videoObj.localTempDir = self.localTempDir + videoObj.baseName + '/'
		videoObj.nManualLabelClips = int(self.nManualLabelClips/len(self.lp.movies))
		videoObj.nManualLabelFrames = int(self.nManualLabelFrames/len(self.lp.movies))
		
		self.createDirectory(videoObj.localTempDir)

		return videoObj

	def _createParameters(self):

		# Depth related parameters
		self.hourlyThreshold = 0.2
		self.dailyThreshold = 0.4
		self.totalThreshold = 1.0
		self.hourlyMinPixels = 1000
		self.dailyMinPixels = 1000
		self.totalMinPixels = 1000
		self.pixelLength = 0.1030168618 # cm / pixel
		self.bowerIndexFraction = 0.1

		# Video related parameters
		self.lightsOnTime = 8
		self.lightsOffTime = 18

		# DB Scan related parameters
		self.minMagnitude = 0
		self.treeR = 22 
		self.leafNum = 190 
		self.neighborR = 22
		self.timeScale = 10
		self.eps = 18
		self.minPts = 90 
		self.delta = 1.0 # Batches to calculate clusters

		# Clip creation parameters
		self.nManualLabelClips = 1200
		self.delta_xy = 100
		self.delta_t = 60
		self.smallLimit = 500

		# Manual Label Frame 
		self.nManualLabelFrames = 500

	def _identifyPiDirectory(self):
		writableDirs = []
		try:
			possibleDirs = os.listdir('/media/pi')
		except FileNotFoundError:
			return

		for d in possibleDirs:

			try:
				with open('/media/pi/' + d + '/temp.txt', 'w') as f:
					print('Test', file = f)
				with open('/media/pi/' + d + '/temp.txt', 'r') as f:
					for line in f:
						if 'Test' in line:
							writableDirs.append(d)
			except:
				pass
			try:
				os.remove('/media/pi/' + d + '/temp.txt')
			except FileNotFoundError:
				continue
		
		if len(writableDirs) == 1:
			self.localMasterDir = '/media/pi/' + d + '/CichlidAnalyzer/'
			self.system = 'pi'
		elif len(writableDirs) == 0:
			raise Exception('No writable drives in /media/pi/')
		else:
			raise Exception('Multiple writable drives in /media/pi/. Options are: ' + str(writableDirs))

	def createDirectory(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def downloadData(self, local_data, tarred = False):

		relative_name = local_data.rstrip('/').split('/')[-1] + '.tar' if tarred else local_data.rstrip('/').split('/')[-1]
		local_path = local_data.split(local_data.rstrip('/').split('/')[-1])[0]
		cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

		cloud_objects = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8').stdout.split()

		if relative_name + '/' in cloud_objects: #directory
			output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path + relative_name], capture_output = True, encoding = 'utf-8')
		elif relative_name in cloud_objects: #file
			output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path], capture_output = True, encoding = 'utf-8')
		else:
			raise FileNotFoundError('Cant find file for download: ' + cloud_path + relative_name)

		if not os.path.exists(local_path + relative_name):
			raise FileNotFoundError('Error downloading: ' + local_path + relative_name)

		if tarred:
			# Untar directory
			output = subprocess.run(['tar', '-xvf', local_path + relative_name, '-C', local_path], capture_output = True, encoding = 'utf-8')
			output = subprocess.run(['rm', '-f', local_path + relative_name], capture_output = True, encoding = 'utf-8')

	def uploadData(self, local_data, tarred = False):

		relative_name = local_data.rstrip('/').split('/')[-1]
		local_path = local_data.split(relative_name)[0]
		cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

		if tarred:
			output = subprocess.run(['tar', '-cvf', local_path + relative_name + '.tar', '-C', local_path, relative_name], capture_output = True, encoding = 'utf-8')
			if output.returncode != 0:
				print(output.stderr)
				raise Exception('Error in tarring ' + local_data)
			relative_name += '.tar'

		if os.path.isdir(local_path + relative_name):
			output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path + relative_name], capture_output = True, encoding = 'utf-8')
			subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path + relative_name], check = True)

		elif os.path.isfile(local_path + relative_name):
			print(['rclone', 'copy', local_path + relative_name, cloud_path])
			output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path], capture_output = True, encoding = 'utf-8')
			output = subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path], check = True, capture_output = True, encoding = 'utf-8')
		else:
			raise Exception(local_data + ' does not exist for upload')

		if output.returncode != 0:
			pdb.set_trace()
			raise Exception('Error in uploading file: ' + output.stderr)


	