from Modules.FileManagers.FileManager import FileManager as FM
import subprocess

fm_obj = FM()
#fm_obj.downloadAnnotationData('LabeledVideos')

subprocess.run(['python3', 'Modules/MachineLearning/3d_resnet.py', '--data', fm_obj.localOrganizedLabeledClipsDir, '--results', fm_obj.local3DVideosDir])


