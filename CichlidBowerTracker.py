import argparse, os, pdb, sys, subprocess
from Modules.Tracking.CichlidTracker import CichlidTracker as CT

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

trackerParser = subparsers.add_parser('CollectData', help='This command runs on Raspberry Pis to collect depth and RGB data')

summarizeParser = subparsers.add_parser('UpdateAnalysisSummary', help='This command identifies any new projects that can be analyzed and merges any updates that are new')

downloadParser = subparsers.add_parser('DownloadData', help='This command is used to download data')
downloadParser.add_argument('AnalysisType', type = str, choices=['All','Prep','Depth','Cluster','ObjectLabeler', 'Figures'], help = 'What type of analysis data do you want to download')
downloadParser.add_argument('ProjectID', type = str, help = 'Which projectID you want to identify')

prepParser = subparsers.add_parser('ManualPrep', help='This command takes user interaction to identify depth crops, RGB crops, and register images')
prepParser.add_argument('-p', '--ProjectIDs', nargs = '+', required = True, type = str, help = 'Manually identify the projects you want to analyze. If All is specified, all non-prepped projects will be analyzed')

projectParser = subparsers.add_parser('ProjectAnalysis', help='This command performs a single type of analysis of the project. It is meant to be chained together to perform the entire analysis')
projectParser.add_argument('AnalysisType', type = str, choices=['Depth','Cluster','CreateFrames','MLClassification', 'MLFishDetection','Figures','Backup'], help = 'What type of analysis to perform')
projectParser.add_argument('ProjectID', type = str, help = 'Which projectID you want to identify')
projectParser.add_argument('-w', '--Workers', type = int, help = 'Use if you want to control how many workers this analysis uses', default = 1)
projectParser.add_argument('-g', '--GPUs', type = int, help = 'Use if you want to control how many GPUs this analysis uses', default = 1)
projectParser.add_argument('-v', '--VideoIndex', type = int, help = 'Restrict cluster analysis to single video')

totalProjectsParser = subparsers.add_parser('TotalProjectAnalysis', help='This command runs the entire pipeline on list of projectIDs')
totalProjectsParser.add_argument('Computer', type = str, choices=['NURF','SRG','PACE'], help = 'What computer are you running this analysis from?')
totalProjectsParser.add_argument('-p', '--ProjectIDs', nargs = '+', required = True, type = str, help = 'Manually identify the projects you want to analyze. If All is specified, all non-prepped projects will be analyzed')
totalProjectsParser.add_argument('-w', '--Workers', type = int, help = 'Use if you want to control how many workers this analysis uses', default = 1)

args = parser.parse_args()

if args.command is None:
	parser.print_help()

if args.command == 'CollectData':
	CT()

if args.command == 'UpdateAnalysisSummary':
	
	ap_obj = AP()
	ap_obj.updateAnalysisFile()

elif args.command == 'DownloadData':
	pp_obj = PP(args.ProjectID, 1)
	pp_obj.downloadData(args.AnalysisType)

elif args.command == 'ManualPrep':
	
	ap_obj = AP()
	if ap_obj.checkProjects(args.ProjectIDs):
		sys.exit()

	for projectID in args.ProjectIDs:
		pp_obj = PP(projectID, args.Workers)
		pp_obj.runPrepAnalysis()
		
	#pp_obj.backupAnalysis()
	ap_obj.updateAnalysisFile(newProjects = False, projectSummary = False)

elif args.command == 'ProjectAnalysis':


	args.ProjectIDs = args.ProjectID # format that parseProjects expects

	pp_obj = PP(args.ProjectID, args.Workers)

	if args.AnalysisType == 'Download' or args.DownloadOnly:
		pp_obj.downloadData(args.AnalysisType)

	elif args.AnalysisType == 'Depth':
		pp_obj.runDepthAnalysis()

	elif args.AnalysisType == 'Cluster':
		pp_obj.runClusterAnalysis(args.VideoIndex)

	elif args.AnalysisType == 'CreateFrames':
		pp_obj.createAnnotationFrames()

	elif args.AnalysisType == 'MLClassification':
		pp_obj.runMLClusterClassifier()

	elif args.AnalysisType == 'MLFishDetection':
		pp_obj.runMLFishDetection()

	elif args.AnalysisType == 'Figures':
		pp_obj.runFigureCreation()

	elif args.AnalysisType == 'Backup':
		pp_obj.backupAnalysis()

if args.command == 'TotalProjectAnalysis':
	ap_obj = AP()
	if ap_obj.checkProjects(args.ProjectIDs):
		sys.exit()
	f = open('Analysis.log', 'w')
	for projectID in args.ProjectIDs:
		if args.Computer == 'SRG':
			print('Analyzing projectID: ' + projectID, file = f)
			downloadProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Download', projectID], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			
			print(downloadProcess.stdout, file = f)
			depthProcess = subprocess.Popen(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Depth', projectID, '-w', '1'], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			clusterProcess = subprocess.Popen(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Cluster', projectID, '-w', '23'], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			depthOut = depthProcess.communicate()
			clusterOut = clusterProcess.communicate()
			print(depthOut[0], file = f)
			print(clusterOut[0], file = f)
			mlProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'MLClassification', projectID], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			print(mlProcess.stdout, file = f)
			#print(mlProcess.stderr, file = f)
			
			error = False
			if depthOut[1] != '':
				print('DepthError: ' + depthOut[1])
				print('DepthError: ' + depthOut[1], file = f)
				error = True

			if clusterOut[1] != '': 
				print('ClusterError: ' + clusterOut[1])
				print('ClusterError: ' + clusterOut[1], file = f)
				error = True
			
			if mlProcess.stderr != '':
				print('MLError: ' + mlProcess.stderr)
				print('MLError: ' + mlProcess.stderr, file = f)
				error = True

			if error:
				f.close()
				sys.exit()

			backupProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Backup', projectID], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')

		elif args.Computer == 'NURF':
			print('Analyzing projectID: ' + projectID, file = f)
			downloadProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Download', projectID], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			print(downloadProcess.stdout, file = f)
			depthProcess = subprocess.Popen(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Depth', projectID, '-w', '1'], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			clusterProcess = subprocess.Popen(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Cluster', projectID, '-w', '23'], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
			depthOut = depthProcess.communicate()
			clusterOut = clusterProcess.communicate()
			print(depthOut[0], file = f)
			print(clusterOut[0], file = f)

			if depthOut[1] != '' or clusterOut[1] != '':
				print('DepthError: ' + depthOut[1])
				print('ClusterError: ' + clusterOut[1])
				sys.exit()


			backupProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'ProjectAnalysis', 'Backup', projectID], stderr = subprocess.PIPE, stdout = subprocess.PIPE, encoding = 'utf-8')
	f.close()	
	summarizeProcess = subprocess.run(['python3', 'CichlidBowerTracker.py', 'UpdateAnalysisSummary'])





