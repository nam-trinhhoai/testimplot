/*
 *
 *
 *  Created on: 15 June 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOG_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOG_H_

#include "QtImGuiCore.h"

#include "WellLogsIO.h"

#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"

#include "ImGuiFileBrowser.h"
#include "implot.h"
#include "imgui.h"


#include "imgui_internal.h"

#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>


#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else

#endif

class PlotWellLog : public QtImGuiCore
{
public:
	PlotWellLog(WorkingSetManager* manager);

	virtual ~PlotWellLog();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

    // Setup window size and position
	void setWindowSize();

	// Setup plot style
	void showPlotStyleDialog();

	// Create the main menu
	void mainMenu();

	// Create the main layout
	void mainLayout();
	
	// Read LAS file
	imgui_addons::ImGuiFileBrowser file_dialog;
	
	// LAS file name
	std::string LASFileName;

	// LAS file content
	std::string fileContent;
	
	// Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// LAS Data is stored in a single vector line by line
	std::vector<std::string> logData;

	// Multiwell data
	std::vector<std::string> LASFileName_multiWells;	
	std::vector<int> numPoints_multiWells;
	std::vector<std::vector<std::string>> logNames_multiWells;	
	std::vector<std::vector<std::string>> logData_multiWells;	


	void showLASContentsDialog();
	
	// Plot well logs
	void showWellLogs_Tables();
	void showWellLogs_Subplots();
	void showWellLogs_QuickPlot();
	void showWellLogs_QuickPlot_DB();
	//void showLogLimits(std::string const logMin, std::string const logMax);
	void showWellLogs_dragAndDrop();

	// Show windows flags
	bool showImGuiDemo = false;
	bool showImPlotDemo = false;
	bool showPlotStyle = false;

	int showFileContentFlag = -1;
	bool showLASContents = false;
	bool showLogViewer = false;
	bool showLogViewer_table = false;
	bool showQuickPlot = false;
	bool showQuickPlot_DB = false;
	bool showLogViewer_dragAndDrop = false;

	bool showLongCrossHairCursor = true;

};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOG_H_
