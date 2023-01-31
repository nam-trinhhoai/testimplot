/*
 *  Created on: 01 Aug 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVEMULTIPLEWINDOWS_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVEMULTIPLEWINDOWS_H_

#include "QtImGuiCore.h"

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
#include <tuple>
#include <string>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>


#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else

#endif

class InteractiveMultipleWindows : public QtImGuiCore
{
public:
	InteractiveMultipleWindows(WorkingSetManager* manager);

	virtual ~InteractiveMultipleWindows();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

   // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Plot well logs
	void showHistogram();
	
	// Interactive helpers
	std::tuple<int, int, std::vector<double>, double, double> interactiveHistogramHelper(double *log, int numPoints, int bins);	
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVEMULTIPLEWINDOWS_H_
