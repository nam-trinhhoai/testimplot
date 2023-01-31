/*
 *
 *
 *  Created on: 26 Jul 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOGINTERACTIVE_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOGINTERACTIVE_H_

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
#include <tuple>
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

class PlotWellLogInteractive : public QtImGuiCore
{
public:
	PlotWellLogInteractive(WorkingSetManager* manager);

	virtual ~PlotWellLogInteractive();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

   // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Interactive Helper
	std::tuple<int, bool> interactiveHelper(Logs log);
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PLOTWELLLOGINTERACTIVE_H_
