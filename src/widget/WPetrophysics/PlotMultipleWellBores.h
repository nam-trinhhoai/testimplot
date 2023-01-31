/*
 *
 *
 *  Created on: 18 Jul 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PlotMultipleWellBores_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PlotMultipleWellBores_H_

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
#include <string>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>


#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else

#endif

class PlotMultipleWellBores : public QtImGuiCore
{
public:
	PlotMultipleWellBores(WorkingSetManager* manager);

	virtual ~PlotMultipleWellBores();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

   // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Long crosshair cursor
	void longCrossHairCursor();
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PlotMultipleWellBores_H_
