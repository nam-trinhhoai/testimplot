/*
 *
 *
 *  Created on: 01 Aug 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_SELECTMULTIPLEINTERVALS_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_SELECTMULTIPLEINTERVALS_H_

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

class SelectMultipleIntervals : public QtImGuiCore
{
public:
	SelectMultipleIntervals(WorkingSetManager* manager);

	virtual ~SelectMultipleIntervals();

	void showPlot() override;

private:
	// Data manager
	WorkingSetManager* m_manager;

   	// Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Log interval selector
	void dragLines();
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_SELECTMULTIPLEINTERVALS_H_
