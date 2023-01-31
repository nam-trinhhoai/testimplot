/*
 *
 *
 *  Created on: 01 Aug 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVECROSSPLOTWITHHISTOGRAM_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVECROSSPLOTWITHHISTOGRAM_H_

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

class InteractiveCrossPlotWithHistogram : public QtImGuiCore
{
public:
	InteractiveCrossPlotWithHistogram(WorkingSetManager* manager);

	virtual ~InteractiveCrossPlotWithHistogram();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

   // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Interactive helper			
	int interactiveHelper(double *log1, double *log2, int numPoints);
	int interactiveHistogramHelper(int point_idx, double *log, int numPoints, int bins);

	// Index of active point on cross-plot chart
	int point_idx = -1;
	int bin1_idx = -1, bin2_idx = -1;

	bool const cumulative = false;
	bool const density = false;
	bool const outliers = true;
	int const bins = 100;
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_INTERACTIVECROSSPLOTWITHHISTOGRAM_H_
