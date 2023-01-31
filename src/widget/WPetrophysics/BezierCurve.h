/*
 *
 *
 *  Created on: 29 Aug 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_BEZIERCURVE_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_BEZIERCURVE_H_

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

class BezierCurve : public QtImGuiCore
{
public:
	BezierCurve(WorkingSetManager* manager);

	virtual ~BezierCurve();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

    // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;

	// Drag ellipse
	void plotBezierCurve(double &perimeterPoint_xMin, double &perimeterPoint_xMax, double &perimeterPoint_yMin, double &perimeterPoint_yMax, double log1Min, double log1Max, double log2Min, double log2Max);

};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_BEZIERCURVE_H_
