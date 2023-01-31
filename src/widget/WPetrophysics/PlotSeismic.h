/*
 *
 *
 *  Created on: 06 Sept 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTSEISMIC_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTSEISMIC_H_

#include "QtImGuiCore.h"

#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"
#include "seismicsurvey.h"
#include "seismic3ddataset.h"
#include "viewutils.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"

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

class PlotSeismic : public QtImGuiCore
{
public:
	PlotSeismic(WorkingSetManager* manager);

	virtual ~PlotSeismic();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;

    // Number of depth points	
	int numPoints;	
	
	// Names of log curves		 
	std::vector<std::string> logNames;


    typedef struct IJKPoint {
        int i;
        int j;
        int k;
    } IJKPoint;

	std::pair<bool, IJKPoint> isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore);
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTSEISMIC_H_
