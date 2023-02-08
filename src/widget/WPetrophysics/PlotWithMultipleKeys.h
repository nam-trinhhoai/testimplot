/*
 *
 *
 *  Created on: 21 Sept 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_

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

template<typename T>
inline T RandomRange(T min, T max);
inline ImVec4 RandomColor();

class PlotWithMultipleKeys : public QtImGuiCore
{
public:
	PlotWithMultipleKeys(WorkingSetManager* manager);
	typedef struct well_logs {
		std::vector <int> numpoints;
		std::vector<std::vector<double>> logs;
		std::vector<std::vector<double>> md;
		std::vector<std::vector<double>> tvd;
		std::vector<std::vector<double>> twt;
	}processed_logs;
	struct MyDndItem {
		int              Idx;
		int              Plt;
		int 			 chartIdx;
		ImVector<ImVec2> Data;
		ImVec4           Color;
		MyDndItem() {
			static int i = 0;
			Idx = i++;
			Plt = 0;
			chartIdx = 0;

			Color = RandomColor();

			// tmp solution to activate ImGuiPayLoad
			Data.reserve(2);
			for (int k = 0; k < 2; ++k) {
				float t = k;
				Data.push_back(ImVec2(t, Idx));
			}
		}
		void Reset() { Plt = 0; chartIdx = 0; }
	};

	virtual ~PlotWithMultipleKeys();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;
	FolderData* wells;
	FolderData* seismics;
	QList<IData*> iData;
	QList<IData*> iData_Seismic;
	bool linkDepthAxis = false;
	bool useLongCrossHair = false;
	WellUnit selectedWellUnit = WellUnit::MD;
	ImGuiComboFlags flags = 0;
	// Etablish a list of wellbores that have logs from selected case study
	std::vector<WellBore*> listWellBores;
	int totalNumberOfWellBores;
	// Etablish a list of seismic dataset from selected case study
	std::vector<Seismic3DAbstractDataset*> listSeismicDatasets;

	//Charts variable
	// Number of chart areas
	int numChartAreas;

	MyDndItem dnd[100]; // 100 is a limit number of logs
	MyDndItem dnd_seismic[100];// 100 is a limit number of seismic extractions

	// Number of depth points
	int numPoints;
	
	std::vector <well_logs> processed_wells;
	// Names of log curves		 
	std::vector<std::string> logNames;


	// Long crosshair cursor
	void longCrossHairCursor();
	void setting(ImGuiStyle& style);
	// Interactive helper
	int interactiveHelper(double* depth);

	typedef struct IJKPoint {
		int i;
		int j;
		int k;
	} IJKPoint;

	std::pair<bool, IJKPoint> isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore);
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
